//! The out-of-process runner behind this backend.
//!
//! nereid's Python (`main.py`) and C++ backends both run a model as a child
//! process that speaks one language-agnostic tensor contract:
//!
//! - **input** (optional): the raw tensor bytes on stdin, with
//!   `NEREID_INPUT_SHAPE` / `NEREID_INPUT_DTYPE` in the environment;
//! - **output**: a self-describing framed tensor written to the file named by
//!   `NEREID_OUTPUT_PATH` — a UTF-8 header line `"<dtype> <d0>,<d1>,...\n"`
//!   followed by the raw little-endian bytes.
//!
//! Only the launch command (and the one-time build step — a venv vs. a compile)
//! differ between the two, so the runner is written once and a backend just
//! hands [`run_subprocess_inference`] a configured [`Command`].
//!
//! For now it lives inside the C++ backend, because the C++ backend is its only
//! user: the Python backend carries its own copy of the same framing logic. A
//! follow-up should retrofit the Python path onto this runner, at which point
//! this module graduates to `crate::backend` alongside `native_common` — shared
//! machinery belongs in the core, but only once it is actually shared.

use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;

use tonic::Status;

/// A validated input tensor to feed a subprocess model. `bytes` are the tensor
/// values in row-major little-endian order for the element type named by
/// `dtype`; `shape` is the (already batch-normalized) tensor shape.
#[derive(Debug, Clone)]
pub struct TensorInput {
    pub shape: Vec<i64>,
    pub bytes: Vec<u8>,
    /// Canonical lowercase dtype name (e.g. `float32`, `int32`), advertised to
    /// the child via `NEREID_INPUT_DTYPE`.
    pub dtype: String,
}

/// Spawn `command`, and if `input` is present, write its bytes to the child's
/// stdin on a dedicated thread. Writing on a separate thread is what prevents a
/// deadlock when the tensor is larger than the OS pipe buffer and the child is
/// slow to (or never) drains it.
pub fn spawn_with_optional_stdin(
    mut command: Command,
    input: Option<TensorInput>,
) -> std::io::Result<(Child, Option<JoinHandle<()>>)> {
    let mut child = command.spawn()?;
    let stdin_handle = match input {
        Some(input) => child.stdin.take().map(|mut stdin| {
            std::thread::spawn(move || {
                let _ = stdin.write_all(&input.bytes);
                let _ = stdin.flush();
            })
        }),
        None => None,
    };
    Ok((child, stdin_handle))
}

/// A process-unique scratch path for a model's output tensor. Avoids
/// `Date`/random by combining the pid with a monotonic counter.
pub fn unique_output_path(model_name: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let safe: String = model_name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    std::env::temp_dir().join(format!("nereid-out-{safe}-{}-{n}.bin", std::process::id()))
}

/// Read a child pipe to EOF (so the process can't wedge on a full pipe),
/// retaining at most `DRAIN_CAP` bytes for diagnostics.
pub fn drain(stream: Option<impl Read>) -> Vec<u8> {
    const DRAIN_CAP: usize = 64 * 1024;
    let mut buf = Vec::new();
    if let Some(mut stream) = stream {
        let mut chunk = [0u8; 8192];
        loop {
            match stream.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if buf.len() < DRAIN_CAP {
                        let take = n.min(DRAIN_CAP - buf.len());
                        buf.extend_from_slice(&chunk[..take]);
                    }
                    // Keep reading past the cap to EOF so the child isn't wedged.
                }
            }
        }
    }
    buf
}

/// Parse the self-describing output tensor a model writes to
/// `NEREID_OUTPUT_PATH`: a UTF-8 header line `"<dtype> <d0>,<d1>,...\n"`
/// followed by the raw little-endian tensor bytes. Returns `(shape, bytes, dtype)`.
pub fn parse_framed_tensor(
    raw: &[u8],
    model_name: &str,
) -> Result<(Vec<i64>, Vec<u8>, String), Status> {
    let newline = raw.iter().position(|b| *b == b'\n').ok_or_else(|| {
        Status::internal(format!(
            "model '{model_name}' output is missing the framed header line"
        ))
    })?;
    let header = std::str::from_utf8(&raw[..newline]).map_err(|_| {
        Status::internal(format!(
            "model '{model_name}' output header is not valid UTF-8"
        ))
    })?;
    let data = &raw[newline + 1..];

    let mut parts = header.split_whitespace();
    let dtype = parts.next().unwrap_or_default();
    let (_kserve, elem_size) = crate::dtype::canonical_to_kserve(dtype).ok_or_else(|| {
        Status::internal(format!(
            "model '{model_name}' output dtype '{dtype}' is unsupported"
        ))
    })?;
    let dims_str = parts.next().ok_or_else(|| {
        Status::internal(format!(
            "model '{model_name}' output header is missing the shape: '{header}'"
        ))
    })?;

    let mut dims = Vec::new();
    for dim_str in dims_str.split(',').filter(|s| !s.is_empty()) {
        let dim = dim_str.parse::<i64>().map_err(|err| {
            Status::internal(format!(
                "model '{model_name}' output shape dimension '{dim_str}' is invalid: {err}"
            ))
        })?;
        if dim <= 0 {
            return Err(Status::internal(format!(
                "model '{model_name}' output shape dimensions must be positive, got {dim}"
            )));
        }
        dims.push(dim);
    }
    if dims.is_empty() {
        return Err(Status::internal(format!(
            "model '{model_name}' output header has an empty shape"
        )));
    }

    if !data.len().is_multiple_of(elem_size) {
        return Err(Status::internal(format!(
            "model '{model_name}' output byte length {} is not a multiple of {elem_size} ({dtype})",
            data.len()
        )));
    }
    let expected = dims
        .iter()
        .try_fold(1i64, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| Status::internal(format!("model '{model_name}' output shape overflow")))?;
    let actual = (data.len() / elem_size) as i64;
    if expected != actual {
        return Err(Status::internal(format!(
            "model '{model_name}' output size mismatch: header shape implies {expected} {dtype} \
             elements, file holds {actual}"
        )));
    }

    Ok((dims, data.to_vec(), dtype.to_string()))
}

/// Run a pre-configured subprocess model for one inference request and return
/// its framed output tensor — the Triton `ModelInfer` path.
///
/// The caller sets the executable, args, and working directory on `command`;
/// this function adds the shared `NEREID_*` environment + stdin wiring, runs the
/// child to completion (draining stdout/stderr so it can't wedge), and parses
/// the framed tensor the model wrote to `NEREID_OUTPUT_PATH`. Blocking — run it
/// off the async runtime (e.g. `tokio::task::spawn_blocking`).
pub fn run_subprocess_inference(
    mut command: Command,
    model_name: &str,
    input: Option<TensorInput>,
) -> Result<(Vec<i64>, Vec<u8>, String), Status> {
    let output_path = unique_output_path(model_name);
    // The output dtype hint defaults to float32 for an output-only model.
    let output_dtype = input
        .as_ref()
        .map(|i| i.dtype.clone())
        .unwrap_or_else(|| "float32".to_string());

    command
        .env("NEREID_OUTPUT_PATH", &output_path)
        .env("NEREID_OUTPUT_DTYPE", &output_dtype)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // A model with a declared input receives the validated tensor on stdin;
    // an output-only model gets no input (stdin closed).
    if let Some(input) = &input {
        let shape = input
            .shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        command
            .env("NEREID_INPUT_SHAPE", shape)
            .env("NEREID_INPUT_DTYPE", &input.dtype)
            .stdin(Stdio::piped());
    } else {
        command.stdin(Stdio::null());
    }

    let (mut child, stdin_handle) = spawn_with_optional_stdin(command, input)
        .map_err(|err| Status::internal(format!("failed to run model '{model_name}': {err}")))?;

    // Drain both pipes concurrently so a chatty model can't wedge the child by
    // filling a pipe buffer while we wait on the other stream.
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdout_handle = std::thread::spawn(move || drain(stdout));
    let stderr_handle = std::thread::spawn(move || drain(stderr));

    let status = child.wait();
    let _ = stdout_handle.join();
    let stderr_bytes = stderr_handle.join().unwrap_or_default();
    if let Some(handle) = stdin_handle {
        let _ = handle.join();
    }

    let status = status.map_err(|err| {
        let _ = fs::remove_file(&output_path);
        Status::internal(format!("failed waiting on model '{model_name}': {err}"))
    })?;

    let stderr_text = String::from_utf8_lossy(&stderr_bytes);
    let stderr_text = stderr_text.trim();
    let stderr_suffix = |prefix: &str| {
        if stderr_text.is_empty() {
            String::new()
        } else {
            format!("{prefix}{stderr_text}")
        }
    };

    if !status.success() {
        let _ = fs::remove_file(&output_path);
        let code = status
            .code()
            .map(|c| c.to_string())
            .unwrap_or_else(|| "signal".to_string());
        return Err(Status::internal(format!(
            "model '{model_name}' exited with status {code}{}",
            stderr_suffix(": ")
        )));
    }

    // Read then remove unconditionally (best-effort) so the temp file never
    // leaks — on read success, read failure, or a later parse failure.
    let raw = fs::read(&output_path);
    let _ = fs::remove_file(&output_path);
    let raw = raw.map_err(|err| {
        Status::failed_precondition(format!(
            "model '{model_name}' declares a tensor output contract but wrote no readable tensor \
             to NEREID_OUTPUT_PATH ({err}){}",
            stderr_suffix("; stderr: ")
        ))
    })?;

    parse_framed_tensor(&raw, model_name)
}
