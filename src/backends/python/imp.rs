use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;

use tokio::sync::mpsc;
use tonic::Status;

use crate::backend::{Backend, CheckpointStream, Contract, Tensor};
use crate::config::ModelConfig;
use crate::proto::{CheckpointResponse, TensorChunk};

const OUTPUT_CHUNK_BYTES: usize = 64 * 1024;

pub struct PythonBackend {
    model_name: String,
    model_dir: PathBuf,
}

impl PythonBackend {
    pub fn load(
        model_dir: &Path,
        model_cfg: &ModelConfig,
    ) -> Result<(Box<dyn Backend>, Contract), Status> {
        prepare_venv(model_dir, &model_cfg.name)?;

        if !model_dir.join("model_inference.textproto").is_file() {
            return Err(Status::failed_precondition(format!(
                "Python model '{}' must include a model_inference.textproto declaring output_shape",
                model_cfg.name
            )));
        }
        let contract = Contract::parse(model_dir)?;
        if contract.outputs.is_empty() {
            return Err(Status::failed_precondition(format!(
                "Python model '{}' must declare output_shape in model_inference.textproto (every reply is a typed tensor)",
                model_cfg.name
            )));
        }

        Ok((
            Box::new(PythonBackend {
                model_name: model_cfg.name.clone(),
                model_dir: model_dir.to_path_buf(),
            }),
            contract,
        ))
    }
}

impl Backend for PythonBackend {
    fn platform(&self) -> &'static str {
        "python"
    }

    fn infer(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, Status> {
        let input = inputs.into_iter().next();
        let (shape, data, dtype) = run_python_inference(&self.model_name, &self.model_dir, input)?;
        Ok(vec![Tensor {
            name: "output".to_string(),
            shape,
            dtype,
            data,
        }])
    }

    fn checkpoint_stream(
        &self,
        model_name: &str,
        input: Option<Tensor>,
        contract: &Contract,
    ) -> Option<CheckpointStream> {
        Some(spawn_python_checkpoint_stream(
            model_name,
            self.model_dir.clone(),
            input,
            contract.clone(),
        ))
    }
}

fn venv_python_path(venv_dir: &Path) -> PathBuf {
    if cfg!(windows) {
        venv_dir.join("Scripts").join("python.exe")
    } else {
        venv_dir.join("bin").join("python")
    }
}

fn venv_pip_path(venv_dir: &Path) -> PathBuf {
    if cfg!(windows) {
        venv_dir.join("Scripts").join("pip.exe")
    } else {
        venv_dir.join("bin").join("pip")
    }
}

fn output_details(output: &std::process::Output) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        "no output".to_string()
    }
}

/// Create (if missing) and populate a model's virtualenv from requirements.txt.
fn prepare_venv(model_dir: &Path, model_name: &str) -> Result<(), Status> {
    // Serialize venv creation process-wide: several `ModelManager`s can build
    // concurrently (parallel tests) and race `python3 -m venv` on one dir.
    static PREPARE_LOCK: Mutex<()> = Mutex::new(());
    let _guard = PREPARE_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let requirements = model_dir.join("requirements.txt");
    if !requirements.is_file() {
        return Err(Status::failed_precondition(format!(
            "requirements.txt not found for model '{model_name}'"
        )));
    }
    let venv_dir = model_dir.join("venv");
    if venv_dir.exists() && !venv_dir.is_dir() {
        return Err(Status::failed_precondition(format!(
            "path exists but is not a directory: {}",
            venv_dir.display()
        )));
    }
    if !venv_dir.is_dir() {
        let create = Command::new("python3")
            .args(["-m", "venv", "venv"])
            .current_dir(model_dir)
            .output()
            .map_err(|err| Status::failed_precondition(format!("failed to run python3: {err}")))?;
        if !create.status.success() {
            return Err(Status::failed_precondition(format!(
                "failed to create venv for model '{model_name}': {}",
                output_details(&create)
            )));
        }
    }
    let pip_path = venv_pip_path(&venv_dir);
    if !pip_path.is_file() {
        return Err(Status::failed_precondition(format!(
            "venv pip not found for model '{model_name}'"
        )));
    }
    let install = Command::new(&pip_path)
        .args(["install", "-r", "requirements.txt"])
        .current_dir(model_dir)
        .output()
        .map_err(|err| Status::failed_precondition(format!("failed to run pip: {err}")))?;
    if !install.status.success() {
        return Err(Status::failed_precondition(format!(
            "failed to install requirements for model '{model_name}': {}",
            output_details(&install)
        )));
    }
    Ok(())
}

/// Spawn `command`, and if `input` is present, write its bytes to stdin on a
/// dedicated thread (prevents a deadlock when the tensor exceeds the pipe buffer).
fn spawn_with_optional_stdin(
    mut command: Command,
    input_bytes: Option<Vec<u8>>,
) -> std::io::Result<(Child, Option<JoinHandle<()>>)> {
    let mut child = command.spawn()?;
    let stdin_handle = match input_bytes {
        Some(bytes) => child.stdin.take().map(|mut stdin| {
            std::thread::spawn(move || {
                let _ = stdin.write_all(&bytes);
                let _ = stdin.flush();
            })
        }),
        None => None,
    };
    Ok((child, stdin_handle))
}

fn unique_output_path(model_name: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let safe: String = model_name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    std::env::temp_dir().join(format!("nereid-out-{safe}-{}-{n}.bin", std::process::id()))
}

fn drain(stream: Option<impl Read>) -> Vec<u8> {
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
                }
            }
        }
    }
    buf
}

/// Parse the framed output tensor a model writes to `NEREID_OUTPUT_PATH`:
/// `"<dtype> d0,d1,...\n"` + raw little-endian bytes.
fn parse_framed_tensor(
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
            "model '{model_name}' output size mismatch: header shape implies {expected} {dtype} elements, file holds {actual}"
        )));
    }
    Ok((dims, data.to_vec(), dtype.to_string()))
}

/// Build the `python -u main.py` command for a model.
fn python_command(model_dir: &Path, model_name: &str) -> Result<Command, Status> {
    if !model_dir.join("main.py").is_file() {
        return Err(Status::not_found(format!(
            "main.py not found for model '{model_name}'"
        )));
    }
    let python_path = venv_python_path(&model_dir.join("venv"));
    if !python_path.is_file() {
        return Err(Status::not_found(format!(
            "venv python not found for model '{model_name}'"
        )));
    }
    let mut command = Command::new(&python_path);
    command.arg("-u").arg("main.py").current_dir(model_dir);
    Ok(command)
}

/// Run a model for one unary request; return `(shape, bytes, canonical dtype)`.
fn run_python_inference(
    model_name: &str,
    model_dir: &Path,
    input: Option<Tensor>,
) -> Result<(Vec<i64>, Vec<u8>, String), Status> {
    let output_path = unique_output_path(model_name);
    let output_dtype = input
        .as_ref()
        .map(|t| t.dtype.clone())
        .unwrap_or_else(|| "float32".to_string());

    let mut command = python_command(model_dir, model_name)?;
    command
        .env("NEREID_OUTPUT_PATH", &output_path)
        .env("NEREID_OUTPUT_DTYPE", &output_dtype)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let input_bytes = if let Some(t) = &input {
        let shape = t
            .shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        command
            .env("NEREID_INPUT_SHAPE", shape)
            .env("NEREID_INPUT_DTYPE", &t.dtype)
            .stdin(Stdio::piped());
        Some(t.data.clone())
    } else {
        command.stdin(Stdio::null());
        None
    };

    let (mut child, stdin_handle) =
        spawn_with_optional_stdin(command, input_bytes).map_err(|err| {
            Status::internal(format!(
                "failed to run main.py for model '{model_name}': {err}"
            ))
        })?;
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
        let _ = std::fs::remove_file(&output_path);
        Status::internal(format!(
            "failed waiting on main.py for model '{model_name}': {err}"
        ))
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
        let _ = std::fs::remove_file(&output_path);
        let code = status
            .code()
            .map(|c| c.to_string())
            .unwrap_or_else(|| "signal".to_string());
        return Err(Status::internal(format!(
            "main.py for model '{model_name}' exited with status {code}{}",
            stderr_suffix(": ")
        )));
    }

    let raw = std::fs::read(&output_path);
    let _ = std::fs::remove_file(&output_path);
    let raw = raw.map_err(|err| {
        Status::failed_precondition(format!(
            "model '{model_name}' declares a tensor output contract but wrote no readable tensor to NEREID_OUTPUT_PATH ({err}){}",
            stderr_suffix("; stderr: ")
        ))
    })?;
    parse_framed_tensor(&raw, model_name)
}

/// The streaming Checkpoint path: stream stdout/stderr log lines, then the
/// validated output tensor. Float32-only (the Checkpoint chunk carries no dtype).
fn spawn_python_checkpoint_stream(
    model_name: &str,
    model_dir: PathBuf,
    input: Option<Tensor>,
    contract: Contract,
) -> CheckpointStream {
    let model_name = model_name.to_string();
    let (tx, rx) = mpsc::channel::<Result<CheckpointResponse, Status>>(64);

    let expected_batch = input
        .as_ref()
        .and_then(|t| contract.expected_batch(&t.shape));

    std::thread::spawn(move || {
        let output_path = unique_output_path(&model_name);
        let mut command = match python_command(&model_dir, &model_name) {
            Ok(c) => c,
            Err(status) => {
                let _ = tx.blocking_send(Err(status));
                return;
            }
        };
        command
            .env("NEREID_OUTPUT_PATH", &output_path)
            .env("NEREID_OUTPUT_DTYPE", "float32")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let input_bytes = if let Some(t) = &input {
            let shape = t
                .shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            command
                .env("NEREID_INPUT_SHAPE", shape)
                .env("NEREID_INPUT_DTYPE", &t.dtype)
                .stdin(Stdio::piped());
            Some(t.data.clone())
        } else {
            command.stdin(Stdio::null());
            None
        };

        let (mut child, stdin_handle) = match spawn_with_optional_stdin(command, input_bytes) {
            Ok(parts) => parts,
            Err(err) => {
                let _ = tx.blocking_send(Err(Status::internal(format!(
                    "failed to run main.py: {err}"
                ))));
                return;
            }
        };
        let stdout = match child.stdout.take() {
            Some(s) => s,
            None => {
                let _ = tx.blocking_send(Err(Status::internal(
                    "failed to capture stdout from main.py process",
                )));
                return;
            }
        };
        let stderr = match child.stderr.take() {
            Some(s) => s,
            None => {
                let _ = tx.blocking_send(Err(Status::internal(
                    "failed to capture stderr from main.py process",
                )));
                return;
            }
        };

        let tx_out = tx.clone();
        let stdout_handle = std::thread::spawn(move || stream_lines(tx_out, stdout, false));
        let tx_err = tx.clone();
        let stderr_handle = std::thread::spawn(move || stream_lines(tx_err, stderr, true));

        let status = match child.wait() {
            Ok(s) => s,
            Err(err) => {
                let _ = tx.blocking_send(Err(Status::internal(format!(
                    "failed waiting on main.py process for model '{model_name}': {err}"
                ))));
                return;
            }
        };
        let _ = stdout_handle.join();
        let _ = stderr_handle.join();
        if let Some(h) = stdin_handle {
            let _ = h.join();
        }

        let exit_code = status.code().unwrap_or(-1);
        if !status.success() {
            let _ = std::fs::remove_file(&output_path);
            let _ = tx.blocking_send(Ok(CheckpointResponse {
                chunk: String::new(),
                done: true,
                exit_code,
                output_chunk: None,
            }));
            return;
        }

        let tensor = std::fs::read(&output_path).map_err(|err| {
            Status::failed_precondition(format!(
                "Python model '{model_name}' exited 0 but wrote no readable output tensor to NEREID_OUTPUT_PATH: {err}"
            ))
        });
        let _ = std::fs::remove_file(&output_path);
        let tensor = tensor
            .and_then(|raw| parse_framed_tensor(&raw, &model_name))
            .and_then(|(shape, bytes, dtype)| {
                if dtype != "float32" {
                    return Err(Status::failed_precondition(format!(
                        "Python model '{model_name}' wrote a '{dtype}' output tensor, but the Checkpoint path only supports float32"
                    )));
                }
                let output_spec = contract.outputs.first();
                if let Some(spec) = output_spec {
                    contract.validate_output_shape(spec, &shape, expected_batch, &model_name)?;
                }
                Ok((shape, bytes))
            });

        match tensor {
            Ok((shape, bytes)) => {
                if emit_output_tensor(&tx, &shape, &bytes) {
                    let _ = tx.blocking_send(Ok(CheckpointResponse {
                        chunk: String::new(),
                        done: true,
                        exit_code,
                        output_chunk: None,
                    }));
                }
            }
            Err(status) => {
                let _ = tx.blocking_send(Err(status));
            }
        }
    });

    CheckpointStream::new(rx)
}

fn stream_lines(
    tx: mpsc::Sender<Result<CheckpointResponse, Status>>,
    stream: impl Read,
    is_stderr: bool,
) {
    let reader = BufReader::new(stream);
    for line in reader.lines() {
        match line {
            Ok(line) => {
                let chunk = if is_stderr {
                    format!("stderr: {line}")
                } else {
                    line
                };
                if tx
                    .blocking_send(Ok(CheckpointResponse {
                        chunk,
                        done: false,
                        exit_code: 0,
                        output_chunk: None,
                    }))
                    .is_err()
                {
                    break;
                }
            }
            Err(err) => {
                let _ = tx.blocking_send(Err(Status::internal(format!(
                    "failed reading output: {err}"
                ))));
                break;
            }
        }
    }
}

fn emit_output_tensor(
    tx: &mpsc::Sender<Result<CheckpointResponse, Status>>,
    shape: &[i64],
    bytes: &[u8],
) -> bool {
    let chunk_response = |data: Vec<u8>, chunk_index: u64, end_of_tensor: bool| {
        Ok(CheckpointResponse {
            chunk: String::new(),
            done: false,
            exit_code: 0,
            output_chunk: Some(TensorChunk {
                tensor_name: "output".to_string(),
                shape: shape.to_vec(),
                data,
                chunk_index,
                end_of_tensor,
            }),
        })
    };
    if bytes.is_empty() {
        return tx
            .blocking_send(chunk_response(Vec::new(), 0, true))
            .is_ok();
    }
    let num_chunks = bytes.len().div_ceil(OUTPUT_CHUNK_BYTES);
    for (chunk_index, data_chunk) in bytes.chunks(OUTPUT_CHUNK_BYTES).enumerate() {
        let end_of_tensor = chunk_index + 1 == num_chunks;
        if tx
            .blocking_send(chunk_response(
                data_chunk.to_vec(),
                chunk_index as u64,
                end_of_tensor,
            ))
            .is_err()
        {
            return false;
        }
    }
    true
}
