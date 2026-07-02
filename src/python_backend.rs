use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::mpsc;
use tonic::Status;

use crate::model_runtime::InputShapeContract;
use crate::proto::{CheckpointResponse, TensorChunk};

pub type CheckpointStream =
    tonic::codegen::tokio_stream::wrappers::ReceiverStream<Result<CheckpointResponse, Status>>;

/// Emit a Python model's output tensor back over the checkpoint stream in the
/// same 64 KiB-chunked form the Rust path uses.
const OUTPUT_CHUNK_BYTES: usize = 64 * 1024;

/// A process-unique scratch path for a Python model's output tensor. Avoids
/// `Date`/random by combining the pid with a monotonic counter.
fn unique_output_path(model_name: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let safe: String = model_name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    std::env::temp_dir().join(format!("nereid-out-{safe}-{}-{n}.bin", std::process::id()))
}

/// Parse the framed output tensor a Python model writes to `NEREID_OUTPUT_PATH`:
/// a UTF-8 header line `"float32 d0,d1,...\n"` followed by raw little-endian
/// float32 bytes. Returns `(shape, bytes)`.
fn parse_framed_tensor(raw: &[u8], model_name: &str) -> Result<(Vec<i64>, Vec<u8>), Status> {
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
    if dtype != "float32" {
        return Err(Status::internal(format!(
            "model '{model_name}' output dtype '{dtype}' is unsupported; only float32 is supported"
        )));
    }
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

    if !data.len().is_multiple_of(4) {
        return Err(Status::internal(format!(
            "model '{model_name}' output byte length {} is not a multiple of 4 (float32)",
            data.len()
        )));
    }
    let expected = dims
        .iter()
        .try_fold(1i64, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| Status::internal(format!("model '{model_name}' output shape overflow")))?;
    let actual = (data.len() / 4) as i64;
    if expected != actual {
        return Err(Status::internal(format!(
            "model '{model_name}' output size mismatch: header shape implies {expected} float32 \
             elements, file holds {actual}"
        )));
    }

    Ok((dims, data.to_vec()))
}

/// A validated input tensor to feed `main.py`. The raw `bytes` are the
/// little-endian `float32` tensor values; `shape` is the (already
/// batch-normalized) tensor shape. Delivered to the subprocess on stdin, with
/// the shape and dtype exposed via the `NEREID_INPUT_SHAPE` /
/// `NEREID_INPUT_DTYPE` environment variables.
#[derive(Debug, Clone)]
pub struct PythonInput {
    pub shape: Vec<i64>,
    pub bytes: Vec<u8>,
}

pub fn venv_python_path(venv_dir: &Path) -> PathBuf {
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
        "no command output".to_string()
    }
}

pub fn prepare_model_envs(
    model_names: &[String],
    ml_backends_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    for model_name in model_names {
        let model_dir = fs::canonicalize(ml_backends_path.join(model_name))?;
        let requirements = model_dir.join("requirements.txt");
        if !requirements.is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("requirements.txt not found for model '{model_name}'"),
            )
            .into());
        }

        let venv_dir = model_dir.join("venv");
        if venv_dir.exists() && !venv_dir.is_dir() {
            return Err(std::io::Error::other(format!(
                "path exists but is not a directory: {}",
                venv_dir.display()
            ))
            .into());
        }

        if !venv_dir.is_dir() {
            let create_venv = Command::new("python3")
                .args(["-m", "venv", "venv"])
                .current_dir(&model_dir)
                .output()?;
            if !create_venv.status.success() {
                return Err(std::io::Error::other(format!(
                    "failed to create venv for model '{model_name}': {}",
                    output_details(&create_venv)
                ))
                .into());
            }
        }

        let pip_path = venv_pip_path(&venv_dir);
        if !pip_path.is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("venv pip not found for model '{model_name}'"),
            )
            .into());
        }

        let install = Command::new(&pip_path)
            .args(["install", "-r", "requirements.txt"])
            .current_dir(&model_dir)
            .output()?;
        if !install.status.success() {
            return Err(std::io::Error::other(format!(
                "failed to install requirements for model '{model_name}': {}",
                output_details(&install)
            ))
            .into());
        }
    }

    Ok(())
}

pub fn spawn_python_checkpoint_stream(
    model_name: &str,
    model_dir: PathBuf,
    input: Option<PythonInput>,
    contract: InputShapeContract,
    expected_batch: Option<i64>,
) -> Result<CheckpointStream, Status> {
    let main_py = model_dir.join("main.py");
    if !main_py.is_file() {
        return Err(Status::not_found(format!(
            "main.py not found for model '{model_name}'"
        )));
    }

    let venv_dir = model_dir.join("venv");
    let python_path = venv_python_path(&venv_dir);
    if !python_path.is_file() {
        return Err(Status::not_found(format!(
            "venv python not found for model '{model_name}'"
        )));
    }

    let model_name = model_name.to_string();
    let (tx, rx) = mpsc::channel::<Result<CheckpointResponse, Status>>(64);

    std::thread::spawn(move || {
        // Every Python reply is a typed tensor the model writes here.
        let output_path = unique_output_path(&model_name);

        let mut command = Command::new(&python_path);
        command
            .arg("-u")
            .arg("main.py")
            .current_dir(&model_dir)
            .env("NEREID_OUTPUT_PATH", &output_path)
            .env("NEREID_OUTPUT_DTYPE", "float32")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // When an input tensor is supplied it is delivered on stdin (raw
        // little-endian float32), with the shape/dtype advertised via env vars.
        // Otherwise stdin is closed so `main.py` reads EOF immediately.
        if let Some(input) = &input {
            let shape = input
                .shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            command
                .env("NEREID_INPUT_SHAPE", shape)
                .env("NEREID_INPUT_DTYPE", "float32")
                .stdin(Stdio::piped());
        } else {
            command.stdin(Stdio::null());
        }

        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(err) => {
                let _ = tx.blocking_send(Err(Status::internal(format!(
                    "failed to run main.py: {err}"
                ))));
                return;
            }
        };

        // Stream the input tensor to stdin on a dedicated thread so it runs
        // concurrently with the stdout/stderr readers (a single-threaded
        // write-then-read would deadlock once the OS pipe buffer fills). Write
        // errors (e.g. a `main.py` that ignores stdin and exits) are ignored;
        // dropping the handle at the end signals EOF.
        let stdin_handle = match input {
            Some(input) => child.stdin.take().map(|mut stdin| {
                std::thread::spawn(move || {
                    let _ = stdin.write_all(&input.bytes);
                    let _ = stdin.flush();
                })
            }),
            None => None,
        };

        let stdout = match child.stdout.take() {
            Some(stdout) => stdout,
            None => {
                let _ = tx.blocking_send(Err(Status::internal(
                    "failed to capture stdout from main.py process",
                )));
                return;
            }
        };
        let stderr = match child.stderr.take() {
            Some(stderr) => stderr,
            None => {
                let _ = tx.blocking_send(Err(Status::internal(
                    "failed to capture stderr from main.py process",
                )));
                return;
            }
        };

        let tx_stdout = tx.clone();
        let stdout_handle = std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                match line {
                    Ok(line) => {
                        if tx_stdout
                            .blocking_send(Ok(CheckpointResponse {
                                chunk: line,
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
                        let _ = tx_stdout.blocking_send(Err(Status::internal(format!(
                            "failed reading stdout: {err}"
                        ))));
                        break;
                    }
                }
            }
        });

        let tx_stderr = tx.clone();
        let stderr_handle = std::thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                match line {
                    Ok(line) => {
                        if tx_stderr
                            .blocking_send(Ok(CheckpointResponse {
                                chunk: format!("stderr: {line}"),
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
                        let _ = tx_stderr.blocking_send(Err(Status::internal(format!(
                            "failed reading stderr: {err}"
                        ))));
                        break;
                    }
                }
            }
        });

        let status = match child.wait() {
            Ok(status) => status,
            Err(err) => {
                let _ = tx.blocking_send(Err(Status::internal(format!(
                    "failed waiting on main.py process for model '{model_name}': {err}"
                ))));
                return;
            }
        };

        let _ = stdout_handle.join();
        let _ = stderr_handle.join();
        if let Some(stdin_handle) = stdin_handle {
            let _ = stdin_handle.join();
        }

        let exit_code = status.code().unwrap_or(-1);
        if !status.success() {
            // On failure there is no valid tensor to return; the text chunks
            // already streamed carry the diagnostics.
            let _ = fs::remove_file(&output_path);
            let _ = tx.blocking_send(Ok(CheckpointResponse {
                chunk: String::new(),
                done: true,
                exit_code,
                output_chunk: None,
            }));
            return;
        }

        // Success: read, validate, and stream the model's output tensor before
        // the terminal `done`. A model that exits 0 without a valid tensor is a
        // contract violation (every Python reply must be a tensor).
        let tensor = fs::read(&output_path).map_err(|err| {
            Status::failed_precondition(format!(
                "Python model '{model_name}' exited 0 but wrote no readable output tensor to NEREID_OUTPUT_PATH: {err}"
            ))
        });
        let _ = fs::remove_file(&output_path);
        let tensor = tensor
            .and_then(|raw| parse_framed_tensor(&raw, &model_name))
            .and_then(|(shape, bytes)| {
                contract
                    .validate_output_shape(&shape, expected_batch, &model_name)
                    .map(|()| (shape, bytes))
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

    Ok(tonic::codegen::tokio_stream::wrappers::ReceiverStream::new(
        rx,
    ))
}

/// Stream a tensor back as one or more `output_chunk` messages (64 KiB each),
/// mirroring the Rust inference path. Returns false if the receiver dropped.
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
