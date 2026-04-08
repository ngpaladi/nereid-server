use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tch::Tensor;
use tokio::sync::mpsc;
use tonic::{Request, Response, Status, transport::Server};

mod inference;

use proto::health_server::{Health, HealthServer};
use proto::sonic_server::{Sonic, SonicServer};
use proto::{
    CheckpointRequest, CheckpointResponse, HealthCheckRequest, HealthCheckResponse,
    ViewModelsRequest, ViewModelsResponse, checkpoint_request::Payload,
};

pub mod proto {
    tonic::include_proto!("inference");
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
        "no command output".to_string()
    }
}

fn prepare_model_envs() -> Result<(), Box<dyn std::error::Error>> {
    let models = get_model_names()
        .map_err(|status| std::io::Error::new(std::io::ErrorKind::Other, status.to_string()))?;

    for model_name in models {
        let model_dir = fs::canonicalize(Path::new("ml-backends").join(&model_name))?;
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
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("path exists but is not a directory: {}", venv_dir.display()),
            )
            .into());
        }

        if !venv_dir.is_dir() {
            let create_venv = Command::new("python3")
                .args(["-m", "venv", "venv"])
                .current_dir(&model_dir)
                .output()?;
            if !create_venv.status.success() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!(
                        "failed to create venv for model '{model_name}': {}",
                        output_details(&create_venv)
                    ),
                )
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
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!(
                    "failed to install requirements for model '{model_name}': {}",
                    output_details(&install)
                ),
            )
            .into());
        }
    }

    Ok(())
}

fn get_model_names() -> Result<Vec<String>, Status> {
    let entries = match fs::read_dir("ml-backends") {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => {
            return Err(Status::internal(format!(
                "failed to read model directory 'ml-backends': {err}"
            )));
        }
    };

    let mut model_names = Vec::new();
    for entry in entries {
        let entry =
            entry.map_err(|err| Status::internal(format!("failed to read model entry: {err}")))?;
        let file_type = entry.file_type().map_err(|err| {
            Status::internal(format!("failed to inspect model entry type: {err}"))
        })?;

        if file_type.is_dir() {
            model_names.push(entry.file_name().to_string_lossy().into_owned());
        }
    }

    model_names.sort();
    Ok(model_names)
}

fn find_pt_model_file(model_dir: &Path) -> Result<Option<PathBuf>, Status> {
    let mut pt_files = Vec::new();
    let entries = fs::read_dir(model_dir)
        .map_err(|err| Status::internal(format!("failed to read model directory: {err}")))?;

    for entry in entries {
        let entry =
            entry.map_err(|err| Status::internal(format!("failed to read model entry: {err}")))?;
        let path = entry.path();
        if path.is_file() && path.extension().is_some_and(|ext| ext == "pt") {
            pt_files.push(path);
        }
    }

    pt_files.sort();
    Ok(pt_files.into_iter().next())
}

fn read_input_shape_from_textproto(model_dir: &Path) -> Result<Vec<i64>, Status> {
    let config_path = model_dir.join("model_inference.textproto");
    let contents = fs::read_to_string(&config_path).map_err(|err| {
        Status::internal(format!(
            "failed to read {}: {err}",
            config_path.to_string_lossy()
        ))
    })?;

    let mut shape = Vec::new();
    for raw_line in contents.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() || !line.starts_with("input_shape") {
            continue;
        }

        let (_, raw_value) = line.split_once(':').ok_or_else(|| {
            Status::failed_precondition(format!(
                "invalid input_shape line in {}: '{raw_line}'",
                config_path.to_string_lossy()
            ))
        })?;

        let dim = raw_value
            .trim()
            .trim_matches('"')
            .parse::<i64>()
            .map_err(|err| {
                Status::failed_precondition(format!(
                    "failed parsing input_shape in {}: {err}",
                    config_path.to_string_lossy()
                ))
            })?;

        if dim <= 0 {
            return Err(Status::failed_precondition(format!(
                "input_shape dimensions in {} must be positive",
                config_path.to_string_lossy()
            )));
        }
        shape.push(dim);
    }

    if shape.is_empty() {
        return Err(Status::failed_precondition(format!(
            "{} must contain at least one input_shape field",
            config_path.to_string_lossy()
        )));
    }

    Ok(shape)
}

fn run_rust_inference(
    model_name: &str,
    model_dir: &Path,
    pt_file: &Path,
    tensor_bytes: &[u8],
) -> Result<(), Status> {
    if tensor_bytes.is_empty() {
        return Err(Status::invalid_argument(
            "no tensor chunk data provided for Rust inference model",
        ));
    }
    if tensor_bytes.len() % 4 != 0 {
        return Err(Status::invalid_argument(
            "tensor chunk bytes must be a multiple of 4 for float32",
        ));
    }

    let shape = read_input_shape_from_textproto(model_dir)?;
    let values: Vec<f32> = tensor_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let expected_numel = shape
        .iter()
        .try_fold(1i64, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| Status::invalid_argument("tensor shape overflow"))?;

    if expected_numel != i64::try_from(values.len()).unwrap_or(-1) {
        return Err(Status::invalid_argument(format!(
            "input tensor size mismatch for model '{model_name}': expected {expected_numel} values from model_inference.textproto, got {}",
            values.len()
        )));
    }

    let tensor = Tensor::f_from_slice(&values)
        .and_then(|t| t.f_reshape(shape.as_slice()))
        .map_err(|err| {
            Status::invalid_argument(format!(
                "failed to build input tensor from stream chunks: {err}"
            ))
        })?;

    let model_path = pt_file.to_string_lossy().into_owned();
    inference::run_forward_pass(&model_path, &tensor)
        .map_err(|err| Status::internal(format!("Rust inference failed: {err}")))?;
    Ok(())
}

#[derive(Debug, Default)]
pub struct HealthService;

#[tonic::async_trait]
impl Health for HealthService {
    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        Ok(Response::new(HealthCheckResponse {
            status: "ok".to_string(),
        }))
    }
}

#[derive(Debug, Default)]
pub struct SonicService;

#[tonic::async_trait]
impl Sonic for SonicService {
    type CheckpointStream =
        tonic::codegen::tokio_stream::wrappers::ReceiverStream<Result<CheckpointResponse, Status>>;

    async fn view_models(
        &self,
        _request: Request<ViewModelsRequest>,
    ) -> Result<Response<ViewModelsResponse>, Status> {
        let mut model_names = get_model_names()?;
        if model_names.is_empty() {
            model_names.push("No models found".to_string());
        }

        Ok(Response::new(ViewModelsResponse { model_names }))
    }

    async fn checkpoint(
        &self,
        request: Request<tonic::Streaming<CheckpointRequest>>,
    ) -> Result<Response<Self::CheckpointStream>, Status> {
        let mut stream = request.into_inner();
        let first_message = stream.message().await.map_err(|err| {
            Status::internal(format!(
                "failed to read first checkpoint stream message: {err}"
            ))
        })?;
        let first_message = first_message.ok_or_else(|| {
            Status::invalid_argument(
                "checkpoint stream is empty; first message must include metadata",
            )
        })?;

        let meta = match first_message.payload {
            Some(Payload::Meta(meta)) => meta,
            Some(Payload::Chunk(_)) => {
                return Err(Status::invalid_argument(
                    "first checkpoint stream message must be metadata",
                ));
            }
            None => {
                return Err(Status::invalid_argument(
                    "first checkpoint stream message has no payload",
                ));
            }
        };

        let model_name = meta.model_name.trim().to_string();

        if model_name.is_empty() {
            return Err(Status::invalid_argument("model_name is required"));
        }

        let model_names = get_model_names()?;
        if !model_names.iter().any(|name| name == &model_name) {
            return Err(Status::not_found(format!(
                "model '{model_name}' was not found in ml-backends"
            )));
        }

        let model_dir = fs::canonicalize(Path::new("ml-backends").join(&model_name))
            .map_err(|err| Status::internal(format!("failed to resolve model path: {err}")))?;
        let pt_file = find_pt_model_file(&model_dir)?;
        let has_textproto = model_dir.join("model_inference.textproto").is_file();

        if pt_file.is_some() && has_textproto {
            let pt_file = pt_file.expect("pt file existence already checked");
            let mut tensor_bytes = Vec::<u8>::new();
            let mut seen_end_of_tensor = false;
            let expected_shape = read_input_shape_from_textproto(&model_dir)?;

            while let Some(message) = stream.message().await.map_err(|err| {
                Status::internal(format!("failed reading checkpoint stream message: {err}"))
            })? {
                let payload = message.payload.ok_or_else(|| {
                    Status::invalid_argument("checkpoint stream message has no payload")
                })?;
                match payload {
                    Payload::Meta(_) => {
                        return Err(Status::invalid_argument(
                            "metadata can only be sent as the first checkpoint stream message",
                        ));
                    }
                    Payload::Chunk(chunk) => {
                        if seen_end_of_tensor {
                            return Err(Status::invalid_argument(
                                "received tensor chunk after end_of_tensor=true",
                            ));
                        }

                        if !chunk.shape.is_empty() {
                            if chunk.shape != expected_shape {
                                return Err(Status::invalid_argument(
                                    "tensor chunk shape does not match model_inference.textproto",
                                ));
                            }
                        }

                        tensor_bytes.extend_from_slice(&chunk.data);
                        if chunk.end_of_tensor {
                            seen_end_of_tensor = true;
                        }
                    }
                }
            }

            run_rust_inference(&model_name, &model_dir, &pt_file, &tensor_bytes)?;

            let (tx, rx) = mpsc::channel::<Result<CheckpointResponse, Status>>(4);
            let _ = tx
                .send(Ok(CheckpointResponse {
                    chunk: format!("Rust inference completed for model '{model_name}'"),
                    done: false,
                    exit_code: 0,
                }))
                .await;
            let _ = tx
                .send(Ok(CheckpointResponse {
                    chunk: String::new(),
                    done: true,
                    exit_code: 0,
                }))
                .await;
            drop(tx);

            return Ok(Response::new(
                tonic::codegen::tokio_stream::wrappers::ReceiverStream::new(rx),
            ));
        }

        if pt_file.is_some() && !has_textproto {
            return Err(Status::failed_precondition(format!(
                "model '{model_name}' has a .pt file but is missing model_inference.textproto"
            )));
        }
        if pt_file.is_none() && has_textproto {
            return Err(Status::failed_precondition(format!(
                "model '{model_name}' has model_inference.textproto but no .pt file"
            )));
        }

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
        let (tx, rx) = mpsc::channel::<Result<CheckpointResponse, Status>>(64);

        std::thread::spawn(move || {
            let mut child = match Command::new(&python_path)
                .arg("-u")
                .arg("main.py")
                .current_dir(&model_dir)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
            {
                Ok(child) => child,
                Err(err) => {
                    let _ = tx.blocking_send(Err(Status::internal(format!(
                        "failed to run main.py: {err}"
                    ))));
                    return;
                }
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
                        "failed waiting on main.py process: {err}"
                    ))));
                    return;
                }
            };

            let _ = stdout_handle.join();
            let _ = stderr_handle.join();
            let _ = tx.blocking_send(Ok(CheckpointResponse {
                chunk: String::new(),
                done: true,
                exit_code: status.code().unwrap_or(-1),
            }));
        });

        Ok(Response::new(
            tonic::codegen::tokio_stream::wrappers::ReceiverStream::new(rx),
        ))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;
    let health = HealthService;
    let sonic = SonicService;
    println!("gRPC server listening on {}", addr);

    Server::builder()
        .add_service(HealthServer::new(health))
        .add_service(SonicServer::new(sonic))
        .serve(addr)
        .await?;

    Ok(())
}
