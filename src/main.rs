use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tokio::sync::mpsc;
use tonic::{Request, Response, Status, transport::Server};

use inference::health_server::{Health, HealthServer};
use inference::sonic_server::{Sonic, SonicServer};
use inference::{
    CheckpointRequest, CheckpointResponse, HealthCheckRequest, HealthCheckResponse,
    ViewModelsRequest, ViewModelsResponse,
};

pub mod inference {
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
        request: Request<CheckpointRequest>,
    ) -> Result<Response<Self::CheckpointStream>, Status> {
        let request = request.into_inner();
        let model_name = request.model_name.trim().to_string();

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
    prepare_model_envs()?;

    let addr = "[::1]:50051".parse()?;
    let health = HealthService;
    let sonic = SonicService;

    Server::builder()
        .add_service(HealthServer::new(health))
        .add_service(SonicServer::new(sonic))
        .serve(addr)
        .await?;

    Ok(())
}
