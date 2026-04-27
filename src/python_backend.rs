use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use tokio::sync::mpsc;
use tonic::Status;

use crate::get_model_names;
use crate::proto::CheckpointResponse;

pub type CheckpointStream =
    tonic::codegen::tokio_stream::wrappers::ReceiverStream<Result<CheckpointResponse, Status>>;

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

pub fn prepare_model_envs() -> Result<(), Box<dyn std::error::Error>> {
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

pub fn spawn_python_checkpoint_stream(
    model_name: &str,
    model_dir: PathBuf,
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
        let _ = tx.blocking_send(Ok(CheckpointResponse {
            chunk: String::new(),
            done: true,
            exit_code: status.code().unwrap_or(-1),
            output_chunk: None,
        }));
    });

    Ok(tonic::codegen::tokio_stream::wrappers::ReceiverStream::new(
        rx,
    ))
}
