use std::fs;
use std::path::Path;
use std::process::Command;
use tonic::{transport::Server, Request, Response, Status};

use inference::health_server::{Health, HealthServer};
use inference::sonic_server::{Sonic, SonicServer};
use inference::{
    CheckpointRequest, CheckpointResponse, HealthCheckRequest, HealthCheckResponse,
    ViewModelsRequest, ViewModelsResponse,
};

pub mod inference {
    tonic::include_proto!("inference");
}

fn get_model_names() -> Result<Vec<String>, Status> {
    let entries = match fs::read_dir("ml-backends") {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => {
            return Err(Status::internal(format!(
                "failed to read model directory 'ml-backends': {err}"
            )))
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
        tonic::codegen::tokio_stream::Iter<std::vec::IntoIter<Result<CheckpointResponse, Status>>>;

    async fn view_models(
        &self,
        _request: Request<ViewModelsRequest>,
    ) -> Result<Response<ViewModelsResponse>, Status> {
        let mut model_names = get_model_names()?;
        if model_names.is_empty() {
            model_names.push("No models found".to_string());
        }

        Ok(Response::new(ViewModelsResponse {
            model_names,
        }))
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

        let model_dir = Path::new("ml-backends").join(&model_name);
        let main_py = model_dir.join("main.py");
        if !main_py.is_file() {
            return Err(Status::not_found(format!(
                "main.py not found for model '{model_name}'"
            )));
        }

        let output = Command::new("python3")
            .arg("main.py")
            .current_dir(&model_dir)
            .output()
            .map_err(|err| Status::internal(format!("failed to run python3 main.py: {err}")))?;

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let chunk = match (stdout.is_empty(), stderr.is_empty()) {
            (true, true) => "checkpoint command finished with no output".to_string(),
            (false, true) => stdout,
            (true, false) => stderr,
            (false, false) => format!("{stdout}\n{stderr}"),
        };

        let response = CheckpointResponse {
            chunk,
            done: true,
            exit_code: output.status.code().unwrap_or(-1),
        };

        let stream = tonic::codegen::tokio_stream::iter(vec![Ok(response)]);
        Ok(Response::new(stream))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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
