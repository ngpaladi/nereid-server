use std::fs;
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
    let model_roots = ["ml-backend", "ml-backends"];

    for root in model_roots {
        let entries = match fs::read_dir(root) {
            Ok(entries) => entries,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => {
                return Err(Status::internal(format!(
                    "failed to read model directory '{root}': {err}"
                )))
            }
        };

        let mut model_names = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|err| {
                Status::internal(format!("failed to read entry in '{root}': {err}"))
            })?;
            let file_type = entry.file_type().map_err(|err| {
                Status::internal(format!("failed to inspect entry in '{root}': {err}"))
            })?;

            if file_type.is_dir() {
                model_names.push(entry.file_name().to_string_lossy().into_owned());
            }
        }

        model_names.sort();
        if model_names.is_empty() {
            return Ok(vec!["No models found".to_string()]);
        }
        return Ok(model_names);
    }

    Ok(vec!["No models found".to_string()])
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
    type CheckpointStream = tonic::codegen::tokio_stream::wrappers::ReceiverStream<
        Result<CheckpointResponse, Status>,
    >;

    async fn view_models(
        &self,
        _request: Request<ViewModelsRequest>,
    ) -> Result<Response<ViewModelsResponse>, Status> {
        let model_names = get_model_names()?;
        Ok(Response::new(ViewModelsResponse {
            model_names,
        }))
    }

    async fn checkpoint(
        &self,
        _request: Request<CheckpointRequest>,
    ) -> Result<Response<Self::CheckpointStream>, Status> {
        Err(Status::unimplemented("Checkpoint is not implemented yet"))
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
