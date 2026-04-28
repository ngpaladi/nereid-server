use std::path::Path;
use std::sync::Arc;

use tokio::sync::mpsc;
use tonic::{Request, Response, Status, transport::Server};

mod config;
mod inference;
mod model_runtime;

use config::load_server_config;
use model_runtime::{ModelManager, tensor_from_input_bytes};
use proto::nereid_server::{Nereid, NereidServer};
use proto::{
    CheckpointRequest, CheckpointResponse, HealthCheckRequest, HealthCheckResponse, TensorChunk,
    ViewModelsRequest, ViewModelsResponse, checkpoint_request::Payload,
};

pub mod proto {
    tonic::include_proto!("inference");
}

type CheckpointStream =
    tonic::codegen::tokio_stream::wrappers::ReceiverStream<Result<CheckpointResponse, Status>>;

fn output_to_stream(
    model_name: &str,
    output_shape: Vec<i64>,
    output_bytes: Vec<u8>,
) -> CheckpointStream {
    const OUTPUT_CHUNK_BYTES: usize = 64 * 1024;

    let num_chunks = output_bytes.len().div_ceil(OUTPUT_CHUNK_BYTES);
    let response_capacity = usize::max(2, num_chunks + 2);
    let (tx, rx) = mpsc::channel::<Result<CheckpointResponse, Status>>(response_capacity);

    let model_name = model_name.to_string();
    tokio::spawn(async move {
        let _ = tx
            .send(Ok(CheckpointResponse {
                chunk: format!("Rust inference completed for model '{model_name}'"),
                done: false,
                exit_code: 0,
                output_chunk: None,
            }))
            .await;

        if output_bytes.is_empty() {
            let _ = tx
                .send(Ok(CheckpointResponse {
                    chunk: String::new(),
                    done: false,
                    exit_code: 0,
                    output_chunk: Some(TensorChunk {
                        tensor_name: "output".to_string(),
                        shape: output_shape.clone(),
                        data: Vec::new(),
                        chunk_index: 0,
                        end_of_tensor: true,
                    }),
                }))
                .await;
        } else {
            for (chunk_index, data_chunk) in output_bytes.chunks(OUTPUT_CHUNK_BYTES).enumerate() {
                let _ = tx
                    .send(Ok(CheckpointResponse {
                        chunk: String::new(),
                        done: false,
                        exit_code: 0,
                        output_chunk: Some(TensorChunk {
                            tensor_name: "output".to_string(),
                            shape: output_shape.clone(),
                            data: data_chunk.to_vec(),
                            chunk_index: chunk_index as u64,
                            end_of_tensor: chunk_index + 1 == num_chunks,
                        }),
                    }))
                    .await;
            }
        }

        let _ = tx
            .send(Ok(CheckpointResponse {
                chunk: String::new(),
                done: true,
                exit_code: 0,
                output_chunk: None,
            }))
            .await;
    });

    tonic::codegen::tokio_stream::wrappers::ReceiverStream::new(rx)
}

#[derive(Clone)]
pub struct NereidService {
    model_manager: Arc<ModelManager>,
}

impl NereidService {
    fn new(model_manager: Arc<ModelManager>) -> Self {
        Self { model_manager }
    }
}

#[tonic::async_trait]
impl Nereid for NereidService {
    type CheckpointStream = CheckpointStream;

    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        Ok(Response::new(HealthCheckResponse {
            status: "ok".to_string(),
        }))
    }

    async fn view_models(
        &self,
        _request: Request<ViewModelsRequest>,
    ) -> Result<Response<ViewModelsResponse>, Status> {
        Ok(Response::new(ViewModelsResponse {
            model_names: self.model_manager.configured_models(),
        }))
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

        let expected_shape = self
            .model_manager
            .input_shape(&model_name)
            .ok_or_else(|| {
                Status::not_found(format!(
                    "model '{model_name}' is not configured in nereid.yaml"
                ))
            })?
            .to_vec();

        let mut tensor_bytes = Vec::<u8>::new();
        let mut seen_end_of_tensor = false;

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

                    if !chunk.shape.is_empty() && chunk.shape != expected_shape {
                        return Err(Status::invalid_argument(
                            "tensor chunk shape does not match model_inference.textproto",
                        ));
                    }

                    tensor_bytes.extend_from_slice(&chunk.data);
                    if chunk.end_of_tensor {
                        seen_end_of_tensor = true;
                    }
                }
            }
        }

        let input_tensor = tensor_from_input_bytes(&tensor_bytes, &expected_shape, &model_name)?;
        let response_rx = self.model_manager.enqueue(&model_name, input_tensor)?;
        let (output_shape, output_bytes) = response_rx.await.map_err(|_| {
            Status::internal(format!(
                "worker response channel closed for model '{model_name}'"
            ))
        })??;

        Ok(Response::new(output_to_stream(
            &model_name,
            output_shape,
            output_bytes,
        )))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config_path = Path::new("nereid.yaml");
    let config = load_server_config(config_path)?;

    let addr = config.server.bind_addr.parse()?;
    let model_manager = Arc::new(
        ModelManager::from_config(&config)
            .map_err(|status| std::io::Error::other(status.to_string()))?,
    );

    let nereid = NereidService::new(model_manager);
    println!("gRPC server listening on {}", addr);

    Server::builder()
        .add_service(NereidServer::new(nereid))
        .serve(addr)
        .await?;

    Ok(())
}
