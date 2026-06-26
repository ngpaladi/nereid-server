use std::path::Path;
use std::sync::Arc;

use tokio::sync::mpsc;
use tonic::{Request, Response, Status, transport::Server};

mod config;
mod inference;
mod model_runtime;
mod python_backend;

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

        if let Some(model_dir) = self.model_manager.python_model_dir(&model_name) {
            tokio::spawn(async move { while let Ok(Some(_)) = stream.message().await {} });
            let python_stream =
                python_backend::spawn_python_checkpoint_stream(&model_name, model_dir)?;
            return Ok(Response::new(python_stream));
        }

        let input_contract = self
            .model_manager
            .input_contract(&model_name)
            .ok_or_else(|| {
                Status::not_found(format!(
                    "model '{model_name}' is not configured in nereid.yaml"
                ))
            })?
            .clone();

        let mut tensor_bytes = Vec::<u8>::new();
        let mut request_shape = None::<Vec<i64>>;
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

                    if chunk.shape.is_empty() {
                        return Err(Status::invalid_argument("tensor chunk shape is required"));
                    }

                    input_contract.validate_request_shape(&chunk.shape, &model_name)?;

                    match &request_shape {
                        Some(shape) if shape != &chunk.shape => {
                            return Err(Status::invalid_argument(
                                "tensor chunk shape changed within one checkpoint request",
                            ));
                        }
                        None => request_shape = Some(chunk.shape.clone()),
                        Some(_) => {}
                    }

                    tensor_bytes.extend_from_slice(&chunk.data);
                    if chunk.end_of_tensor {
                        seen_end_of_tensor = true;
                    }
                }
            }
        }

        let request_shape =
            request_shape.ok_or_else(|| Status::invalid_argument("no tensor chunks provided"))?;
        let input_tensor = tensor_from_input_bytes(&tensor_bytes, &request_shape, &model_name)?;
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

/// Full-stack inference tests.
///
/// These exercise the complete path a real client travels: a `NereidClient`
/// streaming gRPC over a loopback socket into a live `NereidServer`, through the
/// `ModelManager`, into a backend, and back out as a response stream. The matrix
/// is the two `DetectedBackendKind` variants — Rust (TorchScript `.pt`) and
/// Python (`main.py` + venv) — plus the backend-agnostic protocol surface
/// (model listing, health, malformed streams). CUDA is a *device*, orthogonal to
/// backend kind and unavailable without hardware, so every model here runs on
/// `ModelDevice::Cpu`.
#[cfg(test)]
mod checkpoint_e2e_tests {
    use super::*;
    use crate::config::{ModelConfig, ModelDevice, ServerConfig, ServerSection};
    use crate::inference::run_forward_pass;
    use proto::checkpoint_request::Payload;
    use proto::nereid_client::NereidClient;
    use proto::{
        CheckpointMeta, HealthCheckRequest, TensorChunk, ViewModelsRequest,
    };
    use std::net::SocketAddr;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tch::{CModule, Device, Tensor};
    use tokio::net::TcpListener;
    use tokio_stream::wrappers::TcpListenerStream;

    fn temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("nereid-e2e-{prefix}-{nanos}"));
        std::fs::create_dir_all(&path).expect("create temp dir");
        path
    }

    /// Path to the committed `ml-backends` fixtures (model1/model2 = Python,
    /// model3 = Rust). These are checked into the repo, so absence is a hard
    /// failure rather than a silent skip.
    fn fixtures_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ml-backends")
    }

    async fn spawn_test_server(ml_backends_path: PathBuf, models: Vec<ModelConfig>) -> SocketAddr {
        let config = ServerConfig {
            server: ServerSection {
                bind_addr: "127.0.0.1:0".to_string(),
                ml_backends_path: ml_backends_path.to_string_lossy().into_owned(),
            },
            models,
        };
        let model_manager =
            Arc::new(ModelManager::from_config(&config).expect("model manager should build"));
        let nereid = NereidService::new(model_manager);

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ephemeral port");
        let addr = listener.local_addr().expect("local addr");
        let incoming = TcpListenerStream::new(listener);

        tokio::spawn(async move {
            let _ = Server::builder()
                .add_service(NereidServer::new(nereid))
                .serve_with_incoming(incoming)
                .await;
        });

        addr
    }

    async fn connect(addr: SocketAddr) -> NereidClient<tonic::transport::Channel> {
        NereidClient::connect(format!("http://{addr}"))
            .await
            .expect("client should connect")
    }

    fn cpu_model(name: &str) -> ModelConfig {
        ModelConfig {
            name: name.to_string(),
            device: ModelDevice::Cpu,
            queue_capacity: 4,
        }
    }

    fn meta(model_name: &str) -> CheckpointRequest {
        CheckpointRequest {
            payload: Some(Payload::Meta(CheckpointMeta {
                model_name: model_name.to_string(),
                output_file: String::new(),
            })),
        }
    }

    fn tensor_chunk(shape: Vec<i64>, data: Vec<u8>, end_of_tensor: bool) -> CheckpointRequest {
        CheckpointRequest {
            payload: Some(Payload::Chunk(TensorChunk {
                tensor_name: "input".to_string(),
                shape,
                data,
                chunk_index: 0,
                end_of_tensor,
            })),
        }
    }

    fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// The full response of a checkpoint call, flattened across the stream.
    #[derive(Debug, Default)]
    struct Collected {
        /// Non-empty `chunk` text lines (Python stdout/stderr, Rust status text).
        lines: Vec<String>,
        /// Reassembled output tensor bytes (Rust backend).
        output_bytes: Vec<u8>,
        /// Shape reported on output chunks, if any.
        output_shape: Option<Vec<i64>>,
        saw_output_chunk: bool,
        saw_done: bool,
        exit_code: i32,
    }

    /// Drive a checkpoint call to completion and flatten the response stream.
    ///
    /// In tonic a terminal `Status` may surface either at the initial
    /// `checkpoint(...).await` or on a later `message().await` of the response
    /// stream — which one is timing/version dependent. This helper normalizes
    /// both into a single `Err(Status)`, so error-path tests never depend on
    /// *where* the failure shows up.
    async fn run_checkpoint(
        client: &mut NereidClient<tonic::transport::Channel>,
        requests: Vec<CheckpointRequest>,
    ) -> Result<Collected, Status> {
        let response = client
            .checkpoint(tokio_stream::iter(requests))
            .await?;
        let mut stream = response.into_inner();

        let mut collected = Collected::default();
        while let Some(resp) = stream.message().await? {
            if !resp.chunk.is_empty() {
                collected.lines.push(resp.chunk);
            }
            if let Some(output_chunk) = resp.output_chunk {
                collected.saw_output_chunk = true;
                collected.output_shape = Some(output_chunk.shape);
                collected.output_bytes.extend_from_slice(&output_chunk.data);
            }
            if resp.done {
                collected.saw_done = true;
                collected.exit_code = resp.exit_code;
            }
        }
        Ok(collected)
    }

    /// Create a self-contained Python backend fixture under a fresh temp
    /// `ml-backends` dir. An empty `requirements.txt` keeps the venv build that
    /// `ModelManager::from_config` performs cheap. Returns `(ml_backends, name)`.
    fn make_python_model(prefix: &str, name: &str, main_py: &str) -> (PathBuf, String) {
        let ml_backends = temp_dir(prefix);
        let model_dir = ml_backends.join(name);
        std::fs::create_dir_all(&model_dir).expect("create model dir");
        std::fs::write(model_dir.join("requirements.txt"), b"\n").expect("write requirements.txt");
        std::fs::write(model_dir.join("main.py"), main_py.as_bytes()).expect("write main.py");
        (ml_backends, name.to_string())
    }

    // ---------------------------------------------------------------------
    // Rust backend (TorchScript .pt): model3, input_shape [16], max_batch 10
    // ---------------------------------------------------------------------

    /// The centerpiece: the bytes streamed back over the full gRPC stack must be
    /// bit-for-bit identical to running the model's forward pass directly. This
    /// is the strongest possible "inference works as expected" assertion — it
    /// proves serialization, chunking, reassembly, and the worker thread all
    /// preserve the model's actual output, not merely that *some* tensor came
    /// back.
    #[tokio::test]
    async fn rust_backend_output_matches_direct_forward_pass() {
        let ml_backends = fixtures_dir();
        let model_path = ml_backends.join("model3").join("mlp.pt");
        assert!(
            model_path.is_file(),
            "committed Rust fixture missing at {}",
            model_path.display()
        );

        let input_values: Vec<f32> = (0..16).map(|v| v as f32).collect();

        // Expected: load model3 and run the forward pass directly on CPU.
        let model = CModule::load_on_device(
            model_path.to_str().expect("valid UTF-8 model path"),
            Device::Cpu,
        )
        .expect("model should load");
        let input_tensor = Tensor::from_slice(&input_values).reshape([1, 16]);
        let (expected_shape, expected_bytes) =
            run_forward_pass(&model, Device::Cpu, &input_tensor).expect("direct forward pass");

        // Actual: same input through the full gRPC server.
        let addr = spawn_test_server(ml_backends, vec![cpu_model("model3")]).await;
        let mut client = connect(addr).await;
        let collected = run_checkpoint(
            &mut client,
            vec![
                meta("model3"),
                tensor_chunk(vec![1, 16], f32_le_bytes(&input_values), true),
            ],
        )
        .await
        .expect("checkpoint should succeed");

        assert!(collected.saw_done, "stream must end with done=true");
        assert_eq!(collected.exit_code, 0, "Rust backend exit code");
        assert!(collected.saw_output_chunk, "expected an output tensor chunk");
        assert_eq!(
            collected.output_shape.as_deref(),
            Some(expected_shape.as_slice()),
            "streamed output shape must match direct forward pass"
        );
        assert_eq!(
            collected.output_bytes, expected_bytes,
            "streamed output bytes must match direct forward pass byte-for-byte"
        );
    }

    /// Splitting the input tensor across multiple `TensorChunk` messages must
    /// reassemble to exactly the same output as sending it in one chunk — the
    /// server stitches `chunk.data` in arrival order before inference.
    #[tokio::test]
    async fn rust_backend_chunked_input_matches_single_chunk() {
        let ml_backends = fixtures_dir();
        assert!(ml_backends.join("model3").is_dir(), "model3 fixture missing");

        let input_values: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let all_bytes = f32_le_bytes(&input_values);
        let (first_half, second_half) = all_bytes.split_at(all_bytes.len() / 2);

        let addr = spawn_test_server(ml_backends, vec![cpu_model("model3")]).await;
        let mut client = connect(addr).await;

        let single = run_checkpoint(
            &mut client,
            vec![
                meta("model3"),
                tensor_chunk(vec![1, 16], all_bytes.clone(), true),
            ],
        )
        .await
        .expect("single-chunk checkpoint should succeed");

        let chunked = run_checkpoint(
            &mut client,
            vec![
                meta("model3"),
                tensor_chunk(vec![1, 16], first_half.to_vec(), false),
                tensor_chunk(vec![1, 16], second_half.to_vec(), true),
            ],
        )
        .await
        .expect("chunked checkpoint should succeed");

        assert!(chunked.saw_output_chunk, "expected output from chunked input");
        assert_eq!(
            chunked.output_shape, single.output_shape,
            "chunked input must yield the same output shape"
        );
        assert_eq!(
            chunked.output_bytes, single.output_bytes,
            "chunked input must yield identical output bytes"
        );
    }

    /// A batched request (batch dim within `max_batch_size`) is accepted and the
    /// output carries the batch dimension through.
    #[tokio::test]
    async fn rust_backend_supports_batch_dimension() {
        let ml_backends = fixtures_dir();
        assert!(ml_backends.join("model3").is_dir(), "model3 fixture missing");

        let batch = 2i64;
        let values: Vec<f32> = (0..(batch * 16)).map(|v| v as f32).collect();

        let addr = spawn_test_server(ml_backends, vec![cpu_model("model3")]).await;
        let mut client = connect(addr).await;
        let collected = run_checkpoint(
            &mut client,
            vec![
                meta("model3"),
                tensor_chunk(vec![batch, 16], f32_le_bytes(&values), true),
            ],
        )
        .await
        .expect("batched checkpoint should succeed");

        assert!(collected.saw_done, "batched stream must finish");
        let shape = collected.output_shape.expect("batched output shape");
        assert_eq!(
            shape.first().copied(),
            Some(batch),
            "output should preserve the batch dimension, got {shape:?}"
        );
    }

    /// A trailing-dimension mismatch (model expects 16, client sends 15) is
    /// rejected with `InvalidArgument` by the input-shape contract.
    #[tokio::test]
    async fn rust_backend_rejects_shape_mismatch() {
        let ml_backends = fixtures_dir();
        assert!(ml_backends.join("model3").is_dir(), "model3 fixture missing");

        let values: Vec<f32> = (0..15).map(|v| v as f32).collect();
        let addr = spawn_test_server(ml_backends, vec![cpu_model("model3")]).await;
        let mut client = connect(addr).await;
        let status = run_checkpoint(
            &mut client,
            vec![
                meta("model3"),
                tensor_chunk(vec![1, 15], f32_le_bytes(&values), true),
            ],
        )
        .await
        .expect_err("shape mismatch must be rejected");

        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status:?}");
    }

    /// A rank mismatch — sending `[16]` when the contract expects `[batch, 16]`
    /// — is rejected with `InvalidArgument`.
    #[tokio::test]
    async fn rust_backend_rejects_rank_mismatch() {
        let ml_backends = fixtures_dir();
        assert!(ml_backends.join("model3").is_dir(), "model3 fixture missing");

        let values: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let addr = spawn_test_server(ml_backends, vec![cpu_model("model3")]).await;
        let mut client = connect(addr).await;
        let status = run_checkpoint(
            &mut client,
            vec![
                meta("model3"),
                tensor_chunk(vec![16], f32_le_bytes(&values), true),
            ],
        )
        .await
        .expect_err("rank mismatch must be rejected");

        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status:?}");
    }

    /// A batch size above `max_batch_size` (10 for model3) is rejected with
    /// `InvalidArgument`.
    #[tokio::test]
    async fn rust_backend_rejects_batch_over_max() {
        let ml_backends = fixtures_dir();
        assert!(ml_backends.join("model3").is_dir(), "model3 fixture missing");

        let batch = 11i64;
        let values: Vec<f32> = (0..(batch * 16)).map(|v| v as f32).collect();
        let addr = spawn_test_server(ml_backends, vec![cpu_model("model3")]).await;
        let mut client = connect(addr).await;
        let status = run_checkpoint(
            &mut client,
            vec![
                meta("model3"),
                tensor_chunk(vec![batch, 16], f32_le_bytes(&values), true),
            ],
        )
        .await
        .expect_err("over-max batch must be rejected");

        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status:?}");
    }

    // ---------------------------------------------------------------------
    // Python backend (main.py + venv)
    // ---------------------------------------------------------------------

    /// Happy path: the server builds the venv, runs `main.py`, streams every
    /// stdout line back in order, and ends with `done=true` / exit code 0.
    #[tokio::test]
    async fn python_backend_streams_stdout_and_exit_zero() {
        let (ml_backends, name) = make_python_model(
            "python-stdout",
            "e2e_python_stdout",
            "print('line one')\nprint('line two')\nprint('line three')\n",
        );

        let addr = spawn_test_server(ml_backends.clone(), vec![cpu_model(&name)]).await;
        let mut client = connect(addr).await;
        let collected = run_checkpoint(&mut client, vec![meta(&name)])
            .await
            .expect("python checkpoint should succeed");

        assert!(collected.saw_done, "stream must end with done=true");
        assert_eq!(collected.exit_code, 0, "successful main.py exits 0");
        let stdout: Vec<&String> = collected.lines.iter().collect();
        assert_eq!(
            stdout,
            vec!["line one", "line two", "line three"],
            "stdout lines must stream back in order, got {stdout:?}"
        );

        let _ = std::fs::remove_dir_all(&ml_backends);
    }

    /// Anything `main.py` writes to stderr is surfaced to the client, tagged
    /// with the `stderr: ` prefix the server adds.
    #[tokio::test]
    async fn python_backend_captures_stderr() {
        let (ml_backends, name) = make_python_model(
            "python-stderr",
            "e2e_python_stderr",
            "import sys\nprint('to stdout')\nprint('to stderr', file=sys.stderr)\n",
        );

        let addr = spawn_test_server(ml_backends.clone(), vec![cpu_model(&name)]).await;
        let mut client = connect(addr).await;
        let collected = run_checkpoint(&mut client, vec![meta(&name)])
            .await
            .expect("python checkpoint should succeed");

        assert!(collected.saw_done, "stream must end with done=true");
        assert!(
            collected.lines.iter().any(|l| l == "to stdout"),
            "stdout line missing: {:?}",
            collected.lines
        );
        assert!(
            collected.lines.iter().any(|l| l == "stderr: to stderr"),
            "expected stderr line tagged with 'stderr: ', got {:?}",
            collected.lines
        );

        let _ = std::fs::remove_dir_all(&ml_backends);
    }

    /// A non-zero process exit code from `main.py` is propagated in the terminal
    /// `done` message rather than swallowed.
    #[tokio::test]
    async fn python_backend_propagates_nonzero_exit() {
        let (ml_backends, name) = make_python_model(
            "python-exit",
            "e2e_python_exit",
            "import sys\nprint('about to fail')\nsys.exit(3)\n",
        );

        let addr = spawn_test_server(ml_backends.clone(), vec![cpu_model(&name)]).await;
        let mut client = connect(addr).await;
        let collected = run_checkpoint(&mut client, vec![meta(&name)])
            .await
            .expect("python checkpoint stream should complete even on failure");

        assert!(collected.saw_done, "stream must end with done=true");
        assert_eq!(
            collected.exit_code, 3,
            "non-zero exit code must be propagated"
        );
        assert!(
            collected.lines.iter().any(|l| l == "about to fail"),
            "stdout before exit missing: {:?}",
            collected.lines
        );

        let _ = std::fs::remove_dir_all(&ml_backends);
    }

    // ---------------------------------------------------------------------
    // Backend-agnostic protocol surface
    // ---------------------------------------------------------------------

    /// `view_models` lists exactly the configured models and `health_check`
    /// reports ok — the lightweight RPCs a client uses before streaming.
    #[tokio::test]
    async fn view_models_and_health_check() {
        let ml_backends = fixtures_dir();
        assert!(ml_backends.join("model3").is_dir(), "model3 fixture missing");

        let addr = spawn_test_server(ml_backends, vec![cpu_model("model3")]).await;
        let mut client = connect(addr).await;

        let health = client
            .health_check(HealthCheckRequest {})
            .await
            .expect("health check should succeed")
            .into_inner();
        assert_eq!(health.status, "ok");

        let models = client
            .view_models(ViewModelsRequest {})
            .await
            .expect("view models should succeed")
            .into_inner();
        assert_eq!(models.model_names, vec!["model3".to_string()]);
    }

    /// An unconfigured model name is rejected with `NotFound`.
    #[tokio::test]
    async fn unknown_model_returns_not_found() {
        let ml_backends = fixtures_dir();
        assert!(ml_backends.join("model3").is_dir(), "model3 fixture missing");

        let addr = spawn_test_server(ml_backends, vec![cpu_model("model3")]).await;
        let mut client = connect(addr).await;
        let values: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let status = run_checkpoint(
            &mut client,
            vec![
                meta("does_not_exist"),
                tensor_chunk(vec![1, 16], f32_le_bytes(&values), true),
            ],
        )
        .await
        .expect_err("unknown model must be rejected");

        assert_eq!(status.code(), tonic::Code::NotFound, "{status:?}");
    }

    /// A checkpoint stream whose first message is a tensor chunk (not metadata)
    /// is rejected with `InvalidArgument`.
    #[tokio::test]
    async fn first_message_must_be_metadata() {
        let ml_backends = fixtures_dir();
        assert!(ml_backends.join("model3").is_dir(), "model3 fixture missing");

        let addr = spawn_test_server(ml_backends, vec![cpu_model("model3")]).await;
        let mut client = connect(addr).await;
        let values: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let status = run_checkpoint(
            &mut client,
            vec![tensor_chunk(vec![1, 16], f32_le_bytes(&values), true)],
        )
        .await
        .expect_err("leading chunk must be rejected");

        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status:?}");
    }

    /// An empty checkpoint stream (no messages at all) is rejected with
    /// `InvalidArgument`.
    #[tokio::test]
    async fn empty_stream_is_rejected() {
        let ml_backends = fixtures_dir();
        assert!(ml_backends.join("model3").is_dir(), "model3 fixture missing");

        let addr = spawn_test_server(ml_backends, vec![cpu_model("model3")]).await;
        let mut client = connect(addr).await;
        let status = run_checkpoint(&mut client, vec![])
            .await
            .expect_err("empty stream must be rejected");

        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status:?}");
    }
}
