use std::path::Path;
use std::sync::Arc;

use tonic::{Request, Response, Status, transport::Server};

mod backend;
// The inference backends, discovered from `src/backends/` by build.rs. Declared
// after `backend` (whose registry they populate); each self-registers, so this
// is the only line that knows they exist.
mod backends;
mod config;
mod dtype;
mod triton;

use backend::{Contract, ModelManager, Tensor};
use config::load_server_config;
use proto::grpc_inference_service_server::GrpcInferenceServiceServer;
use proto::nereid_server::{Nereid, NereidServer};
use proto::{
    CheckpointRequest, HealthCheckRequest, HealthCheckResponse, ViewModelsRequest,
    ViewModelsResponse, checkpoint_request::Payload,
};
use triton::TritonService;

pub mod proto {
    tonic::include_proto!("inference");
}

type CheckpointStream = backend::CheckpointStream;

/// Drain the tensor chunks of a checkpoint stream, validating each chunk's shape
/// against the model's `contract` and reassembling the data into the single
/// float32 input tensor. The `Checkpoint` wire contract is little-endian float32;
/// a request that omits the declared batch dimension is expanded to batch 1.
async fn collect_input_tensor(
    stream: &mut tonic::Streaming<CheckpointRequest>,
    contract: &Contract,
    model_name: &str,
) -> Result<Tensor, Status> {
    let spec = contract.inputs.first().ok_or_else(|| {
        Status::internal(format!("model '{model_name}' has no declared input tensor"))
    })?;
    let mut tensor_bytes = Vec::<u8>::new();
    let mut request_shape = None::<Vec<i64>>;
    let mut seen_end_of_tensor = false;

    while let Some(message) = stream.message().await.map_err(|err| {
        Status::internal(format!("failed reading checkpoint stream message: {err}"))
    })? {
        let payload = message
            .payload
            .ok_or_else(|| Status::invalid_argument("checkpoint stream message has no payload"))?;

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

                contract.validate_input_shape(spec, &chunk.shape, model_name)?;

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
    // A request that omits the model's declared batch dimension is expanded to
    // batch size 1 before inference.
    let request_shape = contract.normalize_request_shape(spec, request_shape);
    Ok(Tensor {
        name: "input".to_string(),
        shape: request_shape,
        dtype: "float32".to_string(),
        data: tensor_bytes,
    })
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

        let contract = self
            .model_manager
            .contract(&model_name)
            .cloned()
            .ok_or_else(|| {
                Status::not_found(format!(
                    "model '{model_name}' is not configured in nereid.yaml"
                ))
            })?;

        // When the model declares an input tensor, validate and reassemble it
        // from the request stream; otherwise drain the (input-less) stream. The
        // backend then streams its `Checkpoint` response (Python backends also
        // interleave stdout/stderr log chunks).
        let input = if contract.has_input() {
            Some(collect_input_tensor(&mut stream, &contract, &model_name).await?)
        } else {
            tokio::spawn(async move { while let Ok(Some(_)) = stream.message().await {} });
            None
        };

        Ok(Response::new(
            self.model_manager.checkpoint(&model_name, input)?,
        ))
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

    let nereid = NereidService::new(model_manager.clone());
    let triton = TritonService::new(model_manager);
    println!("gRPC server listening on {}", addr);

    // allow messages up to 128MB in TritonServer
    let max_msg_size = 128 * 1024 * 1024;

    let nereid_server = NereidServer::new(nereid);
    let triton_server = GrpcInferenceServiceServer::new(triton)
        .max_decoding_message_size(max_msg_size)
        .max_encoding_message_size(max_msg_size);

    // Both the native Nereid service and the Triton-compatible
    // GRPCInferenceService are served on the same address, so a KServe v2 client
    // (e.g. tritonclient) can talk to the latter without knowing it isn't Triton.
    Server::builder()
        .add_service(nereid_server)
        .add_service(triton_server)
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
    use proto::checkpoint_request::Payload;
    use proto::nereid_client::NereidClient;
    use proto::{CheckpointMeta, HealthCheckRequest, TensorChunk, ViewModelsRequest};
    use std::net::SocketAddr;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};
    use tch::{CModule, Device, Tensor};
    use tokio::net::TcpListener;
    use tokio_stream::wrappers::TcpListenerStream;

    /// Run a single-input/single-output `.pt` model's forward pass directly and
    /// serialize its float32 output to little-endian bytes — the "expected"
    /// oracle the server's streamed output is compared against.
    fn direct_forward_f32(model_path: &Path, input: &[f32], shape: &[i64]) -> (Vec<i64>, Vec<u8>) {
        let model = CModule::load_on_device(model_path.to_str().expect("utf-8 path"), Device::Cpu)
            .expect("model should load");
        let input_tensor = Tensor::from_slice(input).reshape(shape);
        let output = model
            .forward_ts(&[input_tensor])
            .expect("direct forward pass")
            .to_device(Device::Cpu)
            .contiguous();
        let out_shape = output.size();
        let numel = output.numel();
        let mut bytes = vec![0u8; numel * 4];
        output.reshape([-1]).copy_data_u8(&mut bytes, numel);
        (out_shape, bytes)
    }

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
            backend: None,
            signature: None,
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
        let response = client.checkpoint(tokio_stream::iter(requests)).await?;
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

    /// A tail appended to every Python fixture's `main.py`: writes the unit
    /// output tensor `[42.0]` (shape `[1]`) to `NEREID_OUTPUT_PATH`, satisfying
    /// the "every Python reply is a typed tensor" contract. It runs after the
    /// fixture's own body, so a body that `sys.exit`s first never reaches it.
    const WRITE_UNIT_TENSOR: &str = "
import os as _os, struct as _struct
with open(_os.environ['NEREID_OUTPUT_PATH'], 'wb') as _f:
    _f.write(b'float32 1\\n')
    _f.write(_struct.pack('<f', 42.0))
";

    /// Create a self-contained *output-only* Python backend fixture (declares
    /// only `output_shape: [1]`, consumes no input tensor) under a fresh temp
    /// `ml-backends` dir. An empty `requirements.txt` keeps the venv build that
    /// `ModelManager::from_config` performs cheap. The unit-tensor tail is
    /// appended so the model satisfies the tensor-reply contract.
    fn make_python_model(prefix: &str, name: &str, main_py: &str) -> (PathBuf, String) {
        make_python_model_with_contract(prefix, name, "output_shape: [1]\n", main_py)
    }

    /// Like [`make_python_model`], but with a caller-supplied
    /// `model_inference.textproto` (e.g. declaring `input_shape` so the request
    /// tensor is validated and piped into `main.py` on stdin). The textproto
    /// must declare `output_shape` — required for every Python model.
    fn make_python_model_with_contract(
        prefix: &str,
        name: &str,
        textproto: &str,
        main_py: &str,
    ) -> (PathBuf, String) {
        let ml_backends = temp_dir(prefix);
        let model_dir = ml_backends.join(name);
        std::fs::create_dir_all(&model_dir).expect("create model dir");
        std::fs::write(model_dir.join("requirements.txt"), b"\n").expect("write requirements.txt");
        let main_py = format!("{main_py}{WRITE_UNIT_TENSOR}");
        std::fs::write(model_dir.join("main.py"), main_py.as_bytes()).expect("write main.py");
        std::fs::write(
            model_dir.join("model_inference.textproto"),
            textproto.as_bytes(),
        )
        .expect("write textproto");
        (ml_backends, name.to_string())
    }

    /// Like [`make_python_model_with_contract`] but writes `main.py` verbatim
    /// (no appended unit-tensor tail), for fixtures that produce their own
    /// output tensor.
    fn make_python_model_raw(
        prefix: &str,
        name: &str,
        textproto: &str,
        main_py: &str,
    ) -> (PathBuf, String) {
        let ml_backends = temp_dir(prefix);
        let model_dir = ml_backends.join(name);
        std::fs::create_dir_all(&model_dir).expect("create model dir");
        std::fs::write(model_dir.join("requirements.txt"), b"\n").expect("write requirements.txt");
        std::fs::write(model_dir.join("main.py"), main_py.as_bytes()).expect("write main.py");
        std::fs::write(
            model_dir.join("model_inference.textproto"),
            textproto.as_bytes(),
        )
        .expect("write textproto");
        (ml_backends, name.to_string())
    }

    /// A `main.py` that reads the float32 tensor from stdin (no numpy, so the
    /// venv stays cheap) and echoes back values derived from the bytes: the
    /// advertised shape, the element count, and the sum. The derived sum is what
    /// proves the tensor arrived intact and in order.
    const STDIN_ECHO_MAIN_PY: &str = "\
import os, struct, sys
raw = sys.stdin.buffer.read()
count = len(raw) // 4
values = struct.unpack('<%df' % count, raw) if count else ()
print('shape:', os.environ.get('NEREID_INPUT_SHAPE', ''))
print('count:', count)
print('sum:', sum(values))
";

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
        let (expected_shape, expected_bytes) =
            direct_forward_f32(&model_path, &input_values, &[1, 16]);

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
        assert!(
            collected.saw_output_chunk,
            "expected an output tensor chunk"
        );
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
        assert!(
            ml_backends.join("model3").is_dir(),
            "model3 fixture missing"
        );

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

        assert!(
            chunked.saw_output_chunk,
            "expected output from chunked input"
        );
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
        assert!(
            ml_backends.join("model3").is_dir(),
            "model3 fixture missing"
        );

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
        assert!(
            ml_backends.join("model3").is_dir(),
            "model3 fixture missing"
        );

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

    /// A rank mismatch — sending `[1, 1, 16]` (rank 3) when the contract expects
    /// `[batch, 16]` (rank 2) or the bare `[16]` (rank 1) — is rejected with
    /// `InvalidArgument`. Note that the bare `[16]` is *not* a mismatch: it is
    /// auto-expanded to `[1, 16]` (see `rust_backend_expands_missing_batch_dim`).
    #[tokio::test]
    async fn rust_backend_rejects_rank_mismatch() {
        let ml_backends = fixtures_dir();
        assert!(
            ml_backends.join("model3").is_dir(),
            "model3 fixture missing"
        );

        let values: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let addr = spawn_test_server(ml_backends, vec![cpu_model("model3")]).await;
        let mut client = connect(addr).await;
        let status = run_checkpoint(
            &mut client,
            vec![
                meta("model3"),
                tensor_chunk(vec![1, 1, 16], f32_le_bytes(&values), true),
            ],
        )
        .await
        .expect_err("rank mismatch must be rejected");

        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status:?}");
    }

    /// A request that omits the declared batch dimension — sending the bare
    /// `[16]` when the contract is `input_shape [16]`, `max_batch 10` — is
    /// accepted and auto-expanded to batch size 1. The result must be identical
    /// to sending `[1, 16]` explicitly, and the output must carry the inserted
    /// leading batch dimension of 1.
    #[tokio::test]
    async fn rust_backend_expands_missing_batch_dim() {
        let ml_backends = fixtures_dir();
        assert!(
            ml_backends.join("model3").is_dir(),
            "model3 fixture missing"
        );

        let values: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let addr = spawn_test_server(ml_backends, vec![cpu_model("model3")]).await;
        let mut client = connect(addr).await;

        let explicit = run_checkpoint(
            &mut client,
            vec![
                meta("model3"),
                tensor_chunk(vec![1, 16], f32_le_bytes(&values), true),
            ],
        )
        .await
        .expect("explicit [1, 16] checkpoint should succeed");

        let expanded = run_checkpoint(
            &mut client,
            vec![
                meta("model3"),
                tensor_chunk(vec![16], f32_le_bytes(&values), true),
            ],
        )
        .await
        .expect("bare [16] checkpoint should be auto-expanded and succeed");

        assert!(expanded.saw_output_chunk, "expected an output tensor chunk");
        let shape = expanded
            .output_shape
            .clone()
            .expect("expanded output shape");
        assert_eq!(
            shape.first().copied(),
            Some(1),
            "auto-expanded output must carry a leading batch dimension of 1, got {shape:?}"
        );
        assert_eq!(
            expanded.output_shape, explicit.output_shape,
            "auto-expanded [16] must yield the same output shape as explicit [1, 16]"
        );
        assert_eq!(
            expanded.output_bytes, explicit.output_bytes,
            "auto-expanded [16] must yield identical output bytes to explicit [1, 16]"
        );
    }

    /// A batch size above `max_batch_size` (10 for model3) is rejected with
    /// `InvalidArgument`.
    #[tokio::test]
    async fn rust_backend_rejects_batch_over_max() {
        let ml_backends = fixtures_dir();
        assert!(
            ml_backends.join("model3").is_dir(),
            "model3 fixture missing"
        );

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

    /// When a Python model declares an input contract, the validated tensor is
    /// piped into `main.py` on stdin. The fixture echoes the shape/count/sum it
    /// reads back; asserting on the derived sum proves the bytes arrived intact
    /// and in order, not merely that the process ran.
    #[tokio::test]
    async fn python_backend_receives_validated_tensor_on_stdin() {
        let (ml_backends, name) = make_python_model_with_contract(
            "python-stdin",
            "e2e_python_stdin",
            "input_shape: [16]\nmax_batch_size: 10\noutput_shape: [1]\n",
            STDIN_ECHO_MAIN_PY,
        );

        // Values 0..16 sum to 120; the Python side prints them as a float.
        let values: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let addr = spawn_test_server(ml_backends.clone(), vec![cpu_model(&name)]).await;
        let mut client = connect(addr).await;
        let collected = run_checkpoint(
            &mut client,
            vec![
                meta(&name),
                tensor_chunk(vec![1, 16], f32_le_bytes(&values), true),
            ],
        )
        .await
        .expect("python checkpoint with tensor should succeed");

        assert!(collected.saw_done, "stream must end with done=true");
        assert_eq!(collected.exit_code, 0, "successful main.py exits 0");
        assert!(
            collected.lines.iter().any(|l| l == "shape: 1,16"),
            "main.py should see the advertised shape, got {:?}",
            collected.lines
        );
        assert!(
            collected.lines.iter().any(|l| l == "count: 16"),
            "main.py should read all 16 values, got {:?}",
            collected.lines
        );
        assert!(
            collected.lines.iter().any(|l| l == "sum: 120.0"),
            "main.py should compute the sum of the received bytes, got {:?}",
            collected.lines
        );

        let _ = std::fs::remove_dir_all(&ml_backends);
    }

    /// The tensor-reply contract end to end with checkable math: a Python model
    /// reads the input tensor, sums it, and writes the sum as its output tensor.
    /// The server must stream that tensor back as an `output_chunk`. Asserting
    /// the decoded value equals the known sum proves the reply tensor is the
    /// model's real output, not merely that some tensor came back.
    #[tokio::test]
    async fn python_backend_streams_computed_output_tensor() {
        // Reads float32 stdin, writes the sum back as a [1] float32 tensor.
        let main_py = "\
import os, struct, sys
raw = sys.stdin.buffer.read()
count = len(raw) // 4
values = struct.unpack('<%df' % count, raw) if count else ()
total = float(sum(values))
with open(os.environ['NEREID_OUTPUT_PATH'], 'wb') as f:
    f.write(b'float32 1\\n')
    f.write(struct.pack('<f', total))
";
        let (ml_backends, name) = make_python_model_raw(
            "python-computed-output",
            "e2e_python_computed",
            "input_shape: [16]\nmax_batch_size: 10\noutput_shape: [1]\n",
            main_py,
        );

        // Values 0..16 sum to 120.
        let values: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let addr = spawn_test_server(ml_backends.clone(), vec![cpu_model(&name)]).await;
        let mut client = connect(addr).await;
        let collected = run_checkpoint(
            &mut client,
            vec![
                meta(&name),
                tensor_chunk(vec![1, 16], f32_le_bytes(&values), true),
            ],
        )
        .await
        .expect("python checkpoint with computed output should succeed");

        assert!(collected.saw_done, "stream must end with done=true");
        assert_eq!(collected.exit_code, 0);
        assert!(
            collected.saw_output_chunk,
            "expected a typed output tensor chunk from the Python model"
        );
        assert_eq!(
            collected.output_shape.as_deref(),
            Some([1].as_slice()),
            "output tensor shape"
        );
        let output: Vec<f32> = collected
            .output_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(
            output,
            vec![120.0],
            "output tensor must equal the input sum"
        );

        let _ = std::fs::remove_dir_all(&ml_backends);
    }

    /// The native Checkpoint TensorChunk carries no dtype and is float32 by
    /// contract. A Python model that writes a non-float32 framed output (here
    /// int32) is rejected rather than streaming mislabeled bytes to the client.
    #[tokio::test]
    async fn python_backend_rejects_non_float32_output() {
        let main_py = "\
import os, struct
with open(os.environ['NEREID_OUTPUT_PATH'], 'wb') as f:
    f.write(b'int32 2\\n')
    f.write(struct.pack('<2i', 3, 4))
";
        let (ml_backends, name) = make_python_model_raw(
            "checkpoint-int32",
            "e2e_checkpoint_int32",
            "output_shape: [2]\n",
            main_py,
        );
        let addr = spawn_test_server(ml_backends.clone(), vec![cpu_model(&name)]).await;
        let mut client = connect(addr).await;
        let status = run_checkpoint(&mut client, vec![meta(&name)])
            .await
            .expect_err("non-float32 Checkpoint output must be rejected");
        assert_eq!(status.code(), tonic::Code::FailedPrecondition, "{status:?}");

        let _ = std::fs::remove_dir_all(&ml_backends);
    }

    /// A contract on a Python model lets the server reject a shape mismatch with
    /// `InvalidArgument` before ever launching `main.py` — the reviewer's
    /// motivation for giving Python backends a contract.
    #[tokio::test]
    async fn python_backend_rejects_shape_mismatch() {
        let (ml_backends, name) = make_python_model_with_contract(
            "python-stdin-reject",
            "e2e_python_stdin_reject",
            "input_shape: [16]\nmax_batch_size: 10\noutput_shape: [1]\n",
            STDIN_ECHO_MAIN_PY,
        );

        // Contract expects a trailing dim of 16; send 15.
        let values: Vec<f32> = (0..15).map(|v| v as f32).collect();
        let addr = spawn_test_server(ml_backends.clone(), vec![cpu_model(&name)]).await;
        let mut client = connect(addr).await;
        let status = run_checkpoint(
            &mut client,
            vec![
                meta(&name),
                tensor_chunk(vec![1, 15], f32_le_bytes(&values), true),
            ],
        )
        .await
        .expect_err("shape mismatch must be rejected");

        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status:?}");

        let _ = std::fs::remove_dir_all(&ml_backends);
    }

    /// The batch-dimension auto-expansion applies to the Python path too: a bare
    /// `[16]` is expanded to `[1, 16]` before being advertised to `main.py`.
    #[tokio::test]
    async fn python_backend_expands_missing_batch_dim() {
        let (ml_backends, name) = make_python_model_with_contract(
            "python-stdin-expand",
            "e2e_python_stdin_expand",
            "input_shape: [16]\nmax_batch_size: 10\noutput_shape: [1]\n",
            STDIN_ECHO_MAIN_PY,
        );

        let values: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let addr = spawn_test_server(ml_backends.clone(), vec![cpu_model(&name)]).await;
        let mut client = connect(addr).await;
        let collected = run_checkpoint(
            &mut client,
            vec![
                meta(&name),
                tensor_chunk(vec![16], f32_le_bytes(&values), true),
            ],
        )
        .await
        .expect("bare [16] python checkpoint should be auto-expanded and succeed");

        assert!(
            collected.lines.iter().any(|l| l == "shape: 1,16"),
            "auto-expanded shape must be advertised as 1,16, got {:?}",
            collected.lines
        );
        assert!(
            collected.lines.iter().any(|l| l == "sum: 120.0"),
            "main.py should still receive all values, got {:?}",
            collected.lines
        );

        let _ = std::fs::remove_dir_all(&ml_backends);
    }

    /// A `main.py` that never reads stdin must not hang the response, even when
    /// the input tensor is larger than the OS pipe buffer. The server writes
    /// stdin on a dedicated thread (which blocks, then sees `BrokenPipe` when the
    /// process exits) so the stdout reader and terminal `done` still flow. The
    /// tensor here is ~800 KB — well past the typical 64 KB pipe buffer — so a
    /// naive write-then-read would deadlock.
    #[tokio::test]
    async fn python_backend_unread_large_stdin_does_not_deadlock() {
        let (ml_backends, name) = make_python_model_with_contract(
            "python-stdin-unread",
            "e2e_python_stdin_unread",
            "input_shape: [200000]\nmax_batch_size: 1\noutput_shape: [1]\n",
            // Note: never reads stdin.
            "print('ignored stdin')\n",
        );

        let values = vec![0.0f32; 200_000];
        let addr = spawn_test_server(ml_backends.clone(), vec![cpu_model(&name)]).await;
        let mut client = connect(addr).await;
        let collected = run_checkpoint(
            &mut client,
            vec![
                meta(&name),
                tensor_chunk(vec![1, 200_000], f32_le_bytes(&values), true),
            ],
        )
        .await
        .expect("checkpoint must complete even when main.py ignores a large stdin");

        assert!(collected.saw_done, "stream must end with done=true");
        assert_eq!(collected.exit_code, 0, "main.py exits 0");
        assert!(
            collected.lines.iter().any(|l| l == "ignored stdin"),
            "stdout must still flow while stdin is unread, got {:?}",
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
        assert!(
            ml_backends.join("model3").is_dir(),
            "model3 fixture missing"
        );

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
        assert!(
            ml_backends.join("model3").is_dir(),
            "model3 fixture missing"
        );

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
        assert!(
            ml_backends.join("model3").is_dir(),
            "model3 fixture missing"
        );

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
        assert!(
            ml_backends.join("model3").is_dir(),
            "model3 fixture missing"
        );

        let addr = spawn_test_server(ml_backends, vec![cpu_model("model3")]).await;
        let mut client = connect(addr).await;
        let status = run_checkpoint(&mut client, vec![])
            .await
            .expect_err("empty stream must be rejected");

        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status:?}");
    }
}
