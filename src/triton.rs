//! Triton-compatible KServe v2 inference surface (`GRPCInferenceService`).
//!
//! Implements the client-facing subset of NVIDIA Triton's gRPC protocol so that
//! a stock `tritonclient` (or any KServe v2 speaker) can drive nereid without
//! code changes. The wire format — package `inference`, service
//! `GRPCInferenceService`, message field numbers — is vendored verbatim from
//! upstream in `proto/grpc_service.proto`.
//!
//! Implemented: `ServerLive/Ready`, `ModelReady`, `ServerMetadata`,
//! `ModelMetadata`, unary `ModelInfer`, and streaming `ModelStreamInfer`. This
//! surface is backend-agnostic: it validates the request against the model's
//! [`Contract`], builds canonical [`Tensor`]s, and dispatches through
//! `ModelManager` — so every backend (TorchScript, ONNX, TensorFlow, Python) and
//! both single- and multi-tensor models flow through one path. The request
//! datatype must match the model's declared `data_type` (default `FP32`).
//! nereid serves a single implicit model version, `"1"`.
//!
//! Not implemented (deferred): `BYTES` (variable-length) tensors, the HTTP/REST
//! `/v2` mirror, Prometheus metrics, and the repository/config RPCs.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tokio::sync::mpsc;
use tonic::codegen::tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use crate::backend::{Contract, ModelManager, Tensor, TensorSpec};
use crate::dtype;
use crate::proto::grpc_inference_service_server::GrpcInferenceService;
use crate::proto::model_infer_request::InferInputTensor;
use crate::proto::model_infer_response::InferOutputTensor;
use crate::proto::model_metadata_response::TensorMetadata;
use crate::proto::{
    ModelInferRequest, ModelInferResponse, ModelMetadataRequest, ModelMetadataResponse,
    ModelReadyRequest, ModelReadyResponse, ModelStreamInferResponse, ServerLiveRequest,
    ServerLiveResponse, ServerMetadataRequest, ServerMetadataResponse, ServerReadyRequest,
    ServerReadyResponse,
};

/// KServe spells 32-bit float exactly this way; it's also the default datatype
/// for a model whose contract omits `data_type`.
const FP32: &str = "FP32";

/// nereid serves a single, implicit model version. A request may name it
/// explicitly (`"1"`) or leave it empty; any other version is unavailable.
const MODEL_VERSION: &str = "1";

/// Build the model's input tensors from a KServe `ModelInfer` request, in the
/// contract's declared input order. Backend-agnostic: it validates
/// datatype/shape/byte-length against the `Contract` and produces canonical
/// [`Tensor`]s that any backend consumes. Returns `(tensors, expected_batch)`
/// where `expected_batch` is the request's leading dimension when the model
/// declares a batch dimension (used later to validate the output batch).
///
/// A single-tensor model is just the one-input case; an output-only model has
/// no declared inputs, so the request must carry none.
fn build_input_tensors(
    contract: &Contract,
    request: &mut ModelInferRequest,
    model_name: &str,
) -> Result<(Vec<Tensor>, Option<i64>), Status> {
    if contract.inputs.is_empty() {
        // Output-only model: no input tensor expected. Reject stray client data.
        if !request.inputs.is_empty() || !request.raw_input_contents.is_empty() {
            return Err(Status::invalid_argument(format!(
                "model '{model_name}' declares no input tensor but the request carries \
                 {} input(s) and {} raw buffer(s)",
                request.inputs.len(),
                request.raw_input_contents.len()
            )));
        }
        return Ok((Vec::new(), None));
    }

    // Raw buffers pair positionally with request.inputs; typed contents are
    // matched per input instead.
    let use_raw = !request.raw_input_contents.is_empty();
    if use_raw && request.raw_input_contents.len() != request.inputs.len() {
        return Err(Status::invalid_argument(
            "raw_input_contents length must equal the number of input tensors",
        ));
    }
    // Map input tensors by name. Reject duplicate names: with raw_input_contents
    // paired positionally, a silent last-wins overwrite could bind the wrong raw
    // buffer to a tensor name.
    let mut by_name: HashMap<&str, (&InferInputTensor, usize)> =
        HashMap::with_capacity(request.inputs.len());
    for (i, inp) in request.inputs.iter().enumerate() {
        if by_name.insert(inp.name.as_str(), (inp, i)).is_some() {
            return Err(Status::invalid_argument(format!(
                "duplicate input tensor name '{}' in request for model '{model_name}'",
                inp.name
            )));
        }
    }
    // Reject inputs not declared by the model (silently dropping client data is
    // worse than an explicit error).
    let declared_inputs: HashSet<&str> = contract.inputs.iter().map(|s| s.name.as_str()).collect();
    if let Some(unknown) = request
        .inputs
        .iter()
        .find(|inp| !declared_inputs.contains(inp.name.as_str()))
    {
        return Err(Status::invalid_argument(format!(
            "model '{model_name}' has no input '{}'",
            unknown.name
        )));
    }

    let mut tensors = Vec::with_capacity(contract.inputs.len());
    let mut batch: Option<i64> = None;
    for spec in &contract.inputs {
        let (inp, idx) = *by_name.get(spec.name.as_str()).ok_or_else(|| {
            Status::invalid_argument(format!(
                "missing input tensor '{}' for model '{model_name}'",
                spec.name
            ))
        })?;
        if inp.datatype != spec.dtype {
            return Err(Status::invalid_argument(format!(
                "datatype mismatch for input '{}' of model '{model_name}': expected {}, got '{}'",
                spec.name, spec.dtype, inp.datatype
            )));
        }
        let (elem_size, canonical) = dtype::kserve_fixed_width(&spec.dtype).ok_or_else(|| {
            Status::invalid_argument(format!(
                "unsupported datatype '{}' for input '{}'",
                spec.dtype, spec.name
            ))
        })?;

        contract.validate_input_shape(spec, &inp.shape, model_name)?;
        let shape = contract.normalize_request_shape(spec, inp.shape.clone());

        let bytes = if use_raw {
            request.raw_input_contents[idx].clone()
        } else if spec.dtype == FP32 {
            match &inp.contents {
                Some(contents) => contents
                    .fp32_contents
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
                None => {
                    return Err(Status::invalid_argument(format!(
                        "input '{}' has neither raw_input_contents nor contents",
                        spec.name
                    )));
                }
            }
        } else {
            return Err(Status::invalid_argument(format!(
                "input '{}' ({}) requires raw_input_contents (typed contents is FP32-only)",
                spec.name, spec.dtype
            )));
        };

        let numel = shape
            .iter()
            .try_fold(1i64, |acc, dim| acc.checked_mul(*dim))
            .ok_or_else(|| Status::invalid_argument("input tensor shape overflow"))?;
        let expected_bytes = (numel as usize).saturating_mul(elem_size);
        if bytes.len() != expected_bytes {
            return Err(Status::invalid_argument(format!(
                "input '{}' byte length {} does not match shape {shape:?} \u{d7} {elem_size} bytes \
                 ({expected_bytes}) for model '{model_name}'",
                spec.name,
                bytes.len()
            )));
        }

        // Batch size must agree across all inputs.
        if contract.has_batch_dim() {
            let this_batch = shape.first().copied();
            match batch {
                None => batch = this_batch,
                Some(prev) if this_batch != Some(prev) => {
                    return Err(Status::invalid_argument(
                        "inconsistent batch sizes across input tensors",
                    ));
                }
                _ => {}
            }
        }

        tensors.push(Tensor {
            name: spec.name.clone(),
            shape,
            dtype: canonical.to_string(),
            data: bytes,
        });
    }
    Ok((tensors, batch))
}

/// Whether a requested model version is servable. nereid has no version
/// concept, so only the implicit version `"1"` (or an empty selector) exists.
fn version_available(model_version: &str) -> bool {
    model_version.is_empty() || model_version == MODEL_VERSION
}

/// The advertised metadata shape for `spec`: a leading `-1` (variable batch)
/// when a batch dimension is declared, followed by the declared dims.
fn tensor_metadata(contract: &Contract, spec: &TensorSpec) -> TensorMetadata {
    TensorMetadata {
        name: spec.name.clone(),
        datatype: spec.dtype.clone(),
        shape: contract.metadata_shape(spec),
    }
}

#[derive(Clone)]
pub struct TritonService {
    model_manager: Arc<ModelManager>,
}

impl TritonService {
    pub fn new(model_manager: Arc<ModelManager>) -> Self {
        Self { model_manager }
    }
}

#[tonic::async_trait]
impl GrpcInferenceService for TritonService {
    async fn server_live(
        &self,
        _request: Request<ServerLiveRequest>,
    ) -> Result<Response<ServerLiveResponse>, Status> {
        Ok(Response::new(ServerLiveResponse { live: true }))
    }

    async fn server_ready(
        &self,
        _request: Request<ServerReadyRequest>,
    ) -> Result<Response<ServerReadyResponse>, Status> {
        Ok(Response::new(ServerReadyResponse { ready: true }))
    }

    async fn model_ready(
        &self,
        request: Request<ModelReadyRequest>,
    ) -> Result<Response<ModelReadyResponse>, Status> {
        let request = request.into_inner();
        Ok(Response::new(ModelReadyResponse {
            ready: self.model_manager.is_configured(&request.name)
                && version_available(&request.version),
        }))
    }

    async fn server_metadata(
        &self,
        _request: Request<ServerMetadataRequest>,
    ) -> Result<Response<ServerMetadataResponse>, Status> {
        Ok(Response::new(ServerMetadataResponse {
            name: "nereid".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            extensions: Vec::new(),
        }))
    }

    async fn model_metadata(
        &self,
        request: Request<ModelMetadataRequest>,
    ) -> Result<Response<ModelMetadataResponse>, Status> {
        let request = request.into_inner();
        let name = request.name;
        if !self.model_manager.is_configured(&name) || !version_available(&request.version) {
            return Err(Status::not_found(format!(
                "model '{name}' (version '{}') is not available",
                request.version
            )));
        }

        let contract = self.model_manager.contract(&name).ok_or_else(|| {
            Status::not_found(format!("model '{name}' is not configured in nereid.yaml"))
        })?;
        let platform = self.model_manager.platform(&name).unwrap_or("").to_string();

        let (inputs, outputs) = if contract.strict_output_dtype {
            // Multi-tensor models advertise every named input/output verbatim.
            (
                contract
                    .inputs
                    .iter()
                    .map(|spec| tensor_metadata(contract, spec))
                    .collect(),
                contract
                    .outputs
                    .iter()
                    .map(|spec| tensor_metadata(contract, spec))
                    .collect(),
            )
        } else {
            // Single-tensor (flat) models advertise one `input`/`output`. The
            // datatype comes from the declared contract (default FP32). A model
            // that declares no input (output-only Python) advertises none; a
            // model that declares no output shape falls back to `[-1]`.
            let datatype = contract
                .inputs
                .first()
                .or_else(|| contract.outputs.first())
                .map_or(FP32, |spec| spec.dtype.as_str())
                .to_string();
            let inputs = match contract.inputs.first() {
                Some(spec) => vec![TensorMetadata {
                    name: "input".to_string(),
                    datatype: datatype.clone(),
                    shape: contract.metadata_shape(spec),
                }],
                None => Vec::new(),
            };
            let output_shape = contract
                .outputs
                .first()
                .map_or_else(|| vec![-1], |spec| contract.metadata_shape(spec));
            let outputs = vec![TensorMetadata {
                name: "output".to_string(),
                datatype,
                shape: output_shape,
            }];
            (inputs, outputs)
        };

        Ok(Response::new(ModelMetadataResponse {
            name,
            versions: vec![MODEL_VERSION.to_string()],
            platform,
            inputs,
            outputs,
        }))
    }

    async fn model_infer(
        &self,
        request: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        Ok(Response::new(self.infer_once(request.into_inner()).await?))
    }

    type ModelStreamInferStream = ReceiverStream<Result<ModelStreamInferResponse, Status>>;

    async fn model_stream_infer(
        &self,
        request: Request<tonic::Streaming<ModelInferRequest>>,
    ) -> Result<Response<Self::ModelStreamInferStream>, Status> {
        let mut in_stream = request.into_inner();
        let (tx, rx) = mpsc::channel::<Result<ModelStreamInferResponse, Status>>(16);
        let service = self.clone();

        tokio::spawn(async move {
            loop {
                match in_stream.message().await {
                    Ok(Some(req)) => {
                        // Per the Triton contract, a per-request failure is
                        // reported inline via `error_message` and the stream
                        // continues; only a transport-level read error ends it.
                        let response = match service.infer_once(req).await {
                            Ok(infer) => ModelStreamInferResponse {
                                error_message: String::new(),
                                infer_response: Some(infer),
                            },
                            Err(status) => ModelStreamInferResponse {
                                error_message: status.message().to_string(),
                                infer_response: None,
                            },
                        };
                        if tx.send(Ok(response)).await.is_err() {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(status) => {
                        let _ = tx.send(Err(status)).await;
                        break;
                    }
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

impl TritonService {
    /// The shared body of a single inference request, used by both the unary
    /// `ModelInfer` and the streaming `ModelStreamInfer` RPCs. Handles version
    /// selection, datatype/shape/batch validation, backend-agnostic dispatch
    /// through `ModelManager` (with its per-model backpressure), and output
    /// serialization — one path for every backend and for single- and
    /// multi-tensor models alike.
    async fn infer_once(
        &self,
        mut request: ModelInferRequest,
    ) -> Result<ModelInferResponse, Status> {
        let model_name = request.model_name.trim().to_string();
        if model_name.is_empty() {
            return Err(Status::invalid_argument("model_name is required"));
        }
        if !version_available(&request.model_version) {
            return Err(Status::not_found(format!(
                "model '{model_name}' version '{}' is not available; nereid serves version {MODEL_VERSION}",
                request.model_version
            )));
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

        let (inputs, expected_batch) = build_input_tensors(&contract, &mut request, &model_name)?;
        let outputs = self.model_manager.infer(&model_name, inputs).await?;

        if contract.strict_output_dtype {
            serialize_multi(&contract, &request, model_name, outputs)
        } else {
            serialize_single(&contract, &request, model_name, outputs, expected_batch)
        }
    }
}

/// Serialize a single-tensor model's output. The one output is named `output`
/// and its datatype is whatever the model produced (the flat contract does not
/// pin output dtypes); when the model declares an output shape, it is validated.
/// A client-requested output name other than `output` is rejected.
fn serialize_single(
    contract: &Contract,
    request: &ModelInferRequest,
    model_name: String,
    mut outputs: Vec<Tensor>,
    expected_batch: Option<i64>,
) -> Result<ModelInferResponse, Status> {
    const OUTPUT_NAME: &str = "output";
    if let Some(unknown) = request.outputs.iter().find(|o| o.name != OUTPUT_NAME) {
        return Err(Status::invalid_argument(format!(
            "model '{model_name}' has no output '{}'; its only output is '{OUTPUT_NAME}'",
            unknown.name
        )));
    }
    if outputs.len() != 1 {
        return Err(Status::internal(format!(
            "single-tensor model '{model_name}' returned {} outputs, expected 1",
            outputs.len()
        )));
    }
    let out = outputs.remove(0);
    if let Some(spec) = contract.outputs.first() {
        contract.validate_output_shape(spec, &out.shape, expected_batch, &model_name)?;
    }
    let (out_kserve, _) = dtype::canonical_to_kserve(&out.dtype).ok_or_else(|| {
        Status::internal(format!(
            "model '{model_name}' returned unsupported output dtype '{}'",
            out.dtype
        ))
    })?;
    Ok(ModelInferResponse {
        model_name,
        model_version: MODEL_VERSION.to_string(),
        id: request.id.clone(),
        parameters: Default::default(),
        outputs: vec![InferOutputTensor {
            name: OUTPUT_NAME.to_string(),
            datatype: out_kserve.to_string(),
            shape: out.shape,
            parameters: Default::default(),
            contents: None,
        }],
        raw_output_contents: vec![out.data],
    })
}

/// Serialize a multi-tensor model's outputs (in contract order), returning all
/// declared outputs or the client-requested subset. Each returned dtype must
/// match the declared contract — a mismatch is a model/config bug. A requested
/// output name that isn't declared is rejected rather than silently omitted.
fn serialize_multi(
    contract: &Contract,
    request: &ModelInferRequest,
    model_name: String,
    outputs: Vec<Tensor>,
) -> Result<ModelInferResponse, Status> {
    if outputs.len() != contract.outputs.len() {
        return Err(Status::internal(format!(
            "model '{model_name}' returned {} output tensors but its contract declares {}",
            outputs.len(),
            contract.outputs.len()
        )));
    }

    let requested: Option<HashSet<&str>> = if request.outputs.is_empty() {
        None
    } else {
        let declared_outputs: HashSet<&str> =
            contract.outputs.iter().map(|o| o.name.as_str()).collect();
        if let Some(unknown) = request
            .outputs
            .iter()
            .find(|o| !declared_outputs.contains(o.name.as_str()))
        {
            return Err(Status::invalid_argument(format!(
                "model '{model_name}' has no output '{}'",
                unknown.name
            )));
        }
        Some(request.outputs.iter().map(|o| o.name.as_str()).collect())
    };

    let mut out_tensors = Vec::new();
    let mut raw_outputs = Vec::new();
    for (spec, out) in contract.outputs.iter().zip(outputs) {
        if let Some(requested) = &requested
            && !requested.contains(spec.name.as_str())
        {
            continue;
        }
        let (kserve, _) = dtype::canonical_to_kserve(&out.dtype).ok_or_else(|| {
            Status::internal(format!(
                "model '{model_name}' output '{}' has unsupported dtype '{}'",
                spec.name, out.dtype
            ))
        })?;
        if kserve != spec.dtype {
            return Err(Status::internal(format!(
                "model '{model_name}' output '{}' returned dtype {kserve} but the contract \
                 declares {}",
                spec.name, spec.dtype
            )));
        }
        out_tensors.push(InferOutputTensor {
            name: spec.name.clone(),
            datatype: kserve.to_string(),
            shape: out.shape,
            parameters: Default::default(),
            contents: None,
        });
        raw_outputs.push(out.data);
    }

    Ok(ModelInferResponse {
        model_name,
        model_version: MODEL_VERSION.to_string(),
        id: request.id.clone(),
        parameters: Default::default(),
        outputs: out_tensors,
        raw_output_contents: raw_outputs,
    })
}

/// End-to-end tests for the Triton-compatible surface. These drive the
/// generated KServe v2 client over a loopback socket into a live
/// `GrpcInferenceServiceServer`, mirroring exactly what a stock `tritonclient`
/// would do on the wire.
#[cfg(test)]
mod triton_e2e_tests {
    use super::*;
    use crate::config::{ModelConfig, ModelDevice, ServerConfig, ServerSection};
    use crate::proto::grpc_inference_service_client::GrpcInferenceServiceClient;
    use crate::proto::grpc_inference_service_server::GrpcInferenceServiceServer;
    use std::net::SocketAddr;
    use std::path::{Path, PathBuf};
    use tch::{CModule, Device, Tensor};
    use tokio::net::TcpListener;
    use tokio_stream::wrappers::TcpListenerStream;
    use tonic::transport::Server;

    fn fixtures_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ml-backends")
    }

    /// Run a single-input/single-output `.pt` model's forward pass directly and
    /// serialize its float32 output to little-endian bytes — the "expected"
    /// oracle the server's output is compared against.
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

    /// Spawn a Triton server for `model_name` found in the committed fixtures.
    async fn spawn_triton_server(model_name: &str) -> SocketAddr {
        spawn_triton_server_at(fixtures_dir(), model_name).await
    }

    async fn spawn_triton_server_at(ml_backends: PathBuf, model_name: &str) -> SocketAddr {
        spawn_triton_server_qc(ml_backends, model_name, 4).await
    }

    /// Like [`spawn_triton_server_at`] but with an explicit `queue_capacity`,
    /// for exercising backpressure.
    async fn spawn_triton_server_qc(
        ml_backends: PathBuf,
        model_name: &str,
        queue_capacity: usize,
    ) -> SocketAddr {
        let config = ServerConfig {
            server: ServerSection {
                bind_addr: "127.0.0.1:0".to_string(),
                ml_backends_path: ml_backends.to_string_lossy().into_owned(),
            },
            models: vec![ModelConfig {
                name: model_name.to_string(),
                device: ModelDevice::Cpu,
                queue_capacity,
                backend: None,
                signature: None,
            }],
        };
        let model_manager =
            Arc::new(ModelManager::from_config(&config).expect("model manager should build"));
        let triton = TritonService::new(model_manager);

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind ephemeral port");
        let addr = listener.local_addr().expect("local addr");
        let incoming = TcpListenerStream::new(listener);
        tokio::spawn(async move {
            let _ = Server::builder()
                .add_service(GrpcInferenceServiceServer::new(triton))
                .serve_with_incoming(incoming)
                .await;
        });
        addr
    }

    async fn connect(addr: SocketAddr) -> GrpcInferenceServiceClient<tonic::transport::Channel> {
        GrpcInferenceServiceClient::connect(format!("http://{addr}"))
            .await
            .expect("client should connect")
    }

    fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// The centerpiece: bytes returned over the Triton `ModelInfer` RPC must be
    /// bit-for-bit identical to running model3's forward pass directly — the
    /// same guarantee the native Checkpoint path makes, now via the KServe wire
    /// format (raw_input_contents in, raw_output_contents out).
    #[tokio::test]
    async fn model_infer_matches_direct_forward_pass() {
        let model_path = fixtures_dir().join("model3").join("mlp.pt");
        assert!(model_path.is_file(), "model3 fixture missing");

        let input_values: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let (expected_shape, expected_bytes) =
            direct_forward_f32(&model_path, &input_values, &[1, 16]);

        let addr = spawn_triton_server("model3").await;
        let mut client = connect(addr).await;
        let response = client
            .model_infer(ModelInferRequest {
                model_name: "model3".to_string(),
                model_version: String::new(),
                id: "req-1".to_string(),
                parameters: Default::default(),
                inputs: vec![InferInputTensor {
                    name: "input".to_string(),
                    datatype: FP32.to_string(),
                    shape: vec![1, 16],
                    parameters: Default::default(),
                    contents: None,
                }],
                outputs: Vec::new(),
                raw_input_contents: vec![f32_le_bytes(&input_values)],
            })
            .await
            .expect("model_infer should succeed")
            .into_inner();

        assert_eq!(response.id, "req-1", "request id must be echoed");
        assert_eq!(response.outputs.len(), 1, "expected one output tensor");
        let output = &response.outputs[0];
        assert_eq!(output.datatype, FP32);
        assert_eq!(output.shape, expected_shape, "output shape must match");
        assert_eq!(
            response.raw_output_contents,
            vec![expected_bytes],
            "raw output bytes must match direct forward pass byte-for-byte"
        );
    }

    /// Typed `fp32_contents` is accepted as an alternative to raw bytes and
    /// yields the same result.
    #[tokio::test]
    async fn model_infer_accepts_typed_contents() {
        assert!(fixtures_dir().join("model3").is_dir(), "model3 missing");
        let input_values: Vec<f32> = (0..16).map(|v| v as f32).collect();

        let addr = spawn_triton_server("model3").await;
        let mut client = connect(addr).await;
        let response = client
            .model_infer(ModelInferRequest {
                model_name: "model3".to_string(),
                model_version: String::new(),
                id: String::new(),
                parameters: Default::default(),
                inputs: vec![InferInputTensor {
                    name: "input".to_string(),
                    datatype: FP32.to_string(),
                    shape: vec![1, 16],
                    parameters: Default::default(),
                    contents: Some(crate::proto::InferTensorContents {
                        fp32_contents: input_values.clone(),
                        ..Default::default()
                    }),
                }],
                outputs: Vec::new(),
                raw_input_contents: Vec::new(),
            })
            .await
            .expect("typed-contents model_infer should succeed")
            .into_inner();

        assert_eq!(response.outputs.len(), 1);
        assert!(
            !response.raw_output_contents.is_empty(),
            "expected raw output bytes"
        );
    }

    /// Health and metadata RPCs report what a KServe client expects: server
    /// live/ready, model ready, and the model's typed input/output shapes.
    #[tokio::test]
    async fn health_and_metadata() {
        assert!(fixtures_dir().join("model3").is_dir(), "model3 missing");
        let addr = spawn_triton_server("model3").await;
        let mut client = connect(addr).await;

        assert!(
            client
                .server_live(ServerLiveRequest {})
                .await
                .expect("server_live")
                .into_inner()
                .live
        );
        assert!(
            client
                .server_ready(ServerReadyRequest {})
                .await
                .expect("server_ready")
                .into_inner()
                .ready
        );
        assert!(
            client
                .model_ready(ModelReadyRequest {
                    name: "model3".to_string(),
                    version: String::new(),
                })
                .await
                .expect("model_ready")
                .into_inner()
                .ready
        );
        assert!(
            !client
                .model_ready(ModelReadyRequest {
                    name: "nope".to_string(),
                    version: String::new(),
                })
                .await
                .expect("model_ready unknown")
                .into_inner()
                .ready,
            "unknown model must not be ready"
        );

        let meta = client
            .model_metadata(ModelMetadataRequest {
                name: "model3".to_string(),
                version: String::new(),
            })
            .await
            .expect("model_metadata")
            .into_inner();
        assert_eq!(meta.name, "model3");
        assert_eq!(meta.platform, "pytorch_libtorch");
        assert_eq!(meta.inputs.len(), 1);
        assert_eq!(meta.inputs[0].datatype, FP32);
        // model3's contract is input_shape [16], max_batch 10 -> advertised
        // shape carries a leading -1 batch dimension.
        assert_eq!(meta.inputs[0].shape, vec![-1, 16]);
    }

    /// A tensor-capable Python model (the committed `pymul` fixture, which
    /// declares `output_shape` and computes `input * 2 + 1`) answers
    /// `ModelInfer` with a numerically-correct tensor. Asserting the actual
    /// arithmetic — not merely "a float32 tensor came back" — is what proves
    /// the stdin-in / framed-file-out tensor contract works end to end.
    #[tokio::test]
    async fn python_model_infer_returns_correct_tensor() {
        assert!(
            fixtures_dir().join("pymul").join("main.py").is_file(),
            "pymul fixture missing"
        );

        let input_values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let expected: Vec<f32> = input_values.iter().map(|v| v * 2.0 + 1.0).collect();

        let addr = spawn_triton_server("pymul").await;
        let mut client = connect(addr).await;
        let response = client
            .model_infer(ModelInferRequest {
                model_name: "pymul".to_string(),
                model_version: String::new(),
                id: "py-1".to_string(),
                parameters: Default::default(),
                inputs: vec![InferInputTensor {
                    name: "input".to_string(),
                    datatype: FP32.to_string(),
                    shape: vec![1, 4],
                    parameters: Default::default(),
                    contents: None,
                }],
                outputs: Vec::new(),
                raw_input_contents: vec![f32_le_bytes(&input_values)],
            })
            .await
            .expect("python model_infer should succeed")
            .into_inner();

        assert_eq!(response.id, "py-1");
        assert_eq!(response.outputs.len(), 1);
        assert_eq!(response.outputs[0].shape, vec![1, 4], "output shape");
        let raw = &response.raw_output_contents[0];
        let got: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(got, expected, "pymul must compute input * 2 + 1");
    }

    /// The ONNX backend (feature `onnx`) serves the committed `onnxadd` fixture
    /// (`output = input + 1`) end to end over ModelInfer, exercising native
    /// backend detection, the ort session, and the tensor round-trip. Runs on
    /// CPU, so it needs no GPU. Generate the fixture with
    /// `scripts/make_example_models.py`.
    #[cfg(feature = "onnx")]
    #[tokio::test]
    async fn onnx_model_infer_returns_input_plus_one() {
        assert!(
            fixtures_dir().join("onnxadd").join("model.onnx").is_file(),
            "onnxadd fixture missing (run scripts/make_example_models.py)"
        );
        let input_values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let expected: Vec<f32> = input_values.iter().map(|v| v + 1.0).collect();

        let addr = spawn_triton_server("onnxadd").await;
        let mut client = connect(addr).await;

        let meta = client
            .model_metadata(ModelMetadataRequest {
                name: "onnxadd".to_string(),
                version: String::new(),
            })
            .await
            .expect("model_metadata")
            .into_inner();
        assert_eq!(meta.platform, "onnxruntime_onnx");

        let response = client
            .model_infer(ModelInferRequest {
                model_name: "onnxadd".to_string(),
                model_version: String::new(),
                id: "onnx-1".to_string(),
                parameters: Default::default(),
                inputs: vec![InferInputTensor {
                    name: "input".to_string(),
                    datatype: FP32.to_string(),
                    shape: vec![1, 4],
                    parameters: Default::default(),
                    contents: None,
                }],
                outputs: Vec::new(),
                raw_input_contents: vec![f32_le_bytes(&input_values)],
            })
            .await
            .expect("onnx model_infer should succeed")
            .into_inner();

        assert_eq!(response.outputs.len(), 1);
        assert_eq!(response.outputs[0].shape, vec![1, 4]);
        let raw = &response.raw_output_contents[0];
        let got: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(got, expected, "onnxadd must compute input + 1");
    }

    /// The TensorFlow backend (feature `tensorflow`) serves the committed `tfadd`
    /// SavedModel fixture (`output = input + 1`) end to end over ModelInfer,
    /// exercising native detection, the libtensorflow SavedModel session, and the
    /// tensor round-trip. CPU-only. Generate the fixture with
    /// `scripts/make_example_models.py` (needs `tensorflow`).
    #[cfg(feature = "tensorflow")]
    #[tokio::test]
    async fn tensorflow_model_infer_returns_input_plus_one() {
        assert!(
            fixtures_dir()
                .join("tfadd")
                .join("saved_model.pb")
                .is_file(),
            "tfadd fixture missing (run scripts/make_example_models.py)"
        );
        let input_values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let expected: Vec<f32> = input_values.iter().map(|v| v + 1.0).collect();

        let addr = spawn_triton_server("tfadd").await;
        let mut client = connect(addr).await;

        let meta = client
            .model_metadata(ModelMetadataRequest {
                name: "tfadd".to_string(),
                version: String::new(),
            })
            .await
            .expect("model_metadata")
            .into_inner();
        assert_eq!(meta.platform, "tensorflow_savedmodel");

        let response = client
            .model_infer(ModelInferRequest {
                model_name: "tfadd".to_string(),
                model_version: String::new(),
                id: "tf-1".to_string(),
                parameters: Default::default(),
                inputs: vec![InferInputTensor {
                    name: "input".to_string(),
                    datatype: FP32.to_string(),
                    shape: vec![1, 4],
                    parameters: Default::default(),
                    contents: None,
                }],
                outputs: Vec::new(),
                raw_input_contents: vec![f32_le_bytes(&input_values)],
            })
            .await
            .expect("tensorflow model_infer should succeed")
            .into_inner();

        assert_eq!(response.outputs.len(), 1);
        assert_eq!(response.outputs[0].shape, vec![1, 4]);
        let raw = &response.raw_output_contents[0];
        let got: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(got, expected, "tfadd must compute input + 1");
    }

    /// The compile-time C++ backend (feature `cxx`) serves the committed `cxxadd`
    /// model (a `cxx`-bridged C++ `output = input + 1`, compiled into the server)
    /// end to end over ModelInfer. The C++ code is registered by name, so the
    /// folder only supplies the contract and the model uses `backend: "cxx"`. It
    /// flows through the same `Backend` dispatch as every other engine.
    #[cfg(feature = "cxx")]
    #[tokio::test]
    async fn cxx_model_infer_returns_input_plus_one() {
        use crate::config::{ModelConfig, ModelDevice, ServerConfig, ServerSection};
        assert!(
            fixtures_dir()
                .join("cxxadd")
                .join("model_inference.textproto")
                .is_file(),
            "cxxadd fixture missing"
        );
        let input_values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let expected: Vec<f32> = input_values.iter().map(|v| v + 1.0).collect();

        let config = ServerConfig {
            server: ServerSection {
                bind_addr: "127.0.0.1:0".to_string(),
                ml_backends_path: fixtures_dir().to_string_lossy().into_owned(),
            },
            models: vec![ModelConfig {
                name: "cxxadd".to_string(),
                device: ModelDevice::Cpu,
                queue_capacity: 4,
                // Declared, not detected: the C++ is compiled in, so there is
                // nothing in the folder to detect it from.
                backend: Some("cxx".to_string()),
                signature: None,
            }],
        };
        let model_manager =
            Arc::new(ModelManager::from_config(&config).expect("cxx model should build"));
        let triton = TritonService::new(model_manager);
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = Server::builder()
                .add_service(GrpcInferenceServiceServer::new(triton))
                .serve_with_incoming(TcpListenerStream::new(listener))
                .await;
        });

        let mut client = connect(addr).await;
        let meta = client
            .model_metadata(ModelMetadataRequest {
                name: "cxxadd".to_string(),
                version: String::new(),
            })
            .await
            .expect("model_metadata")
            .into_inner();
        assert_eq!(meta.platform, "nereid_cxx");

        let response = client
            .model_infer(ModelInferRequest {
                model_name: "cxxadd".to_string(),
                model_version: String::new(),
                id: "cxx-1".to_string(),
                parameters: Default::default(),
                inputs: vec![InferInputTensor {
                    name: "input".to_string(),
                    datatype: FP32.to_string(),
                    shape: vec![1, 4],
                    parameters: Default::default(),
                    contents: None,
                }],
                outputs: Vec::new(),
                raw_input_contents: vec![f32_le_bytes(&input_values)],
            })
            .await
            .expect("cxx model_infer should succeed")
            .into_inner();

        assert_eq!(response.outputs.len(), 1);
        assert_eq!(response.outputs[0].shape, vec![1, 4]);
        let raw = &response.raw_output_contents[0];
        let got: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(got, expected, "cxxadd must compute input + 1");
    }

    /// GPU smoke test: serve the ONNX `onnxadd` fixture on `device: cuda`, which
    /// drives ort's CUDA execution provider. Ignored by default (needs an NVIDIA
    /// GPU with a working CUDA + cuDNN stack); run with `--ignored`. A pass proves
    /// the cuda-config path is wired and produces correct results; ort falls back
    /// to CPU when the GPU stack is unavailable, so it is not by itself proof the
    /// math ran on the GPU (watch ort's stderr for the active execution provider).
    #[cfg(feature = "onnx")]
    #[tokio::test]
    #[ignore = "requires an NVIDIA GPU with CUDA/cuDNN"]
    async fn onnx_cuda_smoke() {
        use crate::config::{ModelConfig, ModelDevice, ServerConfig, ServerSection};
        assert!(
            fixtures_dir().join("onnxadd").join("model.onnx").is_file(),
            "onnxadd fixture missing"
        );
        let config = ServerConfig {
            server: ServerSection {
                bind_addr: "127.0.0.1:0".to_string(),
                ml_backends_path: fixtures_dir().to_string_lossy().into_owned(),
            },
            models: vec![ModelConfig {
                name: "onnxadd".to_string(),
                device: ModelDevice::Cuda(0),
                queue_capacity: 4,
                backend: None,
                signature: None,
            }],
        };
        let model_manager =
            Arc::new(ModelManager::from_config(&config).expect("cuda onnx model should build"));
        let triton = TritonService::new(model_manager);
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().unwrap();
        let incoming = TcpListenerStream::new(listener);
        tokio::spawn(async move {
            let _ = Server::builder()
                .add_service(GrpcInferenceServiceServer::new(triton))
                .serve_with_incoming(incoming)
                .await;
        });

        let mut client = connect(addr).await;
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let response = client
            .model_infer(ModelInferRequest {
                model_name: "onnxadd".to_string(),
                model_version: String::new(),
                id: "cuda-1".to_string(),
                parameters: Default::default(),
                inputs: vec![InferInputTensor {
                    name: "input".to_string(),
                    datatype: FP32.to_string(),
                    shape: vec![1, 4],
                    parameters: Default::default(),
                    contents: None,
                }],
                outputs: Vec::new(),
                raw_input_contents: vec![f32_le_bytes(&input)],
            })
            .await
            .expect("cuda onnx model_infer should succeed")
            .into_inner();
        let raw = &response.raw_output_contents[0];
        let got: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(
            got,
            vec![2.0, 3.0, 4.0, 5.0],
            "onnxadd on cuda must compute input + 1"
        );
    }

    /// The client may request the model's output by its real name (`output`) —
    /// which returns a tensor named `output` — but requesting an unknown output
    /// name is rejected rather than silently renamed.
    #[tokio::test]
    async fn model_infer_validates_requested_output_name() {
        use crate::proto::model_infer_request::InferRequestedOutputTensor;
        assert!(fixtures_dir().join("model3").is_dir(), "model3 missing");
        let values: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let addr = spawn_triton_server("model3").await;
        let mut client = connect(addr).await;

        let request = |out_name: &str| ModelInferRequest {
            model_name: "model3".to_string(),
            model_version: String::new(),
            id: String::new(),
            parameters: Default::default(),
            inputs: vec![InferInputTensor {
                name: "input".to_string(),
                datatype: FP32.to_string(),
                shape: vec![1, 16],
                parameters: Default::default(),
                contents: None,
            }],
            outputs: vec![InferRequestedOutputTensor {
                name: out_name.to_string(),
                parameters: Default::default(),
            }],
            raw_input_contents: vec![f32_le_bytes(&values)],
        };

        // Requesting the real output name works and the tensor keeps that name.
        let ok = client
            .model_infer(request("output"))
            .await
            .expect("requesting 'output' should succeed")
            .into_inner();
        assert_eq!(ok.outputs[0].name, "output");

        // Requesting an unknown output name is rejected.
        let status = client
            .model_infer(request("not_a_real_output"))
            .await
            .expect_err("unknown requested output must be rejected");
        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status:?}");
    }

    /// Create an inline temp Python model under a fresh `ml-backends` dir and
    /// return `(ml_backends_path, model_name)`. Empty `requirements.txt` keeps
    /// the venv build cheap.
    fn make_temp_python_model(tag: &str, main_py: &str, textproto: &str) -> (PathBuf, String) {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let ml_backends = std::env::temp_dir().join(format!("nereid-triton-{tag}-{nanos}"));
        let model_dir = ml_backends.join("m");
        std::fs::create_dir_all(&model_dir).expect("create model dir");
        std::fs::write(model_dir.join("requirements.txt"), b"\n").expect("requirements");
        std::fs::write(model_dir.join("main.py"), main_py.as_bytes()).expect("main.py");
        std::fs::write(
            model_dir.join("model_inference.textproto"),
            textproto.as_bytes(),
        )
        .expect("textproto");
        (ml_backends, "m".to_string())
    }

    fn infer_request(model: &str, shape: Vec<i64>, bytes: Vec<u8>) -> ModelInferRequest {
        ModelInferRequest {
            model_name: model.to_string(),
            model_version: String::new(),
            id: String::new(),
            parameters: Default::default(),
            inputs: vec![InferInputTensor {
                name: "input".to_string(),
                datatype: FP32.to_string(),
                shape,
                parameters: Default::default(),
                contents: None,
            }],
            outputs: Vec::new(),
            raw_input_contents: vec![bytes],
        }
    }

    /// A model whose output shape contradicts its declared `output_shape` is
    /// rejected rather than served as a silently-wrong tensor. Here the model
    /// writes a self-consistent `[1, 5]` tensor while declaring `output_shape:
    /// [4]`, so `parse_framed_tensor` succeeds but `validate_output_shape`
    /// catches the mismatch.
    #[tokio::test]
    async fn python_model_infer_rejects_wrong_output_shape() {
        let main_py = "\
import os, struct, sys
sys.stdin.buffer.read()
with open(os.environ['NEREID_OUTPUT_PATH'], 'wb') as f:
    f.write(b'float32 1,5\\n')
    f.write(struct.pack('<5f', 1.0, 2.0, 3.0, 4.0, 5.0))
";
        let (ml_backends, name) = make_temp_python_model(
            "wrongshape",
            main_py,
            "input_shape: [4]\nmax_batch_size: 4\noutput_shape: [4]\n",
        );

        let addr = spawn_triton_server_at(ml_backends.clone(), &name).await;
        let mut client = connect(addr).await;
        let status = client
            .model_infer(infer_request(
                &name,
                vec![1, 4],
                f32_le_bytes(&[1.0, 2.0, 3.0, 4.0]),
            ))
            .await
            .expect_err("output-shape mismatch must be rejected");
        // A wrong-shaped model output is a model/config bug -> Internal.
        assert_eq!(status.code(), tonic::Code::Internal, "{status:?}");

        let _ = std::fs::remove_dir_all(&ml_backends);
    }

    /// Deadlock guard for the Python ModelInfer path, analogous to the
    /// Checkpoint path's large-unread-stdin test. The model is fed a ~800 KB
    /// stdin tensor AND emits a large volume of stdout (far past the ~64 KB pipe
    /// buffer) before exiting. If stdin writing and stdout/stderr draining
    /// weren't concurrent, the child would wedge on a full stdout pipe while the
    /// server blocked on `wait()`. The call must complete with a correct tensor.
    #[tokio::test]
    async fn python_model_infer_large_io_no_deadlock() {
        let n = 200_000usize;
        let main_py = "\
import array, os, sys
a = array.array('f')
a.frombytes(sys.stdin.buffer.read())
out = array.array('f', (v * 2.0 + 1.0 for v in a))
# Flood stdout well past the OS pipe buffer before writing the result.
for i in range(8000):
    print('log line', i)
shape = os.environ['NEREID_INPUT_SHAPE']
with open(os.environ['NEREID_OUTPUT_PATH'], 'wb') as f:
    f.write(('float32 ' + shape + '\\n').encode('utf-8'))
    f.write(out.tobytes())
";
        let (ml_backends, name) = make_temp_python_model(
            "largeio",
            main_py,
            &format!("input_shape: [{n}]\nmax_batch_size: 1\noutput_shape: [{n}]\n"),
        );

        let input = vec![1.0f32; n];
        let addr = spawn_triton_server_at(ml_backends.clone(), &name).await;
        let mut client = connect(addr).await;
        let response = client
            .model_infer(infer_request(
                &name,
                vec![1, n as i64],
                f32_le_bytes(&input),
            ))
            .await
            .expect("large-io infer must complete without deadlock")
            .into_inner();

        assert_eq!(response.outputs[0].shape, vec![1, n as i64]);
        let raw = &response.raw_output_contents[0];
        assert_eq!(raw.len(), n * 4, "output element count");
        let first = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        let last_off = (n - 1) * 4;
        let last = f32::from_le_bytes([
            raw[last_off],
            raw[last_off + 1],
            raw[last_off + 2],
            raw[last_off + 3],
        ]);
        assert_eq!(first, 3.0, "1 * 2 + 1");
        assert_eq!(last, 3.0, "1 * 2 + 1");

        let _ = std::fs::remove_dir_all(&ml_backends);
    }

    /// The committed `multi` fixture is a genuine two-input/two-output
    /// TorchScript model `(sum, prod) = (a + b, a * b)`. Driving it with two
    /// named inputs must return both named outputs with correct arithmetic —
    /// proving the additive multi-tensor path (named binding, multi-input
    /// forward, tuple-output serialization) end to end.
    #[tokio::test]
    async fn multi_tensor_model_infer() {
        assert!(
            fixtures_dir().join("multi").join("mmodel.pt").is_file(),
            "multi fixture missing"
        );
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [10.0f32, 20.0, 30.0, 40.0];

        let input = |name: &str| InferInputTensor {
            name: name.to_string(),
            datatype: FP32.to_string(),
            shape: vec![1, 4],
            parameters: Default::default(),
            contents: None,
        };

        let addr = spawn_triton_server("multi").await;
        let mut client = connect(addr).await;
        let response = client
            .model_infer(ModelInferRequest {
                model_name: "multi".to_string(),
                model_version: String::new(),
                id: "m-1".to_string(),
                parameters: Default::default(),
                // Deliberately out of contract order (b before a) to prove
                // name-based binding, not positional.
                inputs: vec![input("b"), input("a")],
                outputs: Vec::new(),
                raw_input_contents: vec![f32_le_bytes(&b), f32_le_bytes(&a)],
            })
            .await
            .expect("multi-tensor model_infer should succeed")
            .into_inner();

        assert_eq!(response.outputs.len(), 2, "two named outputs");
        let decode = |t: &InferOutputTensor, raw: &[u8]| -> (String, Vec<f32>) {
            (
                t.name.clone(),
                raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )
        };
        let outs: std::collections::HashMap<String, Vec<f32>> = response
            .outputs
            .iter()
            .zip(response.raw_output_contents.iter())
            .map(|(t, raw)| decode(t, raw))
            .collect();

        assert_eq!(outs["sum"], vec![11.0, 22.0, 33.0, 44.0], "a + b");
        assert_eq!(outs["prod"], vec![10.0, 40.0, 90.0, 160.0], "a * b");
    }

    /// A multi-tensor request missing one of the declared inputs is rejected.
    #[tokio::test]
    async fn multi_tensor_missing_input_rejected() {
        let addr = spawn_triton_server("multi").await;
        let mut client = connect(addr).await;
        let status = client
            .model_infer(ModelInferRequest {
                model_name: "multi".to_string(),
                model_version: String::new(),
                id: String::new(),
                parameters: Default::default(),
                // Only "a"; "b" is missing.
                inputs: vec![InferInputTensor {
                    name: "a".to_string(),
                    datatype: FP32.to_string(),
                    shape: vec![1, 4],
                    parameters: Default::default(),
                    contents: None,
                }],
                outputs: Vec::new(),
                raw_input_contents: vec![f32_le_bytes(&[1.0, 2.0, 3.0, 4.0])],
            })
            .await
            .expect_err("missing input must be rejected");
        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status:?}");
    }

    /// Multi-tensor metadata advertises every named input and output.
    #[tokio::test]
    async fn multi_tensor_metadata_lists_all_tensors() {
        let addr = spawn_triton_server("multi").await;
        let mut client = connect(addr).await;
        let meta = client
            .model_metadata(ModelMetadataRequest {
                name: "multi".to_string(),
                version: String::new(),
            })
            .await
            .expect("metadata")
            .into_inner();
        let names: Vec<&str> = meta.inputs.iter().map(|i| i.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b"], "input names");
        let outs: Vec<&str> = meta.outputs.iter().map(|o| o.name.as_str()).collect();
        assert_eq!(outs, vec!["sum", "prod"], "output names");
    }

    /// An unknown model is rejected with NotFound on the infer path.
    #[tokio::test]
    async fn model_infer_unknown_model_not_found() {
        assert!(fixtures_dir().join("model3").is_dir(), "model3 missing");
        let addr = spawn_triton_server("model3").await;
        let mut client = connect(addr).await;
        let status = client
            .model_infer(ModelInferRequest {
                model_name: "ghost".to_string(),
                model_version: String::new(),
                id: String::new(),
                parameters: Default::default(),
                inputs: vec![InferInputTensor {
                    name: "input".to_string(),
                    datatype: FP32.to_string(),
                    shape: vec![1, 16],
                    parameters: Default::default(),
                    contents: None,
                }],
                outputs: Vec::new(),
                raw_input_contents: vec![f32_le_bytes(&[0.0; 16])],
            })
            .await
            .expect_err("unknown model must be rejected");
        assert_eq!(status.code(), tonic::Code::NotFound, "{status:?}");
    }

    fn i32_le_bytes(values: &[i32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn i64_le_bytes(values: &[i64]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// The committed `rustint` fixture is a genuine non-float TorchScript model
    /// (`output = input + 1`, int64). Driving it over ModelInfer with INT64 raw
    /// bytes must return correct int64 arithmetic — proving the Rust `.pt` path
    /// preserves a non-float dtype end to end (input build, forward pass, output
    /// serialization), not just FP32.
    #[tokio::test]
    async fn rust_int64_model_infer() {
        assert!(
            fixtures_dir().join("rustint").join("rmodel.pt").is_file(),
            "rustint fixture missing"
        );
        let input = [10i64, 20, 30, 40];
        let expected = [11i64, 21, 31, 41];

        let addr = spawn_triton_server("rustint").await;
        let mut client = connect(addr).await;
        let response = client
            .model_infer(ModelInferRequest {
                model_name: "rustint".to_string(),
                model_version: String::new(),
                id: "ri-1".to_string(),
                parameters: Default::default(),
                inputs: vec![InferInputTensor {
                    name: "input".to_string(),
                    datatype: "INT64".to_string(),
                    shape: vec![1, 4],
                    parameters: Default::default(),
                    contents: None,
                }],
                outputs: Vec::new(),
                raw_input_contents: vec![i64_le_bytes(&input)],
            })
            .await
            .expect("int64 rust model_infer should succeed")
            .into_inner();

        assert_eq!(response.outputs[0].datatype, "INT64", "output datatype");
        assert_eq!(response.outputs[0].shape, vec![1, 4]);
        let got: Vec<i64> = response.raw_output_contents[0]
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got, expected, "rustint must compute input + 1 in int64");
    }

    // ---------------------------------------------------------------------
    // Datatypes (Python byte-passthrough path)
    // ---------------------------------------------------------------------

    /// The committed `pyaddint` fixture declares `data_type: INT32` and computes
    /// `input + 1`. A stock client sending INT32 raw bytes must get INT32 output
    /// back with the correct integer arithmetic — proving the non-float dtype
    /// travels end to end, not just that some tensor returned.
    #[tokio::test]
    async fn python_model_infer_int32_datatype() {
        assert!(
            fixtures_dir().join("pyaddint").join("main.py").is_file(),
            "pyaddint fixture missing"
        );
        let input = [10i32, 20, 30, 40];
        let expected = [11i32, 21, 31, 41];

        let addr = spawn_triton_server("pyaddint").await;
        let mut client = connect(addr).await;
        let response = client
            .model_infer(ModelInferRequest {
                model_name: "pyaddint".to_string(),
                model_version: String::new(),
                id: "int-1".to_string(),
                parameters: Default::default(),
                inputs: vec![InferInputTensor {
                    name: "input".to_string(),
                    datatype: "INT32".to_string(),
                    shape: vec![1, 4],
                    parameters: Default::default(),
                    contents: None,
                }],
                outputs: Vec::new(),
                raw_input_contents: vec![i32_le_bytes(&input)],
            })
            .await
            .expect("int32 model_infer should succeed")
            .into_inner();

        assert_eq!(response.outputs[0].datatype, "INT32", "output datatype");
        assert_eq!(response.outputs[0].shape, vec![1, 4]);
        let got: Vec<i32> = response.raw_output_contents[0]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(got, expected, "pyaddint must compute input + 1 in int32");
    }

    /// An output-only Python model (declares `output_shape`, no `input_shape`)
    /// is servable over ModelInfer with zero inputs — matching what
    /// `model_metadata` advertises. Reads no stdin, writes a [2] tensor.
    #[tokio::test]
    async fn python_output_only_model_infer() {
        let main_py = "\
import os, struct
with open(os.environ['NEREID_OUTPUT_PATH'], 'wb') as f:
    f.write(b'float32 2\\n')
    f.write(struct.pack('<2f', 7.0, 8.0))
";
        let (ml_backends, name) =
            make_temp_python_model("output-only", main_py, "output_shape: [2]\n");
        let addr = spawn_triton_server_at(ml_backends.clone(), &name).await;
        let mut client = connect(addr).await;

        // Metadata advertises no inputs, one output.
        let meta = client
            .model_metadata(ModelMetadataRequest {
                name: name.clone(),
                version: String::new(),
            })
            .await
            .expect("metadata")
            .into_inner();
        assert!(meta.inputs.is_empty(), "output-only model has no inputs");
        assert_eq!(meta.outputs.len(), 1);

        // Infer with zero inputs returns the model's tensor.
        let response = client
            .model_infer(ModelInferRequest {
                model_name: name.clone(),
                model_version: String::new(),
                id: String::new(),
                parameters: Default::default(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                raw_input_contents: Vec::new(),
            })
            .await
            .expect("output-only model_infer should succeed")
            .into_inner();
        assert_eq!(response.outputs[0].shape, vec![2]);
        let got: Vec<f32> = response.raw_output_contents[0]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(got, vec![7.0, 8.0]);

        // Sending an input to an output-only model is rejected.
        let status = client
            .model_infer(infer_request(&name, vec![1, 2], f32_le_bytes(&[1.0, 2.0])))
            .await
            .expect_err("unexpected input must be rejected");
        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status:?}");

        // Stray raw_input_contents (with zero declared inputs) is also rejected.
        let status = client
            .model_infer(ModelInferRequest {
                model_name: name.clone(),
                model_version: String::new(),
                id: String::new(),
                parameters: Default::default(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                raw_input_contents: vec![f32_le_bytes(&[1.0, 2.0])],
            })
            .await
            .expect_err("stray raw buffer must be rejected");
        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status:?}");

        let _ = std::fs::remove_dir_all(&ml_backends);
    }

    /// Duplicate input tensor names in a multi-tensor request are rejected
    /// (rather than silently binding the wrong positional raw buffer).
    #[tokio::test]
    async fn multi_tensor_rejects_duplicate_input_names() {
        let addr = spawn_triton_server("multi").await;
        let mut client = connect(addr).await;
        let dup = InferInputTensor {
            name: "a".to_string(),
            datatype: FP32.to_string(),
            shape: vec![1, 4],
            parameters: Default::default(),
            contents: None,
        };
        let status = client
            .model_infer(ModelInferRequest {
                model_name: "multi".to_string(),
                model_version: String::new(),
                id: String::new(),
                parameters: Default::default(),
                inputs: vec![dup.clone(), dup],
                outputs: Vec::new(),
                raw_input_contents: vec![
                    f32_le_bytes(&[1.0, 2.0, 3.0, 4.0]),
                    f32_le_bytes(&[5.0, 6.0, 7.0, 8.0]),
                ],
            })
            .await
            .expect_err("duplicate input name must be rejected");
        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status:?}");
    }

    /// A multi-tensor request naming an input not declared by the model is
    /// rejected rather than silently dropped.
    #[tokio::test]
    async fn multi_tensor_rejects_unknown_input_name() {
        let addr = spawn_triton_server("multi").await;
        let mut client = connect(addr).await;
        let mk = |name: &str| InferInputTensor {
            name: name.to_string(),
            datatype: FP32.to_string(),
            shape: vec![1, 4],
            parameters: Default::default(),
            contents: None,
        };
        // Declared inputs are "a" and "b"; add a stray "c".
        let status = client
            .model_infer(ModelInferRequest {
                model_name: "multi".to_string(),
                model_version: String::new(),
                id: String::new(),
                parameters: Default::default(),
                inputs: vec![mk("a"), mk("b"), mk("c")],
                outputs: Vec::new(),
                raw_input_contents: vec![
                    f32_le_bytes(&[1.0, 2.0, 3.0, 4.0]),
                    f32_le_bytes(&[5.0, 6.0, 7.0, 8.0]),
                    f32_le_bytes(&[9.0, 9.0, 9.0, 9.0]),
                ],
            })
            .await
            .expect_err("unknown input name must be rejected");
        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status:?}");
    }

    /// A multi-tensor request asking for an output not declared by the model is
    /// rejected rather than silently omitted.
    #[tokio::test]
    async fn multi_tensor_rejects_unknown_output_name() {
        use crate::proto::model_infer_request::InferRequestedOutputTensor;
        let addr = spawn_triton_server("multi").await;
        let mut client = connect(addr).await;
        let mk = |name: &str| InferInputTensor {
            name: name.to_string(),
            datatype: FP32.to_string(),
            shape: vec![1, 4],
            parameters: Default::default(),
            contents: None,
        };
        let status = client
            .model_infer(ModelInferRequest {
                model_name: "multi".to_string(),
                model_version: String::new(),
                id: String::new(),
                parameters: Default::default(),
                inputs: vec![mk("a"), mk("b")],
                // Declared outputs are "sum"/"prod"; request an unknown one.
                outputs: vec![InferRequestedOutputTensor {
                    name: "not_an_output".to_string(),
                    parameters: Default::default(),
                }],
                raw_input_contents: vec![
                    f32_le_bytes(&[1.0, 2.0, 3.0, 4.0]),
                    f32_le_bytes(&[5.0, 6.0, 7.0, 8.0]),
                ],
            })
            .await
            .expect_err("unknown output name must be rejected");
        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status:?}");
    }

    /// A multi-tensor model whose returned output dtype contradicts its declared
    /// contract is rejected rather than serving a mislabeled tensor. Uses the
    /// real `multi` `.pt` (returns FP32) with a textproto that declares `sum` as
    /// INT32.
    #[tokio::test]
    async fn multi_tensor_rejects_output_dtype_mismatch() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let ml_backends = std::env::temp_dir().join(format!("nereid-multi-dtype-{nanos}"));
        let dir = ml_backends.join("m");
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::copy(
            fixtures_dir().join("multi").join("mmodel.pt"),
            dir.join("mmodel.pt"),
        )
        .expect("copy .pt");
        std::fs::write(
            dir.join("model_inference.textproto"),
            "max_batch_size: 4\n\
             input {\n  name: \"a\"\n  data_type: \"FP32\"\n  dims: [4]\n}\n\
             input {\n  name: \"b\"\n  data_type: \"FP32\"\n  dims: [4]\n}\n\
             output {\n  name: \"sum\"\n  data_type: \"INT32\"\n  dims: [4]\n}\n\
             output {\n  name: \"prod\"\n  data_type: \"FP32\"\n  dims: [4]\n}\n",
        )
        .expect("write textproto");

        let addr = spawn_triton_server_at(ml_backends.clone(), "m").await;
        let mut client = connect(addr).await;
        let mk = |name: &str| InferInputTensor {
            name: name.to_string(),
            datatype: FP32.to_string(),
            shape: vec![1, 4],
            parameters: Default::default(),
            contents: None,
        };
        let status = client
            .model_infer(ModelInferRequest {
                model_name: "m".to_string(),
                model_version: String::new(),
                id: String::new(),
                parameters: Default::default(),
                inputs: vec![mk("a"), mk("b")],
                outputs: Vec::new(),
                raw_input_contents: vec![
                    f32_le_bytes(&[1.0, 2.0, 3.0, 4.0]),
                    f32_le_bytes(&[5.0, 6.0, 7.0, 8.0]),
                ],
            })
            .await
            .expect_err("output dtype mismatch must be rejected");
        assert_eq!(status.code(), tonic::Code::Internal, "{status:?}");

        let _ = std::fs::remove_dir_all(&ml_backends);
    }

    /// Metadata reflects a non-float declared datatype.
    #[tokio::test]
    async fn model_metadata_reports_declared_datatype() {
        let addr = spawn_triton_server("pyaddint").await;
        let mut client = connect(addr).await;
        let meta = client
            .model_metadata(ModelMetadataRequest {
                name: "pyaddint".to_string(),
                version: String::new(),
            })
            .await
            .expect("metadata")
            .into_inner();
        assert_eq!(meta.inputs[0].datatype, "INT32");
        assert_eq!(meta.outputs[0].datatype, "INT32");
    }

    /// A request whose datatype disagrees with the model's declared datatype is
    /// rejected before any inference runs.
    #[tokio::test]
    async fn model_infer_rejects_datatype_mismatch() {
        let addr = spawn_triton_server("pyaddint").await;
        let mut client = connect(addr).await;
        let status = client
            .model_infer(ModelInferRequest {
                model_name: "pyaddint".to_string(),
                model_version: String::new(),
                id: String::new(),
                parameters: Default::default(),
                // pyaddint expects INT32; send FP32.
                inputs: vec![InferInputTensor {
                    name: "input".to_string(),
                    datatype: FP32.to_string(),
                    shape: vec![1, 4],
                    parameters: Default::default(),
                    contents: None,
                }],
                outputs: Vec::new(),
                raw_input_contents: vec![f32_le_bytes(&[1.0, 2.0, 3.0, 4.0])],
            })
            .await
            .expect_err("datatype mismatch must be rejected");
        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status:?}");
    }

    /// A raw buffer whose length doesn't match shape × element size is rejected.
    #[tokio::test]
    async fn model_infer_rejects_byte_length_mismatch() {
        let addr = spawn_triton_server("model3").await;
        let mut client = connect(addr).await;
        let status = client
            .model_infer(ModelInferRequest {
                model_name: "model3".to_string(),
                model_version: String::new(),
                id: String::new(),
                parameters: Default::default(),
                // shape says 16 floats (64 bytes) but only 8 floats are sent.
                inputs: vec![InferInputTensor {
                    name: "input".to_string(),
                    datatype: FP32.to_string(),
                    shape: vec![1, 16],
                    parameters: Default::default(),
                    contents: None,
                }],
                outputs: Vec::new(),
                raw_input_contents: vec![f32_le_bytes(&[0.0; 8])],
            })
            .await
            .expect_err("short buffer must be rejected");
        assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status:?}");
    }

    // ---------------------------------------------------------------------
    // Versioning
    // ---------------------------------------------------------------------

    /// The implicit version "1" and an empty selector are servable; any other
    /// version is NotFound on infer/metadata and not-ready on ModelReady.
    #[tokio::test]
    async fn model_versioning_only_version_one() {
        let addr = spawn_triton_server("model3").await;
        let mut client = connect(addr).await;

        // ModelReady: "1" and "" ready; "2" not.
        for (ver, want) in [("", true), ("1", true), ("2", false)] {
            let ready = client
                .model_ready(ModelReadyRequest {
                    name: "model3".to_string(),
                    version: ver.to_string(),
                })
                .await
                .expect("model_ready")
                .into_inner()
                .ready;
            assert_eq!(ready, want, "version {ver:?} readiness");
        }

        // Metadata for a bad version is NotFound.
        let status = client
            .model_metadata(ModelMetadataRequest {
                name: "model3".to_string(),
                version: "2".to_string(),
            })
            .await
            .expect_err("bad version metadata");
        assert_eq!(status.code(), tonic::Code::NotFound, "{status:?}");

        // Infer for a bad version is NotFound.
        let status = client
            .model_infer(ModelInferRequest {
                model_name: "model3".to_string(),
                model_version: "2".to_string(),
                id: String::new(),
                parameters: Default::default(),
                inputs: vec![InferInputTensor {
                    name: "input".to_string(),
                    datatype: FP32.to_string(),
                    shape: vec![1, 16],
                    parameters: Default::default(),
                    contents: None,
                }],
                outputs: Vec::new(),
                raw_input_contents: vec![f32_le_bytes(
                    &(0..16).map(|v| v as f32).collect::<Vec<_>>(),
                )],
            })
            .await
            .expect_err("bad version infer");
        assert_eq!(status.code(), tonic::Code::NotFound, "{status:?}");
    }

    // ---------------------------------------------------------------------
    // Streaming (ModelStreamInfer)
    // ---------------------------------------------------------------------

    /// A stock streaming client sends several requests and receives one
    /// `ModelStreamInferResponse` per request, each carrying the correct
    /// `pymul` output. Proves the streaming RPC (verifiable only against a real
    /// client, since it shares nereid's own stubs otherwise).
    #[tokio::test]
    async fn model_stream_infer_streams_results() {
        let addr = spawn_triton_server("pymul").await;
        let mut client = connect(addr).await;

        let batches = [
            [1.0f32, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let requests: Vec<ModelInferRequest> = batches
            .iter()
            .enumerate()
            .map(|(i, vals)| ModelInferRequest {
                model_name: "pymul".to_string(),
                model_version: String::new(),
                id: format!("s{i}"),
                parameters: Default::default(),
                inputs: vec![InferInputTensor {
                    name: "input".to_string(),
                    datatype: FP32.to_string(),
                    shape: vec![1, 4],
                    parameters: Default::default(),
                    contents: None,
                }],
                outputs: Vec::new(),
                raw_input_contents: vec![f32_le_bytes(vals)],
            })
            .collect();

        let mut stream = client
            .model_stream_infer(tokio_stream::iter(requests))
            .await
            .expect("stream_infer should start")
            .into_inner();

        let mut seen = 0;
        while let Some(resp) = stream.message().await.expect("stream message") {
            assert!(
                resp.error_message.is_empty(),
                "unexpected error: {}",
                resp.error_message
            );
            let infer = resp.infer_response.expect("infer_response present");
            let expected: Vec<f32> = batches[seen].iter().map(|v| v * 2.0 + 1.0).collect();
            let got: Vec<f32> = infer.raw_output_contents[0]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(got, expected, "stream response {seen} arithmetic");
            seen += 1;
        }
        assert_eq!(seen, batches.len(), "one response per request");
    }

    /// A per-request failure in a stream is reported inline via `error_message`
    /// and the stream continues delivering subsequent good responses — it must
    /// not terminate the whole stream.
    #[tokio::test]
    async fn model_stream_infer_reports_per_request_error_and_continues() {
        let addr = spawn_triton_server("pymul").await;
        let mut client = connect(addr).await;

        let good = |id: &str| ModelInferRequest {
            model_name: "pymul".to_string(),
            model_version: String::new(),
            id: id.to_string(),
            parameters: Default::default(),
            inputs: vec![InferInputTensor {
                name: "input".to_string(),
                datatype: FP32.to_string(),
                shape: vec![1, 4],
                parameters: Default::default(),
                contents: None,
            }],
            outputs: Vec::new(),
            raw_input_contents: vec![f32_le_bytes(&[1.0, 2.0, 3.0, 4.0])],
        };
        // Middle request has a bad shape (wrong trailing dim).
        let mut bad = good("bad");
        bad.inputs[0].shape = vec![1, 5];
        bad.raw_input_contents = vec![f32_le_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0])];

        let requests = vec![good("a"), bad, good("b")];
        let mut stream = client
            .model_stream_infer(tokio_stream::iter(requests))
            .await
            .expect("stream start")
            .into_inner();

        let mut oks = 0;
        let mut errs = 0;
        while let Some(resp) = stream.message().await.expect("stream message") {
            if resp.error_message.is_empty() {
                assert!(resp.infer_response.is_some());
                oks += 1;
            } else {
                assert!(resp.infer_response.is_none());
                errs += 1;
            }
        }
        assert_eq!(oks, 2, "two good requests should succeed");
        assert_eq!(errs, 1, "the bad request should surface one inline error");
    }

    // ---------------------------------------------------------------------
    // Backpressure
    // ---------------------------------------------------------------------

    /// With `queue_capacity: 1`, two concurrent inferences against a slow Python
    /// model must not both run: exactly one holds the single permit and the
    /// other is rejected with `ResourceExhausted`.
    #[tokio::test]
    async fn python_backpressure_rejects_when_full() {
        // A model that sleeps ~1s so the two requests genuinely overlap.
        let main_py = "\
import os, sys, time
sys.stdin.buffer.read()
time.sleep(1.0)
shape = os.environ['NEREID_INPUT_SHAPE']
with open(os.environ['NEREID_OUTPUT_PATH'], 'wb') as f:
    f.write(('float32 ' + shape + '\\n').encode('utf-8'))
    import struct
    f.write(struct.pack('<4f', 0.0, 0.0, 0.0, 0.0))
";
        let (ml_backends, name) = make_temp_python_model(
            "backpressure",
            main_py,
            "input_shape: [4]\nmax_batch_size: 4\noutput_shape: [4]\n",
        );

        let addr = spawn_triton_server_qc(ml_backends.clone(), &name, 1).await;
        let payload = f32_le_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let fut_a = async {
            let mut client = connect(addr).await;
            client
                .model_infer(infer_request(&name, vec![1, 4], payload.clone()))
                .await
        };
        let fut_b = async {
            let mut client = connect(addr).await;
            client
                .model_infer(infer_request(&name, vec![1, 4], payload.clone()))
                .await
        };
        let (a, b) = tokio::join!(fut_a, fut_b);
        let codes = [&a, &b].map(|r| r.as_ref().err().map(|s| s.code()));
        let exhausted = codes
            .iter()
            .filter(|c| **c == Some(tonic::Code::ResourceExhausted))
            .count();
        let ok = [&a, &b].iter().filter(|r| r.is_ok()).count();
        assert_eq!(ok, 1, "exactly one request should succeed: {a:?} {b:?}");
        assert_eq!(
            exhausted, 1,
            "the other should be ResourceExhausted: {a:?} {b:?}"
        );

        let _ = std::fs::remove_dir_all(&ml_backends);
    }
}
