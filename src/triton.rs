//! Triton-compatible KServe v2 inference surface (`GRPCInferenceService`).
//!
//! Implements the client-facing subset of NVIDIA Triton's gRPC protocol so that
//! a stock `tritonclient` (or any KServe v2 speaker) can drive nereid without
//! code changes. The wire format — package `inference`, service
//! `GRPCInferenceService`, message field numbers — is vendored verbatim from
//! upstream in `proto/grpc_service.proto`.
//!
//! Implemented: `ServerLive/Ready`, `ModelReady`, `ServerMetadata`,
//! `ModelMetadata`, unary `ModelInfer`, and streaming `ModelStreamInfer`. Both
//! backends are servable — Rust `.pt` (single- and multi-tensor) and Python
//! `main.py` (every reply is a typed tensor, via the same `NEREID_OUTPUT_PATH`
//! contract the native `Nereid/Checkpoint` path uses). The request datatype must
//! match the model's declared `data_type` (default `FP32`); the Python path is
//! byte-passthrough over any fixed-width dtype, the Rust path covers the libtorch
//! kinds. nereid serves a single implicit model version, `"1"`.
//!
//! Not implemented (deferred): `UINT16/32/64` and `BYTES` on the Rust path, the
//! HTTP/REST `/v2` mirror, Prometheus metrics, and the repository/config RPCs.

use std::sync::Arc;

use tokio::sync::mpsc;
use tonic::codegen::tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use crate::dtype;
#[cfg(feature = "torch")]
use crate::model_runtime::tensor_from_input_bytes;
use crate::model_runtime::{InputShapeContract, ModelManager};
use crate::native_backend::NativeTensor;
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
#[cfg(feature = "python")]
use crate::python_backend::{PythonInput, run_python_inference};

/// KServe spells 32-bit float exactly this way; it's also the default datatype
/// for a model whose contract omits `data_type`.
const FP32: &str = "FP32";

/// nereid serves a single, implicit model version. A request may name it
/// explicitly (`"1"`) or leave it empty; any other version is unavailable.
const MODEL_VERSION: &str = "1";

/// The extracted single input: `(raw little-endian bytes, batch-normalized
/// shape, canonical dtype, expected batch size)`.
type SingleInput = (Vec<u8>, Vec<i64>, &'static str, Option<i64>);

/// Validate and extract the single input tensor for `model_name` from `request`,
/// returning it, or `None` for an output-only model. Backend-agnostic: it
/// validates datatype/shape/byte-length via the KServe contract and never
/// touches libtorch, so every single-tensor backend shares one implementation.
fn extract_single_input(
    contract: &InputShapeContract,
    request: &mut ModelInferRequest,
    model_name: &str,
) -> Result<Option<SingleInput>, Status> {
    if !contract.has_input() {
        // Output-only model: no input tensor expected. Reject stray client data.
        if !request.inputs.is_empty() || !request.raw_input_contents.is_empty() {
            return Err(Status::invalid_argument(format!(
                "model '{model_name}' declares no input tensor but the request carries \
                 {} input(s) and {} raw buffer(s)",
                request.inputs.len(),
                request.raw_input_contents.len()
            )));
        }
        return Ok(None);
    }

    if request.inputs.len() != 1 {
        return Err(Status::invalid_argument(format!(
            "model '{model_name}' expects one input tensor, got {}",
            request.inputs.len()
        )));
    }
    let input = &request.inputs[0];

    let expected_dt = contract.input_datatype();
    if input.datatype != expected_dt {
        return Err(Status::invalid_argument(format!(
            "datatype mismatch for model '{model_name}': expected {expected_dt}, got '{}'",
            input.datatype
        )));
    }
    let (elem_size, canonical_dtype) = dtype::kserve_fixed_width(expected_dt).ok_or_else(|| {
        Status::invalid_argument(format!(
            "unsupported datatype '{expected_dt}' for model '{model_name}'"
        ))
    })?;

    // Raw bytes take precedence over typed `contents`; typed contents is FP32-only.
    let input_shape = input.shape.clone();
    let input_bytes = if !request.raw_input_contents.is_empty() {
        if request.raw_input_contents.len() != 1 {
            return Err(Status::invalid_argument(
                "raw_input_contents must hold exactly one buffer for a single input",
            ));
        }
        request.raw_input_contents.remove(0)
    } else if expected_dt == FP32 {
        match &input.contents {
            Some(contents) => contents
                .fp32_contents
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            None => {
                return Err(Status::invalid_argument(
                    "input tensor has neither raw_input_contents nor contents",
                ));
            }
        }
    } else {
        return Err(Status::invalid_argument(format!(
            "datatype '{expected_dt}' requires raw_input_contents (typed contents is FP32-only)"
        )));
    };

    contract.validate_request_shape(&input_shape, model_name)?;
    let request_shape = contract.normalize_request_shape(input_shape);

    let numel = request_shape
        .iter()
        .try_fold(1i64, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| Status::invalid_argument("input tensor shape overflow"))?;
    let expected_bytes = (numel as usize).saturating_mul(elem_size);
    if input_bytes.len() != expected_bytes {
        return Err(Status::invalid_argument(format!(
            "input byte length {} does not match shape {request_shape:?} \u{d7} \
             {elem_size} bytes ({expected_bytes}) for model '{model_name}'",
            input_bytes.len()
        )));
    }

    let expected_batch = if contract.max_batch_size() > 0 {
        request_shape.first().copied()
    } else {
        None
    };
    Ok(Some((
        input_bytes,
        request_shape,
        canonical_dtype,
        expected_batch,
    )))
}

/// Whether a requested model version is servable. nereid has no version
/// concept, so only the implicit version `"1"` (or an empty selector) exists.
fn version_available(model_version: &str) -> bool {
    model_version.is_empty() || model_version == MODEL_VERSION
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

        // Multi-tensor models advertise every named input/output from their
        // contract.
        if let Some(multi) = self.model_manager.multi_contract(&name) {
            let to_meta = |spec: &crate::model_runtime::TensorSpec| {
                let mut shape = Vec::new();
                if multi.max_batch_size > 0 {
                    shape.push(-1);
                }
                shape.extend_from_slice(&spec.dims);
                TensorMetadata {
                    name: spec.name.clone(),
                    datatype: spec.dtype.clone(),
                    shape,
                }
            };
            let platform = self.model_manager.platform(&name).to_string();
            return Ok(Response::new(ModelMetadataResponse {
                name,
                versions: vec![MODEL_VERSION.to_string()],
                platform,
                inputs: multi.inputs.iter().map(to_meta).collect(),
                outputs: multi.outputs.iter().map(to_meta).collect(),
            }));
        }

        let contract = self.model_manager.input_contract(&name);
        // Input/output datatypes come from the model's declared contract
        // (default FP32).
        let datatype = contract.map_or(FP32, |c| c.input_datatype()).to_string();

        // Reconstruct the advertised input shape from the model's contract. A
        // declared batch dimension is surfaced as a leading -1 (variable), per
        // Triton convention; -1 input dims are passed through unchanged. A model
        // that declares no input tensor (output-only Python) advertises none.
        let inputs = match contract.and_then(|c| c.input_shape().map(|s| (c, s))) {
            Some((contract, input_shape)) => {
                let mut shape = Vec::new();
                if contract.max_batch_size() > 0 {
                    shape.push(-1);
                }
                shape.extend_from_slice(input_shape);
                vec![TensorMetadata {
                    name: "input".to_string(),
                    datatype: datatype.clone(),
                    shape,
                }]
            }
            None => Vec::new(),
        };

        // Advertise the declared output shape when the model provides one
        // (Python tensor models via `output_shape`); otherwise fall back to a
        // single variable dimension (`[-1]`) since Rust models may not declare an
        // output shape today. A declared batch dimension is surfaced as leading -1.
        let output_shape = contract
            .and_then(|contract| {
                contract.output_shape().map(|declared| {
                    let mut shape = Vec::new();
                    if contract.max_batch_size() > 0 {
                        shape.push(-1);
                    }
                    shape.extend_from_slice(declared);
                    shape
                })
            })
            .unwrap_or_else(|| vec![-1]);
        let outputs = vec![TensorMetadata {
            name: "output".to_string(),
            datatype,
            shape: output_shape,
        }];

        let platform = self.model_manager.platform(&name).to_string();
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
    /// selection, datatype validation, shape/batch normalization, backend
    /// dispatch (Rust `.pt` queue or Python subprocess with backpressure), and
    /// output validation.
    async fn infer_once(&self, request: ModelInferRequest) -> Result<ModelInferResponse, Status> {
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
        if !self.model_manager.is_configured(&model_name) {
            return Err(Status::not_found(format!(
                "model '{model_name}' is not configured in nereid.yaml"
            )));
        }

        // Multi-tensor models take the additive named-tensor path.
        if self.model_manager.is_multi(&model_name) {
            return self.infer_multi(request, model_name).await;
        }
        // Single-tensor dispatch, one method per backend so each compiles only
        // when its feature is enabled.
        if self.model_manager.is_native(&model_name) {
            return self.infer_native_once(request, model_name).await;
        }
        #[cfg(feature = "python")]
        if self.model_manager.is_python(&model_name) {
            return self.infer_python_once(request, model_name).await;
        }
        #[cfg(feature = "torch")]
        {
            self.infer_torch_once(request, model_name).await
        }
        #[cfg(not(feature = "torch"))]
        {
            let _ = &request;
            Err(Status::internal(format!(
                "model '{model_name}' is a non-native single-tensor model, but neither the \
                 torch nor python backend is compiled in"
            )))
        }
    }

    /// Build the single-tensor Triton response for a model whose only output is
    /// named `output`. Rejects a client-requested output name that isn't `output`.
    fn single_output_response(
        request: &ModelInferRequest,
        model_name: String,
        output_shape: Vec<i64>,
        output_bytes: Vec<u8>,
        output_dt: String,
    ) -> Result<ModelInferResponse, Status> {
        const OUTPUT_NAME: &str = "output";
        if let Some(unknown) = request.outputs.iter().find(|o| o.name != OUTPUT_NAME) {
            return Err(Status::invalid_argument(format!(
                "model '{model_name}' has no output '{}'; its only output is '{OUTPUT_NAME}'",
                unknown.name
            )));
        }
        Ok(ModelInferResponse {
            model_name,
            model_version: MODEL_VERSION.to_string(),
            id: request.id.clone(),
            parameters: Default::default(),
            outputs: vec![InferOutputTensor {
                name: OUTPUT_NAME.to_string(),
                datatype: output_dt,
                shape: output_shape,
                parameters: Default::default(),
                contents: None,
            }],
            raw_output_contents: vec![output_bytes],
        })
    }

    /// Native (ONNX/TensorFlow) single-tensor inference over ModelInfer.
    async fn infer_native_once(
        &self,
        mut request: ModelInferRequest,
        model_name: String,
    ) -> Result<ModelInferResponse, Status> {
        let contract = self
            .model_manager
            .input_contract(&model_name)
            .cloned()
            .ok_or_else(|| {
                Status::internal(format!(
                    "model '{model_name}' is missing its input contract"
                ))
            })?;
        let (bytes, shape, canonical, _batch) =
            extract_single_input(&contract, &mut request, &model_name)?.ok_or_else(|| {
                Status::invalid_argument(format!(
                    "native model '{model_name}' requires an input tensor"
                ))
            })?;
        let native = NativeTensor {
            name: "input".to_string(),
            shape,
            dtype: canonical.to_string(),
            data: bytes,
        };
        let response_rx = self
            .model_manager
            .enqueue_native(&model_name, vec![native])?;
        let mut outputs = response_rx.await.map_err(|_| {
            Status::internal(format!(
                "worker response channel closed for model '{model_name}'"
            ))
        })??;
        if outputs.len() != 1 {
            return Err(Status::internal(format!(
                "single-tensor native model '{model_name}' returned {} outputs, expected 1",
                outputs.len()
            )));
        }
        let (out_shape, out_bytes, out_canonical) = outputs.remove(0);
        let (out_kserve, _) = dtype::canonical_to_kserve(&out_canonical).ok_or_else(|| {
            Status::internal(format!(
                "model '{model_name}' returned unsupported output dtype '{out_canonical}'"
            ))
        })?;
        Self::single_output_response(
            &request,
            model_name,
            out_shape,
            out_bytes,
            out_kserve.to_string(),
        )
    }

    /// Rust TorchScript (`.pt`) single-tensor inference over ModelInfer.
    #[cfg(feature = "torch")]
    async fn infer_torch_once(
        &self,
        mut request: ModelInferRequest,
        model_name: String,
    ) -> Result<ModelInferResponse, Status> {
        let contract = self
            .model_manager
            .input_contract(&model_name)
            .cloned()
            .ok_or_else(|| {
                Status::internal(format!(
                    "model '{model_name}' is missing its input contract"
                ))
            })?;
        let expected_dt = contract.input_datatype().to_string();
        let (bytes, shape, canonical, _batch) =
            extract_single_input(&contract, &mut request, &model_name)?.ok_or_else(|| {
                Status::invalid_argument(format!(
                    "Rust .pt model '{model_name}' requires an input tensor"
                ))
            })?;
        // dtypes with no libtorch kind (uint16/32/64) are rejected here.
        let kind = dtype::kind_from_canonical(canonical).ok_or_else(|| {
            Status::invalid_argument(format!(
                "Rust .pt model '{model_name}' does not support datatype {expected_dt} \
                 (no libtorch kind)"
            ))
        })?;
        let tensor = tensor_from_input_bytes(&bytes, &shape, &model_name, kind)?;
        let response_rx = self.model_manager.enqueue(&model_name, tensor)?;
        let (out_shape, out_bytes, out_canonical) = response_rx.await.map_err(|_| {
            Status::internal(format!(
                "worker response channel closed for model '{model_name}'"
            ))
        })??;
        let (out_kserve, _) = dtype::canonical_to_kserve(&out_canonical).ok_or_else(|| {
            Status::internal(format!(
                "model '{model_name}' returned unsupported output dtype '{out_canonical}'"
            ))
        })?;
        Self::single_output_response(
            &request,
            model_name,
            out_shape,
            out_bytes,
            out_kserve.to_string(),
        )
    }

    /// Python (`main.py`) single-tensor inference over ModelInfer.
    #[cfg(feature = "python")]
    async fn infer_python_once(
        &self,
        mut request: ModelInferRequest,
        model_name: String,
    ) -> Result<ModelInferResponse, Status> {
        let contract = self
            .model_manager
            .input_contract(&model_name)
            .cloned()
            .ok_or_else(|| {
                Status::internal(format!(
                    "model '{model_name}' is missing its input contract"
                ))
            })?;
        let (py_input, expected_batch) =
            match extract_single_input(&contract, &mut request, &model_name)? {
                Some((bytes, shape, canonical, batch)) => (
                    Some(PythonInput {
                        shape,
                        bytes,
                        dtype: canonical.to_string(),
                    }),
                    batch,
                ),
                None => (None, None),
            };

        // Bound concurrent subprocesses; hold the permit across the whole
        // blocking call so the pool count reflects work in flight.
        let permits = self
            .model_manager
            .python_permits(&model_name)
            .ok_or_else(|| {
                Status::internal(format!("Python model '{model_name}' has no permit pool"))
            })?;
        let _permit = permits.try_acquire_owned().map_err(|_| {
            Status::resource_exhausted(format!("model '{model_name}' queue full, retry later"))
        })?;
        let model_dir = self
            .model_manager
            .python_model_dir(&model_name)
            .ok_or_else(|| {
                Status::internal(format!(
                    "Python model '{model_name}' has no model directory"
                ))
            })?;
        let name = model_name.clone();
        let (shape, bytes, out_canonical) =
            tokio::task::spawn_blocking(move || run_python_inference(&name, model_dir, py_input))
                .await
                .map_err(|err| {
                    Status::internal(format!("python inference task failed to join: {err}"))
                })??;
        contract.validate_output_shape(&shape, expected_batch, &model_name)?;
        let (out_kserve, _) = dtype::canonical_to_kserve(&out_canonical).ok_or_else(|| {
            Status::internal(format!(
                "model '{model_name}' returned unsupported output dtype '{out_canonical}'"
            ))
        })?;
        Self::single_output_response(&request, model_name, shape, bytes, out_kserve.to_string())
    }

    /// The multi-tensor inference path: bind the request's named input tensors
    /// to the model's declared inputs (in contract order), run the multi-input
    /// forward pass, and return every declared output (or the client-requested
    /// subset). Reuses the single-tensor batch/shape validation per input.
    async fn infer_multi(
        &self,
        request: ModelInferRequest,
        model_name: String,
    ) -> Result<ModelInferResponse, Status> {
        use std::collections::{HashMap, HashSet};

        let contract = self
            .model_manager
            .multi_contract(&model_name)
            .ok_or_else(|| {
                Status::internal(format!(
                    "model '{model_name}' lost its multi-tensor contract"
                ))
            })?;

        // Raw buffers pair positionally with request.inputs; typed contents are
        // matched per input instead.
        let use_raw = !request.raw_input_contents.is_empty();
        if use_raw && request.raw_input_contents.len() != request.inputs.len() {
            return Err(Status::invalid_argument(
                "raw_input_contents length must equal the number of input tensors",
            ));
        }
        // Map input tensors by name. Reject duplicate names: with
        // raw_input_contents paired positionally, a silent last-wins overwrite
        // could bind the wrong raw buffer to a tensor name.
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
        // Reject inputs not declared by the model (silently dropping client data
        // is worse than an explicit error).
        let declared_inputs: HashSet<&str> =
            contract.inputs.iter().map(|s| s.name.as_str()).collect();
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

        let is_native = self.model_manager.is_native(&model_name);

        // Build one tensor per declared input, in contract order. Rust models get
        // libtorch tensors; native (ONNX/TF) models get raw `NativeTensor`s.
        #[cfg(feature = "torch")]
        let mut input_tensors = Vec::with_capacity(contract.inputs.len());
        let mut native_inputs: Vec<NativeTensor> = Vec::with_capacity(contract.inputs.len());
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
            let (elem_size, canonical) =
                dtype::kserve_fixed_width(&spec.dtype).ok_or_else(|| {
                    Status::invalid_argument(format!(
                        "unsupported datatype '{}' for input '{}'",
                        spec.dtype, spec.name
                    ))
                })?;

            // Validate + normalize the shape via an ephemeral single-tensor
            // contract, reusing the tested batch logic.
            let ephemeral =
                InputShapeContract::new_input(spec.dims.clone(), contract.max_batch_size);
            ephemeral.validate_request_shape(&inp.shape, &model_name)?;
            let shape = ephemeral.normalize_request_shape(inp.shape.clone());

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
                    "input '{}' ({}) requires raw_input_contents",
                    spec.name, spec.dtype
                )));
            };

            let numel = shape
                .iter()
                .try_fold(1i64, |acc, dim| acc.checked_mul(*dim))
                .ok_or_else(|| Status::invalid_argument("input tensor shape overflow"))?;
            if bytes.len() != (numel as usize).saturating_mul(elem_size) {
                return Err(Status::invalid_argument(format!(
                    "input '{}' byte length {} does not match shape {shape:?} \u{d7} {elem_size} bytes",
                    spec.name,
                    bytes.len()
                )));
            }
            // Batch size must agree across all inputs.
            if contract.max_batch_size > 0 {
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

            if is_native {
                native_inputs.push(NativeTensor {
                    name: spec.name.clone(),
                    shape,
                    dtype: canonical.to_string(),
                    data: bytes,
                });
            } else {
                #[cfg(feature = "torch")]
                {
                    // The Rust `.pt` path needs a libtorch kind; uint16/32/64
                    // (which libtorch lacks) are rejected here but work natively.
                    let kind = dtype::kind_from_canonical(canonical).ok_or_else(|| {
                        Status::invalid_argument(format!(
                            "input '{}' datatype {} has no libtorch kind",
                            spec.name, spec.dtype
                        ))
                    })?;
                    input_tensors.push(tensor_from_input_bytes(&bytes, &shape, &model_name, kind)?);
                }
                #[cfg(not(feature = "torch"))]
                {
                    let _ = (&shape, &bytes, canonical);
                    return Err(Status::internal(format!(
                        "model '{model_name}' is a multi-tensor .pt model, but the torch backend \
                         is not compiled in"
                    )));
                }
            }
        }

        let response_rx = if is_native {
            self.model_manager
                .enqueue_native(&model_name, native_inputs)?
        } else {
            #[cfg(feature = "torch")]
            {
                self.model_manager
                    .enqueue_multi(&model_name, input_tensors)?
            }
            #[cfg(not(feature = "torch"))]
            {
                return Err(Status::internal(format!(
                    "model '{model_name}' is a multi-tensor .pt model, but the torch backend \
                     is not compiled in"
                )));
            }
        };
        let outputs = response_rx.await.map_err(|_| {
            Status::internal(format!(
                "worker response channel closed for model '{model_name}'"
            ))
        })??;
        if outputs.len() != contract.outputs.len() {
            return Err(Status::internal(format!(
                "model '{model_name}' returned {} output tensors but its contract declares {}",
                outputs.len(),
                contract.outputs.len()
            )));
        }

        // Return all declared outputs, or the client-requested subset if any.
        // A requested name that isn't a declared output is rejected rather than
        // silently omitted (consistent with the single-tensor path).
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
        for (spec, (shape, bytes, canonical)) in contract.outputs.iter().zip(outputs) {
            if let Some(requested) = &requested
                && !requested.contains(spec.name.as_str())
            {
                continue;
            }
            let (kserve, _) = dtype::canonical_to_kserve(&canonical).ok_or_else(|| {
                Status::internal(format!(
                    "model '{model_name}' output '{}' has unsupported dtype '{canonical}'",
                    spec.name
                ))
            })?;
            // The returned dtype must match what the contract declares (and
            // ModelMetadata advertises) — a mismatch is a model/config bug, not
            // a silently-mislabeled tensor.
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
                shape,
                parameters: Default::default(),
                contents: None,
            });
            raw_outputs.push(bytes);
        }

        Ok(ModelInferResponse {
            model_name,
            model_version: MODEL_VERSION.to_string(),
            id: request.id,
            parameters: Default::default(),
            outputs: out_tensors,
            raw_output_contents: raw_outputs,
        })
    }
}

/// End-to-end tests for the Triton-compatible surface. These drive the
/// generated KServe v2 client over a loopback socket into a live
/// `GrpcInferenceServiceServer`, mirroring exactly what a stock `tritonclient`
/// would do on the wire.
#[cfg(test)]
mod triton_e2e_tests {
    use super::*;
    use crate::config::{ModelConfig, ModelDevice, ServerConfig, ServerSection};
    use crate::inference::run_forward_pass;
    use crate::proto::grpc_inference_service_client::GrpcInferenceServiceClient;
    use crate::proto::grpc_inference_service_server::GrpcInferenceServiceServer;
    use std::net::SocketAddr;
    use std::path::PathBuf;
    use tch::{CModule, Device, Tensor};
    use tokio::net::TcpListener;
    use tokio_stream::wrappers::TcpListenerStream;
    use tonic::transport::Server;

    fn fixtures_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ml-backends")
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
        let model = CModule::load_on_device(model_path.to_str().expect("utf-8 path"), Device::Cpu)
            .expect("model should load");
        let input_tensor = Tensor::from_slice(&input_values).reshape([1, 16]);
        let (expected_shape, expected_bytes, _dtype) =
            run_forward_pass(&model, Device::Cpu, &input_tensor).expect("direct forward pass");

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
