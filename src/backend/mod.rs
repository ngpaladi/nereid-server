//! The backend abstraction: one `Backend` trait every inference engine
//! implements, plus the `ModelManager` that owns each configured model and
//! dispatches to its backend. Core server code (gRPC surfaces, validation) is
//! backend-agnostic and talks only to this module — there is no `match` on
//! backend kinds outside `load`/detection.

pub mod contract;

#[cfg(any(feature = "onnx", feature = "tensorflow"))]
mod native_common;
#[cfg(feature = "onnx")]
mod onnx;
#[cfg(feature = "python")]
mod python;
#[cfg(feature = "tensorflow")]
mod tensorflow;
#[cfg(feature = "torch")]
mod torch;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use tokio::sync::Semaphore;
use tonic::Status;

use crate::config::{ConfiguredBackend, ServerConfig};
use crate::proto::{CheckpointResponse, TensorChunk};

pub use contract::{Contract, TensorSpec};

/// A backend-agnostic tensor at the dispatch boundary. `data` is row-major,
/// little-endian, for the element type named by the canonical `dtype`.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: String,
    pub data: Vec<u8>,
}

/// The Nereid `Checkpoint` streaming response.
pub type CheckpointStream =
    tonic::codegen::tokio_stream::wrappers::ReceiverStream<Result<CheckpointResponse, Status>>;

/// A loaded inference backend. One boxed instance backs each model. `infer` is
/// blocking (run off the async runtime); the `ModelManager` serializes/bounds
/// concurrency, so implementations need only be thread-safe for shared `&self`.
pub trait Backend: Send + Sync {
    /// The Triton `platform` string reported by `ModelMetadata`.
    fn platform(&self) -> &'static str;

    /// Run one inference with already-validated inputs (in the model's declared
    /// input order); return outputs in the model's declared output order.
    fn infer(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, Status>;

    /// An optional richer `Checkpoint` stream (e.g. the Python backend streams
    /// stdout/stderr log chunks alongside the output tensor). Returning `None`
    /// (the default) makes the core run `infer` and stream just the output.
    fn checkpoint_stream(
        &self,
        _model_name: &str,
        _input: Option<Tensor>,
        _contract: &Contract,
    ) -> Option<CheckpointStream> {
        None
    }
}

struct ModelEntry {
    backend: Arc<dyn Backend>,
    contract: Contract,
    permits: Arc<Semaphore>,
}

pub struct ModelManager {
    model_names: Vec<String>,
    entries: HashMap<String, ModelEntry>,
}

fn model_log(msg: &str) {
    println!("[nereid-server] {msg}");
}

impl ModelManager {
    pub fn from_config(config: &ServerConfig) -> Result<Self, Status> {
        let ml_backends_dir =
            std::fs::canonicalize(&config.server.ml_backends_path).map_err(|err| {
                Status::failed_precondition(format!(
                    "failed to resolve server.ml_backends_path '{}': {err}",
                    config.server.ml_backends_path
                ))
            })?;
        if !ml_backends_dir.is_dir() {
            return Err(Status::failed_precondition(format!(
                "server.ml_backends_path '{}' is not a directory",
                ml_backends_dir.display()
            )));
        }

        let mut model_names = Vec::with_capacity(config.models.len());
        let mut entries = HashMap::with_capacity(config.models.len());
        for model_cfg in &config.models {
            let model_dir =
                std::fs::canonicalize(ml_backends_dir.join(&model_cfg.name)).map_err(|err| {
                    Status::failed_precondition(format!(
                        "failed to resolve model directory for '{}' under '{}': {err}",
                        model_cfg.name,
                        ml_backends_dir.display()
                    ))
                })?;
            if !model_dir.is_dir() {
                return Err(Status::failed_precondition(format!(
                    "configured model '{}' is not a directory under '{}'",
                    model_cfg.name,
                    ml_backends_dir.display()
                )));
            }

            let kind = detect_backend(&model_dir, &model_cfg.name, model_cfg.backend)?;
            let (backend, contract) = load_backend(kind, &model_dir, model_cfg)?;

            model_names.push(model_cfg.name.clone());
            entries.insert(
                model_cfg.name.clone(),
                ModelEntry {
                    backend: Arc::from(backend),
                    contract,
                    permits: Arc::new(Semaphore::new(model_cfg.queue_capacity)),
                },
            );
        }

        Ok(Self {
            model_names,
            entries,
        })
    }

    pub fn configured_models(&self) -> Vec<String> {
        self.model_names.clone()
    }

    pub fn is_configured(&self, model_name: &str) -> bool {
        self.entries.contains_key(model_name)
    }

    /// The model's tensor contract, for request validation and metadata.
    pub fn contract(&self, model_name: &str) -> Option<&Contract> {
        self.entries.get(model_name).map(|e| &e.contract)
    }

    /// The Triton platform string for the model.
    pub fn platform(&self, model_name: &str) -> Option<&'static str> {
        self.entries.get(model_name).map(|e| e.backend.platform())
    }

    /// Run one inference. Inputs must already be validated and in the model's
    /// declared input order. Bounds in-flight requests by the model's
    /// `queue_capacity` (full -> `ResourceExhausted`) and runs the blocking
    /// backend off the async runtime.
    pub async fn infer(
        &self,
        model_name: &str,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>, Status> {
        let entry = self.entries.get(model_name).ok_or_else(|| {
            Status::not_found(format!(
                "model '{model_name}' is not configured in nereid.yaml"
            ))
        })?;
        let _permit = entry
            .permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| Status::resource_exhausted("model queue full, retry later"))?;
        let backend = entry.backend.clone();
        let name = model_name.to_string();
        let result = tokio::task::spawn_blocking(move || backend.infer(inputs))
            .await
            .map_err(|err| Status::internal(format!("inference task failed to join: {err}")))?;
        match &result {
            Ok(_) => model_log(&format!("job completed model={name}")),
            Err(status) => model_log(&format!(
                "job failed model={name} error={}",
                status.message()
            )),
        }
        result
    }

    /// The backend's `Checkpoint` stream. Backends may provide a richer stream
    /// (Python: live log chunks); otherwise the output tensor is streamed.
    pub fn checkpoint(
        &self,
        model_name: &str,
        input: Option<Tensor>,
    ) -> Result<CheckpointStream, Status> {
        let entry = self.entries.get(model_name).ok_or_else(|| {
            Status::not_found(format!(
                "model '{model_name}' is not configured in nereid.yaml"
            ))
        })?;
        if let Some(stream) =
            entry
                .backend
                .checkpoint_stream(model_name, input.clone(), &entry.contract)
        {
            return Ok(stream);
        }
        // Default: run inference and stream the single output tensor.
        let backend = entry.backend.clone();
        let permits = entry.permits.clone();
        let name = model_name.to_string();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<CheckpointResponse, Status>>(64);
        tokio::spawn(async move {
            let send_err = |status: Status| {
                let tx = tx.clone();
                async move {
                    let _ = tx.send(Err(status)).await;
                }
            };
            let permit = match permits.try_acquire_owned() {
                Ok(p) => p,
                Err(_) => {
                    send_err(Status::resource_exhausted("model queue full, retry later")).await;
                    return;
                }
            };
            let inputs = input.into_iter().collect::<Vec<_>>();
            let result = tokio::task::spawn_blocking(move || backend.infer(inputs)).await;
            drop(permit);
            let outputs = match result {
                Ok(Ok(outputs)) => outputs,
                Ok(Err(status)) => {
                    send_err(status).await;
                    return;
                }
                Err(err) => {
                    send_err(Status::internal(format!(
                        "inference task failed to join: {err}"
                    )))
                    .await;
                    return;
                }
            };
            stream_outputs(&tx, &name, outputs).await;
        });
        Ok(CheckpointStream::new(rx))
    }
}

/// Stream a completed inference's output tensor(s) as `Checkpoint` responses:
/// a leading text chunk, the output tensor in chunks, then a final `done`.
async fn stream_outputs(
    tx: &tokio::sync::mpsc::Sender<Result<CheckpointResponse, Status>>,
    model_name: &str,
    outputs: Vec<Tensor>,
) {
    const OUTPUT_CHUNK_BYTES: usize = 64 * 1024;
    let _ = tx
        .send(Ok(CheckpointResponse {
            chunk: format!("inference completed for model '{model_name}'"),
            done: false,
            exit_code: 0,
            output_chunk: None,
        }))
        .await;

    for out in &outputs {
        let num_chunks = out.data.len().div_ceil(OUTPUT_CHUNK_BYTES).max(1);
        for (chunk_index, data_chunk) in out
            .data
            .chunks(OUTPUT_CHUNK_BYTES)
            .chain(std::iter::once(&[][..]).filter(|_| out.data.is_empty()))
            .enumerate()
        {
            let _ = tx
                .send(Ok(CheckpointResponse {
                    chunk: String::new(),
                    done: false,
                    exit_code: 0,
                    output_chunk: Some(TensorChunk {
                        tensor_name: out.name.clone(),
                        shape: out.shape.clone(),
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
}

/// Which backend a model uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BackendKind {
    Torch,
    Python,
    Onnx,
    Tensorflow,
}

fn dir_has_ext(model_dir: &Path, ext: &str) -> bool {
    std::fs::read_dir(model_dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .any(|e| e.path().extension().is_some_and(|x| x == ext))
        })
        .unwrap_or(false)
}

fn dir_is_saved_model(model_dir: &Path) -> bool {
    model_dir.join("saved_model.pb").is_file() && model_dir.join("variables").is_dir()
}

/// Detect the backend from folder contents, honouring an explicit `backend:`.
/// Detection is pure file inspection (always compiled); whether the backend is
/// actually built in is checked at `load`.
fn detect_backend(
    model_dir: &Path,
    model_name: &str,
    declared: Option<ConfiguredBackend>,
) -> Result<BackendKind, Status> {
    let has_textproto = model_dir.join("model_inference.textproto").is_file();
    let has_python =
        model_dir.join("main.py").is_file() && model_dir.join("requirements.txt").is_file();
    let has_torch = has_textproto && dir_has_ext(model_dir, "pt");
    let has_onnx = has_textproto && dir_has_ext(model_dir, "onnx");
    let has_tf = has_textproto && dir_is_saved_model(model_dir);

    if let Some(declared) = declared {
        return match declared {
            ConfiguredBackend::Python if has_python => Ok(BackendKind::Python),
            ConfiguredBackend::Python => Err(Status::failed_precondition(format!(
                "model '{model_name}' declares backend \"python\" but is missing main.py + requirements.txt"
            ))),
            ConfiguredBackend::Torch if has_torch => Ok(BackendKind::Torch),
            ConfiguredBackend::Torch => Err(Status::failed_precondition(format!(
                "model '{model_name}' declares backend \"torch\" but is missing a .pt model + model_inference.textproto"
            ))),
            ConfiguredBackend::Onnx if has_onnx => Ok(BackendKind::Onnx),
            ConfiguredBackend::Onnx => Err(Status::failed_precondition(format!(
                "model '{model_name}' declares backend \"onnx\" but is missing a .onnx model + model_inference.textproto"
            ))),
            ConfiguredBackend::Tensorflow if has_tf => Ok(BackendKind::Tensorflow),
            ConfiguredBackend::Tensorflow => Err(Status::failed_precondition(format!(
                "model '{model_name}' declares backend \"tensorflow\" but is missing a SavedModel (saved_model.pb + variables/) + model_inference.textproto"
            ))),
        };
    }

    let mut present = Vec::new();
    if has_python {
        present.push(BackendKind::Python);
    }
    if has_torch {
        present.push(BackendKind::Torch);
    }
    if has_onnx {
        present.push(BackendKind::Onnx);
    }
    if has_tf {
        present.push(BackendKind::Tensorflow);
    }
    match present.len() {
        1 => Ok(present[0]),
        0 => Err(Status::failed_precondition(format!(
            "model '{model_name}' folder must contain exactly one backend's files: (main.py + requirements.txt), (a .pt model + model_inference.textproto), (a .onnx model + model_inference.textproto), or (a SavedModel + model_inference.textproto)"
        ))),
        _ => Err(Status::failed_precondition(format!(
            "model '{model_name}' folder contains files for multiple backends ({present:?}); set `backend` in nereid.yaml to disambiguate"
        ))),
    }
}

/// Load the detected backend, or fail with a clear "rebuild with --features"
/// message when the backend isn't compiled in.
#[cfg_attr(
    not(any(
        feature = "torch",
        feature = "python",
        feature = "onnx",
        feature = "tensorflow"
    )),
    allow(unused_variables)
)]
fn load_backend(
    kind: BackendKind,
    model_dir: &Path,
    model_cfg: &crate::config::ModelConfig,
) -> Result<(Box<dyn Backend>, Contract), Status> {
    match kind {
        #[cfg(feature = "torch")]
        BackendKind::Torch => torch::TorchBackend::load(model_dir, model_cfg),
        #[cfg(not(feature = "torch"))]
        BackendKind::Torch => Err(missing_feature(
            &model_cfg.name,
            "TorchScript (.pt)",
            "torch",
        )),
        #[cfg(feature = "python")]
        BackendKind::Python => python::PythonBackend::load(model_dir, model_cfg),
        #[cfg(not(feature = "python"))]
        BackendKind::Python => Err(missing_feature(
            &model_cfg.name,
            "Python (main.py)",
            "python",
        )),
        #[cfg(feature = "onnx")]
        BackendKind::Onnx => onnx::OnnxBackend::load(model_dir, model_cfg),
        #[cfg(not(feature = "onnx"))]
        BackendKind::Onnx => Err(missing_feature(&model_cfg.name, "ONNX", "onnx")),
        #[cfg(feature = "tensorflow")]
        BackendKind::Tensorflow => tensorflow::TensorflowBackend::load(model_dir, model_cfg),
        #[cfg(not(feature = "tensorflow"))]
        BackendKind::Tensorflow => {
            Err(missing_feature(&model_cfg.name, "TensorFlow", "tensorflow"))
        }
    }
}

#[allow(dead_code)]
fn missing_feature(model_name: &str, what: &str, feature: &str) -> Status {
    Status::failed_precondition(format!(
        "model '{model_name}' is a {what} model, but this server was built without that backend. \
         Rebuild with `--features {feature}` (or `./build.sh --{feature}`)."
    ))
}

#[cfg(test)]
mod tests {
    use super::{BackendKind, detect_backend};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("nereid-detect-{prefix}-{nanos}"));
        fs::create_dir_all(&path).expect("create temp dir");
        path
    }

    #[test]
    fn detect_backend_recognizes_python_folder() {
        let base = temp_dir("python");
        fs::write(base.join("main.py"), b"print('hi')").expect("write main.py");
        fs::write(base.join("requirements.txt"), b"numpy").expect("write requirements.txt");

        assert_eq!(
            detect_backend(&base, "model", None).expect("should detect python"),
            BackendKind::Python
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn detect_backend_recognizes_torch_folder() {
        let base = temp_dir("torch");
        fs::write(
            base.join("model_inference.textproto"),
            b"input_shape: [16]\n",
        )
        .expect("write textproto");
        fs::write(base.join("model.pt"), b"x").expect("write model.pt");

        assert_eq!(
            detect_backend(&base, "model", None).expect("should detect torch"),
            BackendKind::Torch
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn detect_backend_rejects_ambiguous_folder() {
        let base = temp_dir("ambiguous");
        fs::write(base.join("main.py"), b"print('hi')").expect("write main.py");
        fs::write(base.join("requirements.txt"), b"numpy").expect("write requirements.txt");
        fs::write(
            base.join("model_inference.textproto"),
            b"input_shape: [16]\n",
        )
        .expect("write textproto");
        fs::write(base.join("model.pt"), b"x").expect("write model.pt");

        assert!(detect_backend(&base, "model", None).is_err());

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn detect_backend_rejects_empty_folder() {
        let base = temp_dir("empty");
        assert!(detect_backend(&base, "model", None).is_err());
        let _ = fs::remove_dir_all(&base);
    }
}
