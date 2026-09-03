//! The backend abstraction: one `Backend` trait every inference engine
//! implements, plus the `ModelManager` that owns each configured model and
//! dispatches to its backend. Core server code (gRPC surfaces, validation) is
//! backend-agnostic and talks only to this module — there is no `match` on
//! backend kinds outside `load`/detection.

pub mod contract;

// Shared tensor helpers for the in-process native backends. `pub(crate)` so the
// backend modules (which live under `crate::backends`) can reach them.
#[cfg(any(feature = "onnx", feature = "tensorflow"))]
pub(crate) mod native_common;

// The backends themselves live one folder each under `src/backends/` and are
// discovered by build.rs — see `crate::backends`. They self-register into the
// registry below, so nothing here enumerates them.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Semaphore;
use tonic::Status;

use crate::config::{ModelConfig, ServerConfig};
use crate::metrics::{InferenceMetrics, RequestTimer};
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
    /// Per-model Prometheus counters, shared with the `/metrics` endpoint.
    metrics: Arc<InferenceMetrics>,
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

            let registration =
                detect_backend(&model_dir, &model_cfg.name, model_cfg.backend.as_deref())?;
            let (backend, contract) = (registration.load)(&model_dir, model_cfg)?;
            model_log(&format!(
                "loaded model={} backend={} version={}",
                model_cfg.name, registration.name, registration.version
            ));

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

        let metrics = Arc::new(InferenceMetrics::new(model_names.iter().cloned()));
        Ok(Self {
            model_names,
            entries,
            metrics,
        })
    }

    pub fn configured_models(&self) -> Vec<String> {
        self.model_names.clone()
    }

    /// The Prometheus registry every request is accounted in.
    pub fn metrics(&self) -> &Arc<InferenceMetrics> {
        &self.metrics
    }

    /// Open the metrics accounting for one request to `model_name`. Call it as
    /// soon as the request has named a configured model, before building its
    /// inputs, so `compute_input` and validation failures are attributed.
    pub fn begin_request(&self, model_name: &str) -> Result<RequestTimer, Status> {
        self.metrics.begin(model_name).ok_or_else(|| {
            Status::not_found(format!(
                "model '{model_name}' is not configured in nereid.yaml"
            ))
        })
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
    /// backend off the async runtime. Records the queue wait, the backend's
    /// execution time, and the batch size on `timer`; the caller still ends the
    /// timer with the request's overall outcome.
    pub async fn infer(
        &self,
        model_name: &str,
        inputs: Vec<Tensor>,
        timer: &mut RequestTimer,
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
        let batch = inference_count(&entry.contract, &inputs);
        let backend = entry.backend.clone();
        let name = model_name.to_string();
        let execution = run_backend(backend, inputs).await;
        let result = match execution {
            Ok(execution) => {
                timer.executed(execution.queued, execution.ran, batch);
                execution.result
            }
            Err(status) => Err(status),
        };
        match &result {
            Ok(_) => model_log(&format!("job completed model={name}")),
            Err(status) => {
                timer.backend_failed();
                model_log(&format!(
                    "job failed model={name} error={}",
                    status.message()
                ))
            }
        }
        result
    }

    /// The backend's `Checkpoint` stream. Backends may provide a richer stream
    /// (Python: live log chunks); otherwise the output tensor is streamed. Takes
    /// over `timer` and ends it when the stream does: `done` with exit code 0
    /// is a success, anything else a failure.
    pub fn checkpoint(
        &self,
        model_name: &str,
        input: Option<Tensor>,
        mut timer: RequestTimer,
    ) -> Result<CheckpointStream, Status> {
        let entry = match self.entries.get(model_name) {
            Some(entry) => entry,
            None => {
                let status = Status::not_found(format!(
                    "model '{model_name}' is not configured in nereid.yaml"
                ));
                timer.fail(&status);
                return Err(status);
            }
        };
        let batch = inference_count(&entry.contract, input.as_slice());
        if let Some(stream) =
            entry
                .backend
                .checkpoint_stream(model_name, input.clone(), &entry.contract)
        {
            return Ok(record_stream(stream, timer, batch));
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
                    let status = Status::resource_exhausted("model queue full, retry later");
                    timer.fail(&status);
                    send_err(status).await;
                    return;
                }
            };
            let inputs = input.into_iter().collect::<Vec<_>>();
            let execution = run_backend(backend, inputs).await;
            drop(permit);
            let outputs = match execution {
                Ok(execution) => {
                    timer.executed(execution.queued, execution.ran, batch);
                    match execution.result {
                        Ok(outputs) => outputs,
                        Err(status) => {
                            timer.backend_failed();
                            timer.fail(&status);
                            send_err(status).await;
                            return;
                        }
                    }
                }
                Err(status) => {
                    timer.backend_failed();
                    timer.fail(&status);
                    send_err(status).await;
                    return;
                }
            };
            stream_outputs(&tx, &name, outputs).await;
            timer.succeed();
        });
        Ok(CheckpointStream::new(rx))
    }
}

/// The number of inferences a request represents for `nv_inference_count`: its
/// batch size when the model declares a batch dimension, else 1 (an input-less
/// model runs once).
fn inference_count(contract: &Contract, inputs: &[Tensor]) -> u64 {
    inputs
        .first()
        .and_then(|tensor| contract.expected_batch(&tensor.shape))
        .map_or(1, |batch| u64::try_from(batch).unwrap_or(1).max(1))
}

/// One timed run of a backend's blocking `infer` on the blocking pool.
struct Execution {
    /// How long the request waited for a blocking worker to pick it up.
    queued: Duration,
    /// How long the backend's `infer` ran.
    ran: Duration,
    result: Result<Vec<Tensor>, Status>,
}

/// Run `backend.infer` off the async runtime, timing the wait and the run.
/// `Err` only when the blocking task itself fails to join (it panicked).
async fn run_backend(backend: Arc<dyn Backend>, inputs: Vec<Tensor>) -> Result<Execution, Status> {
    let queued_at = Instant::now();
    let (started, result) = tokio::task::spawn_blocking(move || {
        let started = Instant::now();
        (started, backend.infer(inputs))
    })
    .await
    .map_err(|err| Status::internal(format!("inference task failed to join: {err}")))?;
    Ok(Execution {
        queued: started.duration_since(queued_at),
        ran: started.elapsed(),
        result,
    })
}

/// Relay a backend-provided `Checkpoint` stream, ending `timer` when it ends.
/// The backend ran the request as one opaque step, so its whole span counts as
/// `compute_infer`; a terminal `done` with exit code 0 is a success, a non-zero
/// exit or an error status a backend failure, and a client that stops reading
/// a cancellation.
fn record_stream(inner: CheckpointStream, mut timer: RequestTimer, batch: u64) -> CheckpointStream {
    let mut inner = inner.into_inner();
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<CheckpointResponse, Status>>(64);
    tokio::spawn(async move {
        let mut outcome: Option<Result<(), Status>> = None;
        while let Some(item) = inner.recv().await {
            match &item {
                Ok(response) if response.done => {
                    outcome = Some(if response.exit_code == 0 {
                        Ok(())
                    } else {
                        Err(Status::internal(format!(
                            "model process exited with code {}",
                            response.exit_code
                        )))
                    });
                }
                Err(status) => outcome = Some(Err(status.clone())),
                Ok(_) => {}
            }
            if tx.send(item).await.is_err() {
                outcome = Some(Err(Status::cancelled("client stopped reading the stream")));
                break;
            }
        }
        match outcome {
            Some(Ok(())) => {
                let ran = timer.phase_elapsed();
                timer.executed(Duration::ZERO, ran, batch);
                timer.succeed();
            }
            Some(Err(status)) => {
                if status.code() != tonic::Code::Cancelled {
                    timer.backend_failed();
                }
                timer.fail(&status);
            }
            None => timer.fail(&Status::internal(
                "checkpoint stream ended without a result",
            )),
        }
    });
    CheckpointStream::new(rx)
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

/// A backend's loader: file path + model config in, a boxed backend plus its
/// parsed contract out (or a missing-feature error when the engine is off).
pub type LoadFn = fn(&Path, &ModelConfig) -> Result<(Box<dyn Backend>, Contract), Status>;

/// A backend's self-registration, collected at link time via [`inventory`]. Each
/// backend module `submit!`s exactly one of these, so the core never enumerates
/// backends — it iterates the registrations. Every field is always compiled (no
/// backend dependency), so the registry is complete even for backends whose
/// engine feature is off; `load` then returns a "rebuild with --features" error.
pub struct BackendRegistration {
    /// The `backend:` value in `nereid.yaml` (and the Cargo feature that
    /// provides this backend).
    pub name: &'static str,
    /// The backend's own version, `major.minor.patch`, independent of the
    /// server's. Backends evolve on their own schedule — especially
    /// out-of-tree ones vendored as submodules — so each advertises where it
    /// is: bump the major when a revision changes the folder shape, the
    /// contract, or the config a model must supply. Reported in the startup log
    /// for every loaded model, so a deployment can be traced to the exact
    /// backend revision that served it.
    pub version: &'static str,
    /// Additional accepted `backend:` spellings (e.g. deprecated aliases).
    pub aliases: &'static [&'static str],
    /// A one-line description of the folder shape, for the ambiguity/no-match
    /// errors.
    pub describes: &'static str,
    /// Whether the backend can be auto-detected from folder contents. A backend
    /// whose code is compiled into the server (no on-disk signature) sets this
    /// `false` and is selectable only via an explicit `backend:` declaration.
    pub auto_detect: bool,
    /// File-signature detection — pure inspection, no backend dependency.
    pub detect: fn(&Path) -> bool,
    /// Load the model, or return a missing-feature error when this backend's
    /// feature is not compiled in.
    pub load: LoadFn,
}

inventory::collect!(BackendRegistration);

fn registrations() -> impl Iterator<Item = &'static BackendRegistration> {
    inventory::iter::<BackendRegistration>()
}

/// "Does the folder contain a file with this extension?" — the one file-signature
/// helper that is genuinely backend-agnostic (torch's `.pt`, ONNX's `.onnx`, and
/// any future single-file format). Used by the backends' `detect` predicates,
/// which live under `crate::backends`, hence `pub(crate)`. Anything specific to
/// one backend's folder shape belongs in that backend's own folder.
pub(crate) fn dir_has_ext(model_dir: &Path, ext: &str) -> bool {
    std::fs::read_dir(model_dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .any(|e| e.path().extension().is_some_and(|x| x == ext))
        })
        .unwrap_or(false)
}

/// Pick the backend for a model: the explicitly-declared one, else the single
/// auto-detected match. Consults only the registry, so it never names a specific
/// backend — new backends slot in by registering.
fn detect_backend(
    model_dir: &Path,
    model_name: &str,
    declared: Option<&str>,
) -> Result<&'static BackendRegistration, Status> {
    if let Some(declared) = declared {
        let registration = registrations()
            .find(|r| r.name == declared || r.aliases.contains(&declared))
            .ok_or_else(|| {
                let known: Vec<&str> = registrations().map(|r| r.name).collect();
                Status::failed_precondition(format!(
                    "model '{model_name}' declares unknown backend \"{declared}\"; \
                 known backends: {}",
                    known.join(", ")
                ))
            })?;
        if registration.auto_detect && !(registration.detect)(model_dir) {
            return Err(Status::failed_precondition(format!(
                "model '{model_name}' declares backend \"{declared}\" but its folder is not {}",
                registration.describes
            )));
        }
        return Ok(registration);
    }

    let matches: Vec<&BackendRegistration> = registrations()
        .filter(|r| r.auto_detect && (r.detect)(model_dir))
        .collect();
    match matches.as_slice() {
        [only] => Ok(only),
        [] => {
            let shapes: Vec<&str> = registrations()
                .filter(|r| r.auto_detect)
                .map(|r| r.describes)
                .collect();
            Err(Status::failed_precondition(format!(
                "model '{model_name}' folder must match exactly one backend, one of: {}",
                shapes.join("; ")
            )))
        }
        many => {
            let names: Vec<&str> = many.iter().map(|r| r.name).collect();
            Err(Status::failed_precondition(format!(
                "model '{model_name}' folder matches multiple backends ({}); set `backend` in nereid.yaml to disambiguate",
                names.join(", ")
            )))
        }
    }
}

/// The "rebuild with --features" error a backend's `load` returns when its
/// engine feature is off. Kept in the core so every backend phrases it the same.
/// (Unused when every backend feature is enabled — no `load` hits its off-arm.)
#[allow(dead_code)]
pub(crate) fn missing_feature(model_name: &str, what: &str, feature: &str) -> Status {
    Status::failed_precondition(format!(
        "model '{model_name}' is a {what} model, but this server was built without that backend. \
         Rebuild with `--features {feature}` (or `./build.sh --{feature}`)."
    ))
}

#[cfg(test)]
mod tests {
    use super::detect_backend;
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
            detect_backend(&base, "model", None)
                .expect("should detect python")
                .name,
            "python"
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
            detect_backend(&base, "model", None)
                .expect("should detect torch")
                .name,
            "torch"
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn detect_backend_declared_unknown_is_rejected() {
        let base = temp_dir("unknown");
        fs::write(base.join("main.py"), b"print('hi')").expect("write main.py");
        fs::write(base.join("requirements.txt"), b"numpy").expect("write requirements.txt");

        assert!(detect_backend(&base, "model", Some("nope")).is_err());

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
