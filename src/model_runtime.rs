use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc as std_mpsc;

use tch::{CModule, Device, Tensor};
use tokio::sync::{Semaphore, oneshot};
use tonic::Status;

use crate::config::{ConfiguredBackend, ServerConfig};
use crate::inference;
use crate::python_backend;

/// `(shape, row-major little-endian bytes, canonical dtype)` — the dtype lets
/// the Rust path return non-float outputs (e.g. `int64`) faithfully.
pub type InferenceOutput = (Vec<i64>, Vec<u8>, String);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputShapeContract {
    /// Declared input tensor shape (excluding batch). `None` for a model that
    /// consumes no tensor input (e.g. an output-only Python producer).
    input_shape: Option<Vec<i64>>,
    max_batch_size: i64,
    /// Declared output tensor shape (excluding the batch dimension). Required
    /// for Python models — every Python reply is a typed tensor.
    output_shape: Option<Vec<i64>>,
    /// Declared input KServe datatype (e.g. `"INT32"`). Absent means `FP32`,
    /// preserving the original float-only behavior. Requests whose datatype
    /// disagrees with this are rejected.
    data_type: Option<String>,
}

/// One named tensor in a multi-tensor model contract (Triton `config.pbtxt`
/// `input {}` / `output {}` block). `dims` excludes the batch dimension.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    /// KServe datatype string, e.g. `"FP32"`.
    pub dtype: String,
    pub dims: Vec<i64>,
}

/// A model that consumes and/or produces more than one named tensor. This is the
/// additive parallel path — the single-tensor [`InputShapeContract`] and its
/// well-tested Checkpoint path are untouched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiTensorContract {
    pub inputs: Vec<TensorSpec>,
    pub outputs: Vec<TensorSpec>,
    pub max_batch_size: i64,
}

#[derive(Debug)]
enum ModelHandle {
    Rust(RustModelHandle),
    RustMulti(RustMultiHandle),
    Python(PythonModelHandle),
}

#[derive(Debug)]
struct RustModelHandle {
    input_contract: InputShapeContract,
    job_tx: std_mpsc::SyncSender<InferenceJob>,
    queue_capacity: usize,
    occupancy: Arc<AtomicUsize>,
}

#[derive(Debug)]
struct PythonModelHandle {
    model_dir: PathBuf,
    /// Required for Python models: declares the output tensor (and optionally an
    /// input tensor). Every Python reply is a typed tensor.
    contract: InputShapeContract,
    /// Bounds concurrent `main.py` subprocesses to `queue_capacity`, mirroring
    /// the Rust backend's bounded job queue. A ModelInfer call must hold a
    /// permit for the duration of the subprocess; when none are free the call
    /// is rejected with `ResourceExhausted`.
    permits: Arc<Semaphore>,
}

#[derive(Debug)]
struct RustMultiHandle {
    contract: MultiTensorContract,
    job_tx: std_mpsc::SyncSender<MultiInferenceJob>,
    queue_capacity: usize,
    occupancy: Arc<AtomicUsize>,
}

#[derive(Debug)]
struct InferenceJob {
    input_tensor: Tensor,
    response_tx: oneshot::Sender<Result<InferenceOutput, Status>>,
    occupancy: Arc<AtomicUsize>,
}

#[derive(Debug)]
struct MultiInferenceJob {
    input_tensors: Vec<Tensor>,
    response_tx: oneshot::Sender<Result<Vec<InferenceOutput>, Status>>,
    occupancy: Arc<AtomicUsize>,
}

#[derive(Debug)]
pub struct ModelManager {
    model_names: Vec<String>,
    handles: HashMap<String, ModelHandle>,
}

fn model_log(msg: &str) {
    println!("[nereid-server] {msg}");
}

impl ModelManager {
    pub fn from_config(config: &ServerConfig) -> Result<Self, Status> {
        let mut model_names = Vec::with_capacity(config.models.len());
        let mut handles = HashMap::with_capacity(config.models.len());
        let ml_backends_path = Path::new(&config.server.ml_backends_path);
        let ml_backends_dir = fs::canonicalize(ml_backends_path).map_err(|err| {
            Status::failed_precondition(format!(
                "failed to resolve server.ml_backends_path '{}': {err}",
                ml_backends_path.display()
            ))
        })?;
        if !ml_backends_dir.is_dir() {
            return Err(Status::failed_precondition(format!(
                "server.ml_backends_path '{}' is not a directory",
                ml_backends_dir.display()
            )));
        }

        for model_cfg in &config.models {
            let model_dir =
                fs::canonicalize(ml_backends_dir.join(&model_cfg.name)).map_err(|err| {
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

            match detect_backend_kind(&model_dir, &model_cfg.name, model_cfg.backend)? {
                DetectedBackendKind::Python => {
                    python_backend::prepare_model_envs(
                        std::slice::from_ref(&model_cfg.name),
                        &ml_backends_dir,
                    )
                    .map_err(|err| {
                        Status::failed_precondition(format!(
                            "failed to prepare python environment for model '{}': {err}",
                            model_cfg.name
                        ))
                    })?;

                    // A Python model must ship a `model_inference.textproto`
                    // that declares `output_shape`: every reply is a typed
                    // output tensor. An `input_shape` is optional.
                    if !model_dir.join("model_inference.textproto").is_file() {
                        return Err(Status::failed_precondition(format!(
                            "Python model '{}' must include a model_inference.textproto declaring output_shape",
                            model_cfg.name
                        )));
                    }
                    let contract = read_input_contract_from_textproto(&model_dir)?;
                    if contract.output_shape().is_none() {
                        return Err(Status::failed_precondition(format!(
                            "Python model '{}' must declare output_shape in model_inference.textproto (every reply is a typed tensor)",
                            model_cfg.name
                        )));
                    }

                    model_names.push(model_cfg.name.clone());
                    handles.insert(
                        model_cfg.name.clone(),
                        ModelHandle::Python(PythonModelHandle {
                            model_dir,
                            contract,
                            permits: Arc::new(Semaphore::new(model_cfg.queue_capacity)),
                        }),
                    );
                }
                DetectedBackendKind::Rust => {
                    let pt_file = find_exactly_one_pt_model_file(&model_dir)?;
                    let device = model_cfg.device.to_tch_device()?;
                    let model_path = pt_file.to_string_lossy().into_owned();

                    let model = CModule::load_on_device(&model_path, device).map_err(|err| {
                        Status::failed_precondition(format!(
                            "failed to load .pt model for '{}': {err}",
                            model_cfg.name
                        ))
                    })?;
                    let occupancy = Arc::new(AtomicUsize::new(0));

                    // A textproto using nested `input {}` / `output {}` blocks is
                    // a multi-tensor model (additive path); a flat
                    // `input_shape` textproto is the original single-tensor one.
                    if textproto_is_multi(&model_dir) {
                        let contract = read_multi_contract_from_textproto(&model_dir)?;
                        let (job_tx, job_rx) =
                            std_mpsc::sync_channel::<MultiInferenceJob>(model_cfg.queue_capacity);
                        spawn_multi_worker(model_cfg.name.clone(), model, device, job_rx);
                        model_names.push(model_cfg.name.clone());
                        handles.insert(
                            model_cfg.name.clone(),
                            ModelHandle::RustMulti(RustMultiHandle {
                                contract,
                                job_tx,
                                queue_capacity: model_cfg.queue_capacity,
                                occupancy,
                            }),
                        );
                    } else {
                        let input_contract = read_input_contract_from_textproto(&model_dir)?;
                        if !input_contract.has_input() {
                            return Err(Status::failed_precondition(format!(
                                "Rust model '{}' must declare input_shape in model_inference.textproto",
                                model_cfg.name
                            )));
                        }
                        let (job_tx, job_rx) =
                            std_mpsc::sync_channel::<InferenceJob>(model_cfg.queue_capacity);
                        spawn_model_worker(model_cfg.name.clone(), model, device, job_rx);
                        model_names.push(model_cfg.name.clone());
                        handles.insert(
                            model_cfg.name.clone(),
                            ModelHandle::Rust(RustModelHandle {
                                input_contract,
                                job_tx,
                                queue_capacity: model_cfg.queue_capacity,
                                occupancy,
                            }),
                        );
                    }
                }
            }
        }

        Ok(Self {
            model_names,
            handles,
        })
    }

    pub fn configured_models(&self) -> Vec<String> {
        self.model_names.clone()
    }

    /// Whether `model_name` is a configured model (any backend kind).
    pub fn is_configured(&self, model_name: &str) -> bool {
        self.handles.contains_key(model_name)
    }

    /// Whether `model_name` is a Python (`main.py`) backend.
    pub fn is_python(&self, model_name: &str) -> bool {
        matches!(self.handles.get(model_name), Some(ModelHandle::Python(_)))
    }

    pub fn input_contract(&self, model_name: &str) -> Option<&InputShapeContract> {
        match self.handles.get(model_name)? {
            ModelHandle::Rust(handle) => Some(&handle.input_contract),
            ModelHandle::Python(handle) => Some(&handle.contract),
            ModelHandle::RustMulti(_) => None,
        }
    }

    /// Whether `model_name` is a multi-tensor Rust model (served via the named
    /// multi-tensor inference path rather than the single-tensor one).
    pub fn is_multi(&self, model_name: &str) -> bool {
        matches!(
            self.handles.get(model_name),
            Some(ModelHandle::RustMulti(_))
        )
    }

    /// The multi-tensor contract for a `RustMulti` model, if `model_name` is one.
    pub fn multi_contract(&self, model_name: &str) -> Option<MultiTensorContract> {
        match self.handles.get(model_name)? {
            ModelHandle::RustMulti(handle) => Some(handle.contract.clone()),
            _ => None,
        }
    }

    pub fn python_model_dir(&self, model_name: &str) -> Option<PathBuf> {
        match self.handles.get(model_name)? {
            ModelHandle::Python(handle) => Some(handle.model_dir.clone()),
            _ => None,
        }
    }

    /// The concurrency permits for a Python model's subprocess pool, for the
    /// ModelInfer backpressure gate. `None` for non-Python models.
    pub fn python_permits(&self, model_name: &str) -> Option<Arc<Semaphore>> {
        match self.handles.get(model_name)? {
            ModelHandle::Python(handle) => Some(handle.permits.clone()),
            _ => None,
        }
    }

    pub fn enqueue(
        &self,
        model_name: &str,
        input_tensor: Tensor,
    ) -> Result<oneshot::Receiver<Result<InferenceOutput, Status>>, Status> {
        let handle = match self.handles.get(model_name) {
            Some(ModelHandle::Rust(handle)) => handle,
            Some(ModelHandle::RustMulti(_)) => {
                return Err(Status::failed_precondition(format!(
                    "model '{model_name}' is a multi-tensor model; use the multi inference path"
                )));
            }
            Some(ModelHandle::Python(_)) => {
                return Err(Status::failed_precondition(format!(
                    "model '{model_name}' is a python backend and does not support tensor inference"
                )));
            }
            None => {
                return Err(Status::not_found(format!(
                    "model '{model_name}' is not configured in nereid.yaml"
                )));
            }
        };

        let (response_tx, response_rx) = oneshot::channel();
        let occupancy = handle.occupancy.clone();
        let job = InferenceJob {
            input_tensor,
            response_tx,
            occupancy: occupancy.clone(),
        };

        let current = occupancy.fetch_add(1, Ordering::SeqCst) + 1;
        match handle.job_tx.try_send(job) {
            Ok(()) => {
                model_log(&format!(
                    "queue status model={model_name} queue={current}/{}",
                    handle.queue_capacity
                ));
                Ok(response_rx)
            }
            Err(std_mpsc::TrySendError::Full(_)) => {
                let current = occupancy.fetch_sub(1, Ordering::SeqCst) - 1;
                model_log(&format!(
                    "queue status model={model_name} queue={current}/{} full",
                    handle.queue_capacity
                ));
                Err(Status::resource_exhausted("model queue full, retry later"))
            }
            Err(std_mpsc::TrySendError::Disconnected(_)) => {
                occupancy.fetch_sub(1, Ordering::SeqCst);
                Err(Status::internal(format!(
                    "worker for model '{model_name}' is unavailable"
                )))
            }
        }
    }

    /// Enqueue a multi-tensor inference job. Mirrors [`Self::enqueue`]'s bounded
    /// queue + backpressure for the `RustMulti` path.
    pub fn enqueue_multi(
        &self,
        model_name: &str,
        input_tensors: Vec<Tensor>,
    ) -> Result<oneshot::Receiver<Result<Vec<InferenceOutput>, Status>>, Status> {
        let handle = match self.handles.get(model_name) {
            Some(ModelHandle::RustMulti(handle)) => handle,
            Some(_) => {
                return Err(Status::failed_precondition(format!(
                    "model '{model_name}' is not a multi-tensor model"
                )));
            }
            None => {
                return Err(Status::not_found(format!(
                    "model '{model_name}' is not configured in nereid.yaml"
                )));
            }
        };

        let (response_tx, response_rx) = oneshot::channel();
        let occupancy = handle.occupancy.clone();
        let job = MultiInferenceJob {
            input_tensors,
            response_tx,
            occupancy: occupancy.clone(),
        };

        let current = occupancy.fetch_add(1, Ordering::SeqCst) + 1;
        match handle.job_tx.try_send(job) {
            Ok(()) => {
                model_log(&format!(
                    "queue status model={model_name} queue={current}/{}",
                    handle.queue_capacity
                ));
                Ok(response_rx)
            }
            Err(std_mpsc::TrySendError::Full(_)) => {
                occupancy.fetch_sub(1, Ordering::SeqCst);
                Err(Status::resource_exhausted("model queue full, retry later"))
            }
            Err(std_mpsc::TrySendError::Disconnected(_)) => {
                occupancy.fetch_sub(1, Ordering::SeqCst);
                Err(Status::internal(format!(
                    "worker for model '{model_name}' is unavailable"
                )))
            }
        }
    }
}

fn spawn_multi_worker(
    model_name: String,
    model: CModule,
    device: Device,
    job_rx: std_mpsc::Receiver<MultiInferenceJob>,
) {
    let thread_name = format!("nereid-multi-{model_name}");
    let _ = std::thread::Builder::new()
        .name(thread_name)
        .spawn(move || {
            for job in job_rx {
                let result = inference::run_multi_forward_pass(&model, device, job.input_tensors)
                    .map_err(|err| {
                        Status::internal(format!(
                            "multi-tensor inference failed for '{model_name}': {err}"
                        ))
                    });
                match &result {
                    Ok(_) => model_log(&format!("multi job completed model={model_name}")),
                    Err(status) => model_log(&format!(
                        "multi job failed model={model_name} error={}",
                        status.message()
                    )),
                }
                let _ = job.response_tx.send(result);
                job.occupancy.fetch_sub(1, Ordering::SeqCst);
            }
        });
}

fn spawn_model_worker(
    model_name: String,
    model: CModule,
    device: Device,
    job_rx: std_mpsc::Receiver<InferenceJob>,
) {
    let thread_name = format!("nereid-model-{model_name}");
    let _ = std::thread::Builder::new()
        .name(thread_name)
        .spawn(move || {
            for job in job_rx {
                let result = inference::run_forward_pass(&model, device, &job.input_tensor)
                    .map_err(|err| {
                        Status::internal(format!("Rust inference failed for '{model_name}': {err}"))
                    });
                match &result {
                    Ok(_) => model_log(&format!("job completed model={model_name}")),
                    Err(status) => model_log(&format!(
                        "job failed model={model_name} error={}",
                        status.message()
                    )),
                }
                let _ = job.response_tx.send(result);
                job.occupancy.fetch_sub(1, Ordering::SeqCst);
            }
        });
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectedBackendKind {
    Python,
    Rust,
}

fn model_dir_has_any_pt_file(model_dir: &Path) -> bool {
    fs::read_dir(model_dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .any(|entry| entry.path().extension().is_some_and(|ext| ext == "pt"))
        })
        .unwrap_or(false)
}

pub fn detect_backend_kind(
    model_dir: &Path,
    model_name: &str,
    declared: Option<ConfiguredBackend>,
) -> Result<DetectedBackendKind, Status> {
    let has_python =
        model_dir.join("main.py").is_file() && model_dir.join("requirements.txt").is_file();
    let has_rust = model_dir.join("model_inference.textproto").is_file()
        && model_dir_has_any_pt_file(model_dir);

    // An explicit `backend` in nereid.yaml decides the kind and disambiguates a
    // folder holding files for both (e.g. a `.pt` alongside `main.py`); we only
    // verify the chosen backend's own required files are present.
    if let Some(declared) = declared {
        return match declared {
            ConfiguredBackend::Python if has_python => Ok(DetectedBackendKind::Python),
            ConfiguredBackend::Python => Err(Status::failed_precondition(format!(
                "model '{model_name}' declares backend \"python\" but is missing main.py + requirements.txt"
            ))),
            ConfiguredBackend::Rust if has_rust => Ok(DetectedBackendKind::Rust),
            ConfiguredBackend::Rust => Err(Status::failed_precondition(format!(
                "model '{model_name}' declares backend \"rust\" but is missing a .pt model + model_inference.textproto"
            ))),
        };
    }

    match (has_python, has_rust) {
        (true, true) => Err(Status::failed_precondition(format!(
            "model '{model_name}' folder contains both a Python backend (main.py + requirements.txt) and a Rust backend (a .pt model + model_inference.textproto); set `backend` in nereid.yaml to disambiguate"
        ))),
        (false, false) => Err(Status::failed_precondition(format!(
            "model '{model_name}' folder must contain either (main.py + requirements.txt) or (a .pt model + model_inference.textproto)"
        ))),
        (true, false) => Ok(DetectedBackendKind::Python),
        (false, true) => Ok(DetectedBackendKind::Rust),
    }
}

pub fn find_exactly_one_pt_model_file(model_dir: &Path) -> Result<PathBuf, Status> {
    let entries = fs::read_dir(model_dir)
        .map_err(|err| Status::internal(format!("failed to read model directory: {err}")))?;

    let mut pt_files = Vec::new();
    for entry in entries {
        let entry =
            entry.map_err(|err| Status::internal(format!("failed to read model entry: {err}")))?;
        let path = entry.path();
        if path.is_file() && path.extension().is_some_and(|ext| ext == "pt") {
            pt_files.push(path);
        }
    }

    pt_files.sort();
    match pt_files.len() {
        1 => Ok(pt_files.remove(0)),
        0 => Err(Status::failed_precondition(
            "model must contain exactly one .pt file; found none",
        )),
        count => Err(Status::failed_precondition(format!(
            "model must contain exactly one .pt file; found {count}"
        ))),
    }
}

pub fn read_input_contract_from_textproto(model_dir: &Path) -> Result<InputShapeContract, Status> {
    let config_path = model_dir.join("model_inference.textproto");
    let contents = fs::read_to_string(&config_path).map_err(|err| {
        Status::failed_precondition(format!(
            "failed to read {}: {err}",
            config_path.to_string_lossy()
        ))
    })?;

    fn parse_shape_dims(
        raw_value: &str,
        raw_line: &str,
        config_path: &Path,
    ) -> Result<Vec<i64>, Status> {
        let trimmed = raw_value.trim();
        let inner = trimmed
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .ok_or_else(|| {
                Status::failed_precondition(format!(
                    "invalid input_shape in {}: '{raw_line}'. expected bracketed dimensions such as `input_shape: [1, 16]`",
                    config_path.to_string_lossy()
                ))
            })?
            .trim();

        let dims_str: Vec<&str> = inner
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();

        if dims_str.is_empty() {
            return Err(Status::failed_precondition(format!(
                "invalid input_shape in {}: '{raw_line}'. expected bracketed dimensions such as `input_shape: [1, 16]`",
                config_path.to_string_lossy()
            )));
        }

        let mut dims = Vec::with_capacity(dims_str.len());
        for dim_str in dims_str {
            let dim = dim_str.parse::<i64>().map_err(|err| {
                Status::failed_precondition(format!(
                    "failed parsing input_shape dimension '{dim_str}' in {}: {err}",
                    config_path.to_string_lossy()
                ))
            })?;
            if dim == 0 || dim < -1 {
                return Err(Status::failed_precondition(format!(
                    "input_shape dimensions in {} must be positive or -1",
                    config_path.to_string_lossy()
                )));
            }
            dims.push(dim);
        }

        Ok(dims)
    }

    let mut input_shape = None::<Vec<i64>>;
    let mut output_shape = None::<Vec<i64>>;
    let mut data_type = None::<String>;
    let mut max_batch_size = 0i64;
    for raw_line in contents.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }

        let (field, raw_value) = line.split_once(':').ok_or_else(|| {
            Status::failed_precondition(format!(
                "invalid model_inference line in {}: '{raw_line}'",
                config_path.to_string_lossy()
            ))
        })?;

        match field.trim() {
            "input_shape" => {
                input_shape = Some(parse_shape_dims(raw_value, raw_line, &config_path)?);
            }
            "output_shape" => {
                let dims = parse_shape_dims(raw_value, raw_line, &config_path)?;
                output_shape = Some(dims);
            }
            "data_type" => {
                let value = raw_value.trim().trim_matches('"').to_string();
                if crate::dtype::kserve_fixed_width(&value).is_none() {
                    return Err(Status::failed_precondition(format!(
                        "unsupported data_type '{value}' in {}",
                        config_path.to_string_lossy()
                    )));
                }
                data_type = Some(value);
            }
            "max_batch_size" => {
                max_batch_size = raw_value.trim().parse::<i64>().map_err(|err| {
                    Status::failed_precondition(format!(
                        "failed parsing max_batch_size in {}: {err}",
                        config_path.to_string_lossy()
                    ))
                })?;
                if max_batch_size < 0 {
                    return Err(Status::failed_precondition(format!(
                        "max_batch_size in {} must be >= 0",
                        config_path.to_string_lossy()
                    )));
                }
            }
            _ => {}
        }
    }

    if input_shape.is_none() && output_shape.is_none() {
        return Err(Status::failed_precondition(format!(
            "{} must declare input_shape and/or output_shape",
            config_path.to_string_lossy()
        )));
    }

    Ok(InputShapeContract {
        input_shape,
        max_batch_size,
        output_shape,
        data_type,
    })
}

/// Whether a model's `model_inference.textproto` uses the nested
/// `input {}` / `output {}` block syntax (multi-tensor) rather than the flat
/// `input_shape` form (single-tensor).
pub fn textproto_is_multi(model_dir: &Path) -> bool {
    let path = model_dir.join("model_inference.textproto");
    let Ok(contents) = fs::read_to_string(path) else {
        return false;
    };
    contents.lines().any(|raw| {
        let line = raw.split('#').next().unwrap_or("").trim();
        line.starts_with("input {")
            || line.starts_with("input{")
            || line.starts_with("output {")
            || line.starts_with("output{")
    })
}

/// Parse a multi-tensor `model_inference.textproto` (nested `input {}` /
/// `output {}` blocks with `name`, optional `data_type` (default `FP32`), and
/// `dims`). Reuses the same `dims`/datatype validation as the single-tensor
/// parser.
pub fn read_multi_contract_from_textproto(model_dir: &Path) -> Result<MultiTensorContract, Status> {
    let config_path = model_dir.join("model_inference.textproto");
    let contents = fs::read_to_string(&config_path).map_err(|err| {
        Status::failed_precondition(format!(
            "failed to read {}: {err}",
            config_path.to_string_lossy()
        ))
    })?;

    let fail = |msg: String| {
        Status::failed_precondition(format!("{msg} in {}", config_path.to_string_lossy()))
    };

    #[derive(Default)]
    struct Pending {
        name: Option<String>,
        dtype: Option<String>,
        dims: Option<Vec<i64>>,
    }

    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut max_batch_size = 0i64;
    // None = outside a block; Some(true) = in `input {`; Some(false) = `output {`.
    let mut block: Option<bool> = None;
    let mut pending = Pending::default();

    let unquote = |s: &str| s.trim().trim_matches('"').to_string();

    for raw_line in contents.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }

        if line.starts_with("input {") || line.starts_with("input{") {
            if block.is_some() {
                return Err(fail("nested/unclosed block".to_string()));
            }
            block = Some(true);
            pending = Pending::default();
            continue;
        }
        if line.starts_with("output {") || line.starts_with("output{") {
            if block.is_some() {
                return Err(fail("nested/unclosed block".to_string()));
            }
            block = Some(false);
            pending = Pending::default();
            continue;
        }
        if line == "}" {
            let is_input = block.ok_or_else(|| fail("stray '}'".to_string()))?;
            let name = pending
                .name
                .take()
                .ok_or_else(|| fail("block missing name".to_string()))?;
            let dims = pending
                .dims
                .take()
                .ok_or_else(|| fail(format!("input/output '{name}' missing dims")))?;
            let dtype = pending.dtype.take().unwrap_or_else(|| "FP32".to_string());
            if crate::dtype::kserve_fixed_width(&dtype).is_none() {
                return Err(fail(format!(
                    "unsupported data_type '{dtype}' for '{name}'"
                )));
            }
            let spec = TensorSpec { name, dtype, dims };
            if is_input {
                inputs.push(spec);
            } else {
                outputs.push(spec);
            }
            block = None;
            continue;
        }

        let (field, value) = line
            .split_once(':')
            .ok_or_else(|| fail(format!("invalid line '{raw_line}'")))?;
        match (block, field.trim()) {
            (None, "max_batch_size") => {
                max_batch_size = value
                    .trim()
                    .parse::<i64>()
                    .map_err(|err| fail(format!("bad max_batch_size: {err}")))?;
                if max_batch_size < 0 {
                    return Err(fail("max_batch_size must be >= 0".to_string()));
                }
            }
            (Some(_), "name") => pending.name = Some(unquote(value)),
            (Some(_), "data_type") => pending.dtype = Some(unquote(value)),
            (Some(_), "dims") => {
                let inner = value
                    .trim()
                    .strip_prefix('[')
                    .and_then(|s| s.strip_suffix(']'))
                    .ok_or_else(|| fail(format!("dims must be bracketed: '{raw_line}'")))?;
                let mut dims = Vec::new();
                for d in inner.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    let dim = d
                        .parse::<i64>()
                        .map_err(|err| fail(format!("bad dim '{d}': {err}")))?;
                    if dim == 0 || dim < -1 {
                        return Err(fail("dims must be positive or -1".to_string()));
                    }
                    dims.push(dim);
                }
                if dims.is_empty() {
                    return Err(fail("dims must be non-empty".to_string()));
                }
                pending.dims = Some(dims);
            }
            _ => {}
        }
    }

    if block.is_some() {
        return Err(fail("unclosed block".to_string()));
    }
    if inputs.is_empty() || outputs.is_empty() {
        return Err(fail(
            "multi-tensor model needs at least one input and one output".to_string(),
        ));
    }

    Ok(MultiTensorContract {
        inputs,
        outputs,
        max_batch_size,
    })
}

impl InputShapeContract {
    /// Build a bare input contract (no output/datatype declaration) for a single
    /// tensor. Used by the multi-tensor path to reuse the tested batch/shape
    /// validation per named input.
    pub fn new_input(input_shape: Vec<i64>, max_batch_size: i64) -> Self {
        Self {
            input_shape: Some(input_shape),
            max_batch_size,
            output_shape: None,
            data_type: None,
        }
    }

    /// The declared input shape (excluding batch), if the model consumes a
    /// tensor.
    pub fn input_shape(&self) -> Option<&[i64]> {
        self.input_shape.as_deref()
    }

    /// Whether the model consumes a tensor input (has a declared `input_shape`).
    pub fn has_input(&self) -> bool {
        self.input_shape.is_some()
    }

    /// The declared maximum batch size (`0` when the model declares no batch
    /// dimension).
    pub fn max_batch_size(&self) -> i64 {
        self.max_batch_size
    }

    /// The declared output tensor shape (excluding the batch dimension), if the
    /// model declares one. `Some(..)` marks a Python model as tensor-capable
    /// (servable over Triton `ModelInfer`); `None` keeps it text-only.
    pub fn output_shape(&self) -> Option<&[i64]> {
        self.output_shape.as_deref()
    }

    /// The declared input KServe datatype, defaulting to `"FP32"` when the
    /// contract omits `data_type` (the original float-only behavior).
    pub fn input_datatype(&self) -> &str {
        self.data_type.as_deref().unwrap_or("FP32")
    }

    /// Whether this contract declares a batch dimension.
    fn has_batch_dim(&self) -> bool {
        self.max_batch_size > 0
    }

    /// The request rank when the (optional) batch dimension is included.
    fn batched_rank(&self, input_shape: &[i64]) -> usize {
        input_shape.len() + usize::from(self.has_batch_dim())
    }

    /// True when a batch dimension is declared but the request omits it, in
    /// which case it is auto-expanded to batch size 1 (see
    /// [`Self::normalize_request_shape`]).
    fn omits_batch_dim(&self, input_shape: &[i64], request_shape: &[i64]) -> bool {
        self.has_batch_dim() && request_shape.len() == input_shape.len()
    }

    pub fn validate_request_shape(
        &self,
        request_shape: &[i64],
        model_name: &str,
    ) -> Result<(), Status> {
        let input_shape = self.input_shape.as_deref().ok_or_else(|| {
            Status::internal(format!("model '{model_name}' declares no input tensor"))
        })?;
        if request_shape.is_empty() {
            return Err(Status::invalid_argument("tensor chunk shape is required"));
        }
        if request_shape.iter().any(|dim| *dim <= 0) {
            return Err(Status::invalid_argument(
                "tensor chunk shape dimensions must be positive",
            ));
        }

        // A request may either carry the declared batch dimension or omit it
        // entirely; an omitted batch dimension is auto-expanded to size 1.
        let shape_offset = if request_shape.len() == self.batched_rank(input_shape) {
            if self.has_batch_dim() {
                let batch_size = request_shape[0];
                if batch_size > self.max_batch_size {
                    return Err(Status::invalid_argument(format!(
                        "batch size {batch_size} exceeds max_batch_size {} for model '{model_name}'",
                        self.max_batch_size
                    )));
                }
                1
            } else {
                0
            }
        } else if self.omits_batch_dim(input_shape, request_shape) {
            0
        } else {
            let allowed = if self.has_batch_dim() {
                format!(
                    "expected {} (or {} without the batch dimension)",
                    self.batched_rank(input_shape),
                    input_shape.len()
                )
            } else {
                format!("expected {}", self.batched_rank(input_shape))
            };
            return Err(Status::invalid_argument(format!(
                "input tensor rank mismatch for model '{model_name}': {allowed}, got {}",
                request_shape.len()
            )));
        };

        for (index, expected_dim) in input_shape.iter().enumerate() {
            let actual_dim = request_shape[index + shape_offset];
            if *expected_dim != -1 && *expected_dim != actual_dim {
                return Err(Status::invalid_argument(format!(
                    "input tensor shape mismatch for model '{model_name}' at dimension {}: expected {}, got {}",
                    index + shape_offset,
                    expected_dim,
                    actual_dim
                )));
            }
        }

        Ok(())
    }

    /// Validate a model's *output* tensor shape against the declared
    /// `output_shape`. Mirrors the input check: a declared batch dimension may
    /// be present (any positive size) or omitted, and `-1` dims match anything.
    /// When the output carries a batch dimension and `expected_batch` is
    /// `Some`, the batch size must match the request's — a model that returns a
    /// different batch count than it was given is rejected. A model that
    /// declares no `output_shape` has nothing to check (returns `Ok`). Returns
    /// `Status::internal` on mismatch — a wrong-shaped output is a model/config
    /// bug, not a client error.
    pub fn validate_output_shape(
        &self,
        actual: &[i64],
        expected_batch: Option<i64>,
        model_name: &str,
    ) -> Result<(), Status> {
        let Some(declared) = self.output_shape.as_deref() else {
            return Ok(());
        };

        // The model may or may not carry a leading batch dimension.
        let offset = if self.has_batch_dim() && actual.len() == declared.len() + 1 {
            1
        } else if actual.len() == declared.len() {
            0
        } else {
            let allowed = if self.has_batch_dim() {
                format!(
                    "{} (or {} with a batch dimension)",
                    declared.len(),
                    declared.len() + 1
                )
            } else {
                declared.len().to_string()
            };
            return Err(Status::internal(format!(
                "model '{model_name}' output rank mismatch: expected {allowed}, got {}",
                actual.len()
            )));
        };

        if offset == 1
            && let Some(expected) = expected_batch
            && actual[0] != expected
        {
            return Err(Status::internal(format!(
                "model '{model_name}' output batch size {} does not match input batch size {expected}",
                actual[0]
            )));
        }

        for (index, expected_dim) in declared.iter().enumerate() {
            let actual_dim = actual[index + offset];
            if *expected_dim != -1 && *expected_dim != actual_dim {
                return Err(Status::internal(format!(
                    "model '{model_name}' output shape mismatch at dimension {}: declared {}, got {}",
                    index + offset,
                    expected_dim,
                    actual_dim
                )));
            }
        }

        Ok(())
    }

    /// Return the effective tensor shape to feed the model, inserting a leading
    /// batch dimension of 1 when the request omitted the declared batch
    /// dimension. Expects `request_shape` to have already passed
    /// [`Self::validate_request_shape`].
    pub fn normalize_request_shape(&self, request_shape: Vec<i64>) -> Vec<i64> {
        match self.input_shape.as_deref() {
            Some(input_shape) if self.omits_batch_dim(input_shape, &request_shape) => {
                let mut expanded = Vec::with_capacity(request_shape.len() + 1);
                expanded.push(1);
                expanded.extend(request_shape);
                expanded
            }
            _ => request_shape,
        }
    }
}

/// Build a tensor of `kind` from raw row-major little-endian `tensor_bytes` and
/// `request_shape`. The buffer length must be exactly `numel × element_size`.
pub fn tensor_from_input_bytes(
    tensor_bytes: &[u8],
    request_shape: &[i64],
    model_name: &str,
    kind: tch::Kind,
) -> Result<Tensor, Status> {
    if tensor_bytes.is_empty() {
        return Err(Status::invalid_argument(
            "no tensor data provided for Rust inference model",
        ));
    }
    let elt = kind.elt_size_in_bytes();
    if !tensor_bytes.len().is_multiple_of(elt) {
        return Err(Status::invalid_argument(format!(
            "tensor bytes length {} is not a multiple of {elt} for {kind:?}",
            tensor_bytes.len()
        )));
    }

    let expected_numel = request_shape
        .iter()
        .try_fold(1i64, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| Status::invalid_argument("tensor shape overflow"))?;
    let actual_numel = (tensor_bytes.len() / elt) as i64;
    if expected_numel != actual_numel {
        return Err(Status::invalid_argument(format!(
            "input tensor size mismatch for model '{model_name}': expected {expected_numel} elements from request shape, got {actual_numel}"
        )));
    }

    Tensor::f_from_data_size(tensor_bytes, request_shape, kind)
        .map_err(|err| Status::invalid_argument(format!("failed to build input tensor: {err}")))
}

#[cfg(test)]
mod tests {
    use super::{
        DetectedBackendKind, InputShapeContract, detect_backend_kind,
        find_exactly_one_pt_model_file, read_input_contract_from_textproto,
    };
    use crate::config::load_server_config;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("nereid-test-{prefix}-{nanos}"));
        fs::create_dir_all(&path).expect("create temp dir");
        path
    }

    #[test]
    fn find_exactly_one_pt_file_rejects_missing_and_multiple() {
        let base = temp_dir("pt-file-check");

        let missing = base.join("missing-pt");
        fs::create_dir_all(&missing).expect("create missing dir");
        assert!(find_exactly_one_pt_model_file(&missing).is_err());

        let multiple = base.join("multi-pt");
        fs::create_dir_all(&multiple).expect("create multi dir");
        fs::write(multiple.join("a.pt"), b"x").expect("write a.pt");
        fs::write(multiple.join("b.pt"), b"x").expect("write b.pt");
        assert!(find_exactly_one_pt_model_file(&multiple).is_err());

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn model3_has_single_pt_file_in_fixture() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let config = load_server_config(&manifest_dir.join("nereid.yaml.example"))
            .expect("example config should load");
        let model3 = manifest_dir
            .join(config.server.ml_backends_path)
            .join("model3");
        if model3.is_dir() {
            find_exactly_one_pt_model_file(&model3).expect("model3 should have one .pt file");
        }
    }

    #[test]
    fn input_contract_allows_variable_dims_and_max_batch_size() {
        let base = temp_dir("input-contract");
        fs::write(
            base.join("model_inference.textproto"),
            b"input_shape: [-1, 16]\nmax_batch_size: 8\n",
        )
        .expect("write textproto");

        let contract = read_input_contract_from_textproto(&base).expect("parse contract");
        assert_eq!(
            contract,
            InputShapeContract {
                input_shape: Some(vec![-1, 16]),
                max_batch_size: 8,
                output_shape: None,
                data_type: None,
            }
        );

        contract
            .validate_request_shape(&[4, 10, 16], "model")
            .expect("shape should match");
        assert!(
            contract
                .validate_request_shape(&[9, 10, 16], "model")
                .is_err()
        );
        assert!(
            contract
                .validate_request_shape(&[4, 10, 15], "model")
                .is_err()
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn multi_contract_parses_nested_blocks() {
        use super::{read_multi_contract_from_textproto, textproto_is_multi};
        let base = temp_dir("multi-contract");
        fs::write(
            base.join("model_inference.textproto"),
            b"max_batch_size: 4\n\
              input {\n  name: \"a\"\n  data_type: \"FP32\"\n  dims: [4]\n}\n\
              input {\n  name: \"b\"\n  dims: [4]\n}\n\
              output {\n  name: \"sum\"\n  data_type: \"FP32\"\n  dims: [4]\n}\n",
        )
        .expect("write textproto");

        assert!(textproto_is_multi(&base), "nested blocks -> multi");
        let contract = read_multi_contract_from_textproto(&base).expect("parse multi contract");
        assert_eq!(contract.max_batch_size, 4);
        assert_eq!(contract.inputs.len(), 2);
        assert_eq!(contract.inputs[0].name, "a");
        assert_eq!(contract.inputs[0].dtype, "FP32");
        assert_eq!(contract.inputs[0].dims, vec![4]);
        // data_type defaults to FP32 when omitted.
        assert_eq!(contract.inputs[1].name, "b");
        assert_eq!(contract.inputs[1].dtype, "FP32");
        assert_eq!(contract.outputs.len(), 1);
        assert_eq!(contract.outputs[0].name, "sum");

        // A flat single-tensor textproto is NOT detected as multi.
        let flat = temp_dir("flat-contract");
        fs::write(
            flat.join("model_inference.textproto"),
            b"input_shape: [16]\nmax_batch_size: 8\n",
        )
        .expect("write flat");
        assert!(!textproto_is_multi(&flat), "flat form -> not multi");

        let _ = fs::remove_dir_all(&base);
        let _ = fs::remove_dir_all(&flat);
    }

    #[test]
    fn multi_contract_rejects_block_missing_dims() {
        use super::read_multi_contract_from_textproto;
        let base = temp_dir("multi-bad");
        fs::write(
            base.join("model_inference.textproto"),
            b"input {\n  name: \"a\"\n}\noutput {\n  name: \"y\"\n  dims: [4]\n}\n",
        )
        .expect("write");
        assert!(
            read_multi_contract_from_textproto(&base).is_err(),
            "input without dims must be rejected"
        );
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn contract_parses_output_shape_when_present() {
        let base = temp_dir("output-shape");
        fs::write(
            base.join("model_inference.textproto"),
            b"input_shape: [4]\nmax_batch_size: 4\noutput_shape: [4]\n",
        )
        .expect("write textproto");

        let contract = read_input_contract_from_textproto(&base).expect("parse contract");
        assert_eq!(
            contract.output_shape(),
            Some([4i64].as_slice()),
            "output_shape should be parsed as the tensor-output signal"
        );

        // A contract with no output_shape line reports None (text-only Python).
        fs::write(
            base.join("model_inference.textproto"),
            b"input_shape: [4]\nmax_batch_size: 4\n",
        )
        .expect("rewrite textproto");
        let no_output = read_input_contract_from_textproto(&base).expect("parse contract");
        assert_eq!(no_output.output_shape(), None);

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn validate_output_shape_accepts_declared_and_rejects_mismatch() {
        let contract = InputShapeContract {
            input_shape: Some(vec![4]),
            max_batch_size: 4,
            output_shape: Some(vec![4]),
            data_type: None,
        };

        // With or without the optional batch dimension.
        contract
            .validate_output_shape(&[1, 4], Some(1), "m")
            .expect("batched output ok");
        contract
            .validate_output_shape(&[4], None, "m")
            .expect("unbatched output ok");

        // Wrong trailing dim and wrong rank are both rejected.
        assert!(
            contract
                .validate_output_shape(&[1, 5], Some(1), "m")
                .is_err()
        );
        assert!(
            contract
                .validate_output_shape(&[1, 1, 4], Some(1), "m")
                .is_err()
        );

        // Output batch that disagrees with the input batch is rejected.
        assert!(
            contract
                .validate_output_shape(&[3, 4], Some(2), "m")
                .is_err(),
            "output batch 3 must not pass for input batch 2"
        );
        contract
            .validate_output_shape(&[2, 4], Some(2), "m")
            .expect("matching batch ok");

        // A -1 declared dim matches any positive size.
        let variable = InputShapeContract {
            input_shape: Some(vec![4]),
            max_batch_size: 0,
            output_shape: Some(vec![-1]),
            data_type: None,
        };
        variable
            .validate_output_shape(&[7], None, "m")
            .expect("variable output dim ok");

        // No declared output_shape -> nothing to check.
        let none = InputShapeContract {
            input_shape: Some(vec![4]),
            max_batch_size: 0,
            output_shape: None,
            data_type: None,
        };
        none.validate_output_shape(&[9, 9, 9], None, "m")
            .expect("undeclared output is unchecked");
    }

    #[test]
    fn input_contract_without_batch_uses_request_shape_directly() {
        let contract = InputShapeContract {
            input_shape: Some(vec![-1, 16]),
            max_batch_size: 0,
            output_shape: None,
            data_type: None,
        };

        contract
            .validate_request_shape(&[10, 16], "model")
            .expect("shape should match");
        assert!(
            contract
                .validate_request_shape(&[1, 10, 16], "model")
                .is_err()
        );
    }

    #[test]
    fn input_contract_auto_expands_missing_batch_dim() {
        let contract = InputShapeContract {
            input_shape: Some(vec![16]),
            max_batch_size: 10,
            output_shape: None,
            data_type: None,
        };

        // Bare shape (batch omitted) and explicit batch shape both validate.
        contract
            .validate_request_shape(&[16], "model")
            .expect("bare shape should be accepted");
        contract
            .validate_request_shape(&[2, 16], "model")
            .expect("explicit batch shape should be accepted");

        // The bare shape is expanded to a leading batch dimension of 1, while an
        // explicit batch shape is passed through untouched.
        assert_eq!(contract.normalize_request_shape(vec![16]), vec![1, 16]);
        assert_eq!(contract.normalize_request_shape(vec![2, 16]), vec![2, 16]);

        // A genuinely wrong rank is still rejected.
        assert!(
            contract
                .validate_request_shape(&[1, 1, 16], "model")
                .is_err()
        );
    }

    #[test]
    fn detect_backend_kind_recognizes_python_folder() {
        let base = temp_dir("kind-python");
        fs::write(base.join("main.py"), b"print('hi')").expect("write main.py");
        fs::write(base.join("requirements.txt"), b"numpy").expect("write requirements.txt");

        assert_eq!(
            detect_backend_kind(&base, "model", None).expect("should detect python"),
            DetectedBackendKind::Python
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn detect_backend_kind_recognizes_rust_folder() {
        let base = temp_dir("kind-rust");
        fs::write(
            base.join("model_inference.textproto"),
            b"input_shape: [16]\n",
        )
        .expect("write textproto");
        fs::write(base.join("model.pt"), b"x").expect("write model.pt");

        assert_eq!(
            detect_backend_kind(&base, "model", None).expect("should detect rust"),
            DetectedBackendKind::Rust
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn detect_backend_kind_rejects_ambiguous_folder() {
        let base = temp_dir("kind-ambiguous");
        fs::write(base.join("main.py"), b"print('hi')").expect("write main.py");
        fs::write(base.join("requirements.txt"), b"numpy").expect("write requirements.txt");
        fs::write(
            base.join("model_inference.textproto"),
            b"input_shape: [16]\n",
        )
        .expect("write textproto");
        fs::write(base.join("model.pt"), b"x").expect("write model.pt");

        assert!(detect_backend_kind(&base, "model", None).is_err());

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn detect_backend_kind_rejects_empty_folder() {
        let base = temp_dir("kind-empty");

        assert!(detect_backend_kind(&base, "model", None).is_err());

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn input_contract_allows_multiple_variable_dims() {
        let contract = InputShapeContract {
            input_shape: Some(vec![-1, -1, 16]),
            max_batch_size: 4,
            output_shape: None,
            data_type: None,
        };

        contract
            .validate_request_shape(&[2, 5, 7, 16], "model")
            .expect("shape should match");
        assert!(
            contract
                .validate_request_shape(&[2, 5, 7, 15], "model")
                .is_err()
        );
    }
}
