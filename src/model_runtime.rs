use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc as std_mpsc;

use tch::{CModule, Device, Tensor};
use tokio::sync::oneshot;
use tonic::Status;

use crate::config::{ConfiguredBackend, ServerConfig};
use crate::inference;
use crate::python_backend;

pub type InferenceOutput = (Vec<i64>, Vec<u8>);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputShapeContract {
    /// Declared input tensor shape (excluding batch). `None` for a model that
    /// consumes no tensor input (e.g. an output-only Python producer).
    input_shape: Option<Vec<i64>>,
    /// Declared output tensor shape (excluding batch). Required for Python
    /// models — every Python reply is a typed tensor written to
    /// `NEREID_OUTPUT_PATH` and validated/streamed against this.
    output_shape: Option<Vec<i64>>,
    max_batch_size: i64,
}

#[derive(Debug)]
enum ModelHandle {
    Rust(RustModelHandle),
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
}

#[derive(Debug)]
struct InferenceJob {
    input_tensor: Tensor,
    response_tx: oneshot::Sender<Result<InferenceOutput, Status>>,
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
                    // output tensor. An `input_shape` is optional (a model may
                    // consume no tensor input).
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
                        }),
                    );
                }
                DetectedBackendKind::Rust => {
                    let pt_file = find_exactly_one_pt_model_file(&model_dir)?;
                    let input_contract = read_input_contract_from_textproto(&model_dir)?;
                    if !input_contract.has_input() {
                        return Err(Status::failed_precondition(format!(
                            "Rust model '{}' must declare input_shape in model_inference.textproto",
                            model_cfg.name
                        )));
                    }
                    let device = model_cfg.device.to_tch_device()?;
                    let model_path = pt_file.to_string_lossy().into_owned();

                    let model = CModule::load_on_device(&model_path, device).map_err(|err| {
                        Status::failed_precondition(format!(
                            "failed to load .pt model for '{}': {err}",
                            model_cfg.name
                        ))
                    })?;

                    let (job_tx, job_rx) =
                        std_mpsc::sync_channel::<InferenceJob>(model_cfg.queue_capacity);
                    let occupancy = Arc::new(AtomicUsize::new(0));
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

        Ok(Self {
            model_names,
            handles,
        })
    }

    pub fn configured_models(&self) -> Vec<String> {
        self.model_names.clone()
    }

    pub fn input_contract(&self, model_name: &str) -> Option<&InputShapeContract> {
        match self.handles.get(model_name)? {
            ModelHandle::Rust(handle) => Some(&handle.input_contract),
            ModelHandle::Python(handle) => Some(&handle.contract),
        }
    }

    pub fn python_model_dir(&self, model_name: &str) -> Option<PathBuf> {
        match self.handles.get(model_name)? {
            ModelHandle::Python(handle) => Some(handle.model_dir.clone()),
            ModelHandle::Rust(_) => None,
        }
    }

    pub fn enqueue(
        &self,
        model_name: &str,
        input_tensor: Tensor,
    ) -> Result<oneshot::Receiver<Result<InferenceOutput, Status>>, Status> {
        let handle = match self.handles.get(model_name) {
            Some(ModelHandle::Rust(handle)) => handle,
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
        field: &str,
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
                    "invalid {field} in {}: '{raw_line}'. expected bracketed dimensions such as `{field}: [1, 16]`",
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
                "invalid {field} in {}: '{raw_line}'. expected bracketed dimensions such as `{field}: [1, 16]`",
                config_path.to_string_lossy()
            )));
        }

        let mut dims = Vec::with_capacity(dims_str.len());
        for dim_str in dims_str {
            let dim = dim_str.parse::<i64>().map_err(|err| {
                Status::failed_precondition(format!(
                    "failed parsing {field} dimension '{dim_str}' in {}: {err}",
                    config_path.to_string_lossy()
                ))
            })?;
            if dim == 0 || dim < -1 {
                return Err(Status::failed_precondition(format!(
                    "{field} dimensions in {} must be positive or -1",
                    config_path.to_string_lossy()
                )));
            }
            dims.push(dim);
        }

        Ok(dims)
    }

    let mut input_shape = None::<Vec<i64>>;
    let mut output_shape = None::<Vec<i64>>;
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
                input_shape = Some(parse_shape_dims(
                    "input_shape",
                    raw_value,
                    raw_line,
                    &config_path,
                )?);
            }
            "output_shape" => {
                output_shape = Some(parse_shape_dims(
                    "output_shape",
                    raw_value,
                    raw_line,
                    &config_path,
                )?);
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
        output_shape,
        max_batch_size,
    })
}

impl InputShapeContract {
    /// The declared output shape (excluding batch), if declared.
    pub fn output_shape(&self) -> Option<&[i64]> {
        self.output_shape.as_deref()
    }

    /// Whether the model consumes a tensor input (has a declared `input_shape`).
    pub fn has_input(&self) -> bool {
        self.input_shape.is_some()
    }

    /// The declared maximum batch size (`0` when no batch dimension is declared).
    pub fn max_batch_size(&self) -> i64 {
        self.max_batch_size
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

    /// Validate a Python model's *output* tensor shape (as written to
    /// `NEREID_OUTPUT_PATH`) against the declared `output_shape`. A declared
    /// batch dimension may be present or omitted; when present and
    /// `expected_batch` is `Some`, it must match the request's batch. `-1`
    /// declared dims match anything. Returns `Status::internal` on mismatch — a
    /// wrong-shaped reply is a model bug.
    pub fn validate_output_shape(
        &self,
        actual: &[i64],
        expected_batch: Option<i64>,
        model_name: &str,
    ) -> Result<(), Status> {
        let Some(declared) = self.output_shape.as_deref() else {
            return Ok(());
        };

        let offset = if self.has_batch_dim() && actual.len() == declared.len() + 1 {
            1
        } else if actual.len() == declared.len() {
            0
        } else {
            return Err(Status::internal(format!(
                "model '{model_name}' output rank mismatch: declared {} dims, got {}",
                declared.len(),
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

pub fn tensor_from_input_bytes(
    tensor_bytes: &[u8],
    request_shape: &[i64],
    model_name: &str,
) -> Result<Tensor, Status> {
    if tensor_bytes.is_empty() {
        return Err(Status::invalid_argument(
            "no tensor chunk data provided for Rust inference model",
        ));
    }
    if !tensor_bytes.len().is_multiple_of(4) {
        return Err(Status::invalid_argument(
            "tensor chunk bytes must be a multiple of 4 for float32",
        ));
    }

    let values: Vec<f32> = tensor_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let expected_numel = request_shape
        .iter()
        .try_fold(1i64, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| Status::invalid_argument("tensor shape overflow"))?;

    if expected_numel != i64::try_from(values.len()).unwrap_or(-1) {
        return Err(Status::invalid_argument(format!(
            "input tensor size mismatch for model '{model_name}': expected {expected_numel} values from request shape, got {}",
            values.len()
        )));
    }

    Tensor::f_from_slice(&values)
        .and_then(|t| t.f_reshape(request_shape))
        .map_err(|err| {
            Status::invalid_argument(format!(
                "failed to build input tensor from stream chunks: {err}"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::{
        DetectedBackendKind, InputShapeContract, detect_backend_kind,
        find_exactly_one_pt_model_file, read_input_contract_from_textproto,
    };
    use crate::config::{ConfiguredBackend, load_server_config};
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
                output_shape: None,
                max_batch_size: 8
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
    fn input_contract_without_batch_uses_request_shape_directly() {
        let contract = InputShapeContract {
            input_shape: Some(vec![-1, 16]),
            output_shape: None,
            max_batch_size: 0,
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
            output_shape: None,
            max_batch_size: 10,
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

        // Auto-detection (no declared backend) is ambiguous and rejected...
        assert!(detect_backend_kind(&base, "model", None).is_err());
        // ...but an explicit backend disambiguates, allowing a `.pt` to live
        // alongside main.py (and vice versa).
        assert_eq!(
            detect_backend_kind(&base, "model", Some(ConfiguredBackend::Python))
                .expect("declared python resolves"),
            DetectedBackendKind::Python
        );
        assert_eq!(
            detect_backend_kind(&base, "model", Some(ConfiguredBackend::Rust))
                .expect("declared rust resolves"),
            DetectedBackendKind::Rust
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn detect_backend_kind_declared_backend_requires_its_files() {
        // A folder with only Python files but declaring backend "rust" fails.
        let base = temp_dir("kind-declared-mismatch");
        fs::write(base.join("main.py"), b"print('hi')").expect("write main.py");
        fs::write(base.join("requirements.txt"), b"numpy").expect("write requirements.txt");
        assert!(
            detect_backend_kind(&base, "model", Some(ConfiguredBackend::Rust)).is_err(),
            "declared rust without a .pt/textproto must fail"
        );
        assert_eq!(
            detect_backend_kind(&base, "model", Some(ConfiguredBackend::Python))
                .expect("declared python matches files"),
            DetectedBackendKind::Python
        );
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
            output_shape: None,
            max_batch_size: 4,
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
