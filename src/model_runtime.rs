use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc as std_mpsc;

use tch::{CModule, Device, Tensor};
use tokio::sync::oneshot;
use tonic::Status;

use crate::config::ServerConfig;
use crate::inference;

pub type InferenceOutput = (Vec<i64>, Vec<u8>);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputShapeContract {
    input_shape: Vec<i64>,
    max_batch_size: i64,
}

#[derive(Debug)]
struct ModelHandle {
    input_contract: InputShapeContract,
    job_tx: std_mpsc::SyncSender<InferenceJob>,
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

            let pt_file = find_exactly_one_pt_model_file(&model_dir)?;
            let input_contract = read_input_contract_from_textproto(&model_dir)?;
            let device = model_cfg.device.to_tch_device()?;
            let model_path = pt_file.to_string_lossy().into_owned();

            let model = CModule::load_on_device(&model_path, device).map_err(|err| {
                Status::failed_precondition(format!(
                    "failed to load .pt model for '{}': {err}",
                    model_cfg.name
                ))
            })?;

            let (job_tx, job_rx) = std_mpsc::sync_channel::<InferenceJob>(model_cfg.queue_capacity);
            let occupancy = Arc::new(AtomicUsize::new(0));
            spawn_model_worker(model_cfg.name.clone(), model, device, job_rx);

            model_names.push(model_cfg.name.clone());
            handles.insert(
                model_cfg.name.clone(),
                ModelHandle {
                    input_contract,
                    job_tx,
                    queue_capacity: model_cfg.queue_capacity,
                    occupancy,
                },
            );
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
        self.handles
            .get(model_name)
            .map(|handle| &handle.input_contract)
    }

    pub fn enqueue(
        &self,
        model_name: &str,
        input_tensor: Tensor,
    ) -> Result<oneshot::Receiver<Result<InferenceOutput, Status>>, Status> {
        let handle = self.handles.get(model_name).ok_or_else(|| {
            Status::not_found(format!(
                "model '{model_name}' is not configured in nereid.yaml"
            ))
        })?;

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

    let mut shape = Vec::new();
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
                let dims = parse_shape_dims(raw_value, raw_line, &config_path)?;
                shape.extend(dims);
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

    if shape.is_empty() {
        return Err(Status::failed_precondition(format!(
            "{} must contain at least one input_shape field",
            config_path.to_string_lossy()
        )));
    }

    Ok(InputShapeContract {
        input_shape: shape,
        max_batch_size,
    })
}

impl InputShapeContract {
    pub fn validate_request_shape(
        &self,
        request_shape: &[i64],
        model_name: &str,
    ) -> Result<(), Status> {
        if request_shape.is_empty() {
            return Err(Status::invalid_argument("tensor chunk shape is required"));
        }
        if request_shape.iter().any(|dim| *dim <= 0) {
            return Err(Status::invalid_argument(
                "tensor chunk shape dimensions must be positive",
            ));
        }

        let expected_rank = self.input_shape.len() + usize::from(self.max_batch_size > 0);
        if request_shape.len() != expected_rank {
            return Err(Status::invalid_argument(format!(
                "input tensor rank mismatch for model '{model_name}': expected {expected_rank}, got {}",
                request_shape.len()
            )));
        }

        let shape_offset = if self.max_batch_size > 0 {
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
        };

        for (index, expected_dim) in self.input_shape.iter().enumerate() {
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
    if tensor_bytes.len() % 4 != 0 {
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
        InputShapeContract, find_exactly_one_pt_model_file, read_input_contract_from_textproto,
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
                input_shape: vec![-1, 16],
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
            input_shape: vec![-1, 16],
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
    fn input_contract_allows_multiple_variable_dims() {
        let contract = InputShapeContract {
            input_shape: vec![-1, -1, 16],
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
