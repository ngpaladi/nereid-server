//! TorchScript (`.pt`) backend, via libtorch (`tch`). All libtorch-specific code
//! — device resolution, the `tch::Kind` dtype mapping, tensor (de)serialization,
//! and the forward pass — lives here rather than in the core server.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use tch::{CModule, Cuda, Device, IValue, Kind, Tensor};
use tonic::Status;

use super::Tensor as NTensor;
use super::{Backend, Contract};
use crate::config::{ModelConfig, ModelDevice};

pub struct TorchBackend {
    // libtorch's `CModule` isn't `Sync`; the `Mutex` serializes inference (as the
    // per-model worker thread did before) and makes the backend shareable.
    module: Mutex<CModule>,
    device: Device,
    output_names: Vec<String>,
}

impl TorchBackend {
    pub fn load(
        model_dir: &Path,
        model_cfg: &ModelConfig,
    ) -> Result<(Box<dyn Backend>, Contract), Status> {
        let pt_file = find_exactly_one_pt_model_file(model_dir)?;
        let device = to_tch_device(model_cfg.device)?;
        let module = CModule::load_on_device(&pt_file, device).map_err(|err| {
            Status::failed_precondition(format!(
                "failed to load .pt model for '{}': {err}",
                model_cfg.name
            ))
        })?;

        let contract = Contract::parse(model_dir)?;
        if !contract.has_input() {
            return Err(Status::failed_precondition(format!(
                "Torch model '{}' must declare input_shape in model_inference.textproto",
                model_cfg.name
            )));
        }
        let output_names = contract.outputs.iter().map(|s| s.name.clone()).collect();

        Ok((
            Box::new(TorchBackend {
                module: Mutex::new(module),
                device,
                output_names,
            }),
            contract,
        ))
    }
}

impl Backend for TorchBackend {
    fn platform(&self) -> &'static str {
        "pytorch_libtorch"
    }

    fn infer(&self, inputs: Vec<NTensor>) -> Result<Vec<NTensor>, Status> {
        // Build libtorch tensors of the declared kinds (uint16/32/64 have no
        // libtorch kind and are rejected here).
        let mut tch_inputs = Vec::with_capacity(inputs.len());
        for t in &inputs {
            let kind = kind_from_canonical(&t.dtype).ok_or_else(|| {
                Status::invalid_argument(format!(
                    "Torch backend does not support datatype '{}' (no libtorch kind)",
                    t.dtype
                ))
            })?;
            tch_inputs.push(tensor_from_input_bytes(
                &t.data,
                &t.shape,
                "torch model",
                kind,
            )?);
        }

        let outputs = {
            let module = self.module.lock().unwrap_or_else(|e| e.into_inner());
            run_forward_pass(&module, self.device, tch_inputs)
                .map_err(|err| Status::internal(format!("Torch inference failed: {err}")))?
        };

        Ok(outputs
            .into_iter()
            .enumerate()
            .map(|(i, (shape, data, dtype))| NTensor {
                name: self
                    .output_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| "output".to_string()),
                shape,
                dtype,
                data,
            })
            .collect())
    }
}

fn find_exactly_one_pt_model_file(model_dir: &Path) -> Result<PathBuf, Status> {
    let entries = std::fs::read_dir(model_dir)
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

// ---------------------------------------------------------------------------
// Device resolution (moved out of the core config module).
// ---------------------------------------------------------------------------
fn to_tch_device(device: ModelDevice) -> Result<Device, Status> {
    match device {
        ModelDevice::Cpu => Ok(Device::Cpu),
        ModelDevice::Cuda(index) => {
            preload_libtorch_cuda();
            if !Cuda::is_available() {
                return Err(Status::failed_precondition(
                    "CUDA model configured but CUDA is not available",
                ));
            }
            let device_count = Cuda::device_count();
            if i64::try_from(index).unwrap_or(i64::MAX) >= device_count {
                return Err(Status::failed_precondition(format!(
                    "cuda device index {index} is out of range; {device_count} CUDA device(s) available"
                )));
            }
            Ok(Device::Cuda(index))
        }
    }
}

fn preload_libtorch_cuda() {
    #[cfg(target_os = "linux")]
    {
        for library in ["libc10_cuda.so", "libtorch_cuda.so"] {
            if let Err(err) = dlopen_library(library) {
                eprintln!("failed to preload {library}: {err}");
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn dlopen_library(library: &str) -> Result<(), String> {
    use std::ffi::CString;
    use std::os::raw::c_int;
    const RTLD_NOW: c_int = 2;
    const RTLD_GLOBAL: c_int = 0x100;
    let path = CString::new(library).map_err(|err| err.to_string())?;
    let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
    if handle.is_null() {
        Err(dlopen_error())
    } else {
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn dlopen_error() -> String {
    let error = unsafe { dlerror() };
    if error.is_null() {
        "unknown dlopen error".to_owned()
    } else {
        unsafe { std::ffi::CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(target_os = "linux")]
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(
        filename: *const std::os::raw::c_char,
        flag: std::os::raw::c_int,
    ) -> *mut std::os::raw::c_void;
    fn dlerror() -> *const std::os::raw::c_char;
}

// ---------------------------------------------------------------------------
// Dtype <-> libtorch kind (moved out of the core dtype module).
// ---------------------------------------------------------------------------
fn kind_from_canonical(canonical: &str) -> Option<Kind> {
    use Kind::*;
    Some(match canonical {
        "bool" => Bool,
        "uint8" => Uint8,
        "int8" => Int8,
        "int16" => Int16,
        "int32" => Int,
        "int64" => Int64,
        "float16" => Half,
        "float32" => Float,
        "float64" => Double,
        "bfloat16" => BFloat16,
        _ => return None,
    })
}

fn canonical_from_kind(kind: Kind) -> Option<&'static str> {
    use Kind::*;
    Some(match kind {
        Bool => "bool",
        Uint8 => "uint8",
        Int8 => "int8",
        Int16 => "int16",
        Int => "int32",
        Int64 => "int64",
        Half => "float16",
        Float => "float32",
        Double => "float64",
        BFloat16 => "bfloat16",
        _ => return None,
    })
}

/// Build a tensor of `kind` from raw row-major little-endian bytes. The buffer
/// length must be exactly `numel × element_size`.
fn tensor_from_input_bytes(
    tensor_bytes: &[u8],
    request_shape: &[i64],
    model_name: &str,
    kind: Kind,
) -> Result<Tensor, Status> {
    let elt_size = kind.elt_size_in_bytes();
    if elt_size == 0 || !tensor_bytes.len().is_multiple_of(elt_size) {
        return Err(Status::invalid_argument(format!(
            "input byte length {} is not a multiple of element size {elt_size} for model '{model_name}'",
            tensor_bytes.len()
        )));
    }
    Tensor::f_from_data_size(tensor_bytes, request_shape, kind).map_err(|err| {
        Status::internal(format!(
            "failed to build input tensor for model '{model_name}': {err}"
        ))
    })
}

type TensorOutput = (Vec<i64>, Vec<u8>, String);

fn tensor_to_bytes(tensor: &Tensor) -> Result<TensorOutput, tch::TchError> {
    let tensor = tensor.to_device(Device::Cpu).contiguous();
    let shape = tensor.size();
    let kind = tensor.kind();
    let dtype = canonical_from_kind(kind)
        .ok_or_else(|| tch::TchError::Torch(format!("unsupported output tensor kind {kind:?}")))?
        .to_string();
    let flat = tensor.reshape([-1]);
    let numel = flat.numel();
    let mut bytes = vec![0u8; numel * kind.elt_size_in_bytes()];
    flat.copy_data_u8(&mut bytes, numel);
    Ok((shape, bytes, dtype))
}

/// Run the model's forward pass over `inputs` (positionally). The model may
/// return a single tensor or a tuple; each output is serialized to bytes.
fn run_forward_pass(
    model: &CModule,
    device: Device,
    inputs: Vec<Tensor>,
) -> Result<Vec<TensorOutput>, tch::TchError> {
    let ivalues: Vec<IValue> = inputs
        .into_iter()
        .map(|t| IValue::Tensor(t.to_device(device)))
        .collect();
    match model.forward_is(&ivalues)? {
        IValue::Tensor(tensor) => Ok(vec![tensor_to_bytes(&tensor)?]),
        IValue::Tuple(items) => {
            let mut outputs = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    IValue::Tensor(tensor) => outputs.push(tensor_to_bytes(&tensor)?),
                    other => {
                        return Err(tch::TchError::Torch(format!(
                            "model tuple output element is non-tensor: {other:?}"
                        )));
                    }
                }
            }
            Ok(outputs)
        }
        other => Err(tch::TchError::Torch(format!(
            "model output was neither tensor nor tuple: {other:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::find_exactly_one_pt_model_file;
    use crate::config::load_server_config;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("nereid-torch-{prefix}-{nanos}"));
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
}
