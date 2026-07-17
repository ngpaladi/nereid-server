//! ONNX backend, via ONNX Runtime (`ort`). In-process; the CUDA execution
//! provider is selected when the model's device is `cuda`.

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ort::execution_providers::{CPUExecutionProvider, CUDAExecutionProvider};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{DynValue, TensorElementType, Value};
use tonic::Status;

use super::Tensor;
use super::native_common::{bytes_to_vec, dispatch_dtype, slice_to_bytes};
use super::{Backend, Contract};
use crate::config::ModelConfig;

pub struct OnnxBackend {
    // ort's `Session::run` takes `&mut self`; the `Mutex` serializes inference
    // and makes the backend shareable across the async runtime.
    session: Mutex<Session>,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl OnnxBackend {
    pub fn load(
        model_dir: &Path,
        model_cfg: &ModelConfig,
    ) -> Result<(Box<dyn Backend>, Contract), Status> {
        let onnx_file = find_exactly_one_onnx_file(model_dir)?;
        let contract = Contract::parse(model_dir)?;
        if !contract.has_input() {
            return Err(Status::failed_precondition(format!(
                "ONNX model '{}' must declare input_shape in model_inference.textproto",
                model_cfg.name
            )));
        }

        let mut builder = Session::builder()
            .map_err(|e| ort_err("create session builder", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| ort_err("set optimization level", e))?;
        builder = if let Some(idx) = model_cfg.device.cuda_index() {
            builder
                .with_execution_providers([
                    CUDAExecutionProvider::default()
                        .with_device_id(idx as i32)
                        .build(),
                    CPUExecutionProvider::default().build(),
                ])
                .map_err(|e| ort_err("register CUDA execution provider", e))?
        } else {
            builder
                .with_execution_providers([CPUExecutionProvider::default().build()])
                .map_err(|e| ort_err("register CPU execution provider", e))?
        };
        let session = builder
            .commit_from_file(&onnx_file)
            .map_err(|e| ort_err("load ONNX model", e))?;

        // Trust the graph's own input/output names; fall back to the contract's
        // declared order if the model omits names.
        let mut input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        if input_names.is_empty() {
            input_names = contract.inputs.iter().map(|s| s.name.clone()).collect();
        }
        let mut output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        if output_names.is_empty() {
            output_names = contract.outputs.iter().map(|s| s.name.clone()).collect();
        }

        Ok((
            Box::new(OnnxBackend {
                session: Mutex::new(session),
                input_names,
                output_names,
            }),
            contract,
        ))
    }
}

impl Backend for OnnxBackend {
    fn platform(&self) -> &'static str {
        "onnxruntime_onnx"
    }

    fn infer(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, Status> {
        if inputs.len() != self.input_names.len() {
            return Err(Status::invalid_argument(format!(
                "ONNX model expects {} input(s), got {}",
                self.input_names.len(),
                inputs.len()
            )));
        }
        let mut ort_inputs: Vec<(Cow<'static, str>, DynValue)> = Vec::with_capacity(inputs.len());
        for (name, t) in self.input_names.iter().zip(inputs.iter()) {
            ort_inputs.push((Cow::Owned(name.clone()), tensor_to_ort_value(t)?));
        }

        let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());
        let outputs = session
            .run(ort_inputs)
            .map_err(|e| ort_err("run inference", e))?;

        let mut result = Vec::with_capacity(self.output_names.len());
        for name in &self.output_names {
            let value = outputs
                .get(name.as_str())
                .ok_or_else(|| Status::internal(format!("ONNX output '{name}' missing")))?;
            result.push(ort_value_to_tensor(name, value)?);
        }
        Ok(result)
    }
}

fn find_exactly_one_onnx_file(model_dir: &Path) -> Result<PathBuf, Status> {
    let entries = std::fs::read_dir(model_dir)
        .map_err(|err| Status::internal(format!("failed to read model directory: {err}")))?;
    let mut onnx_files = Vec::new();
    for entry in entries {
        let entry =
            entry.map_err(|err| Status::internal(format!("failed to read model entry: {err}")))?;
        let path = entry.path();
        if path.is_file() && path.extension().is_some_and(|ext| ext == "onnx") {
            onnx_files.push(path);
        }
    }
    onnx_files.sort();
    match onnx_files.len() {
        1 => Ok(onnx_files.remove(0)),
        0 => Err(Status::failed_precondition(
            "model must contain exactly one .onnx file; found none",
        )),
        count => Err(Status::failed_precondition(format!(
            "model must contain exactly one .onnx file; found {count}"
        ))),
    }
}

fn tensor_to_ort_value(t: &Tensor) -> Result<DynValue, Status> {
    let shape: Vec<i64> = t.shape.clone();
    dispatch_dtype!(
        t.dtype.as_str(),
        |Elem| {
            let data = bytes_to_vec::<Elem>(&t.data, &t.name)?;
            let tensor = ort::value::Tensor::<Elem>::from_array((shape, data))
                .map_err(|e| ort_err("build input tensor", e))?;
            Ok(tensor.into_dyn())
        },
        Err(Status::invalid_argument(format!(
            "ONNX backend: unsupported input dtype '{}'",
            t.dtype
        )))
    )
}

fn ort_value_to_tensor(name: &str, value: &DynValue) -> Result<Tensor, Status> {
    let dtype = ort_dtype_to_canonical(value)?;
    let (shape, data): (Vec<i64>, Vec<u8>) = dispatch_dtype!(
        dtype,
        |Elem| {
            let (out_shape, slice) = value
                .try_extract_tensor::<Elem>()
                .map_err(|e| ort_err("extract output tensor", e))?;
            (out_shape.iter().copied().collect(), slice_to_bytes(slice))
        },
        return Err(Status::internal(format!(
            "ONNX backend: unsupported output dtype '{dtype}'"
        )))
    );
    Ok(Tensor {
        name: name.to_string(),
        shape,
        dtype: dtype.to_string(),
        data,
    })
}

fn ort_dtype_to_canonical(value: &Value) -> Result<&'static str, Status> {
    use TensorElementType as T;
    let et = value
        .dtype()
        .tensor_type()
        .ok_or_else(|| Status::internal("ONNX output is not a tensor"))?;
    Ok(match et {
        T::Float32 => "float32",
        T::Float64 => "float64",
        T::Int8 => "int8",
        T::Int16 => "int16",
        T::Int32 => "int32",
        T::Int64 => "int64",
        T::Uint8 => "uint8",
        T::Uint16 => "uint16",
        T::Uint32 => "uint32",
        T::Uint64 => "uint64",
        T::Bool => "bool",
        T::Float16 => "float16",
        T::Bfloat16 => "bfloat16",
        other => {
            return Err(Status::internal(format!(
                "ONNX backend: unsupported output element type {other:?}"
            )));
        }
    })
}

fn ort_err<E: std::fmt::Display>(ctx: &str, e: E) -> Status {
    Status::internal(format!("ONNX backend: failed to {ctx}: {e}"))
}
