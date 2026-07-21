use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use tensorflow::{
    DataType, Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs,
    Tensor as TfTensor, TensorType,
};
use tonic::Status;

use crate::backend::Tensor;
use crate::backend::native_common::{bytes_to_vec, dispatch_dtype_tf, slice_to_bytes};
use crate::backend::{Backend, Contract};
use crate::config::ModelConfig;

/// A resolved feed/fetch: the graph operation plus its output index.
struct IoBinding {
    name: String,
    operation: Operation,
    index: i32,
}

pub struct TensorflowBackend {
    // The SavedModel session; the `Mutex` serializes inference and makes the
    // backend shareable across the async runtime.
    inner: Mutex<Loaded>,
}

struct Loaded {
    bundle: SavedModelBundle,
    _graph: Graph,
    inputs: Vec<IoBinding>,
    outputs: Vec<IoBinding>,
}

impl TensorflowBackend {
    pub fn load(
        model_dir: &Path,
        model_cfg: &ModelConfig,
    ) -> Result<(Box<dyn Backend>, Contract), Status> {
        let contract = Contract::parse(model_dir)?;
        if !contract.has_input() {
            return Err(Status::failed_precondition(format!(
                "TensorFlow model '{}' must declare input_shape in model_inference.textproto",
                model_cfg.name
            )));
        }
        let input_names: Vec<String> = contract.inputs.iter().map(|s| s.name.clone()).collect();
        let output_names: Vec<String> = contract.outputs.iter().map(|s| s.name.clone()).collect();

        let mut graph = Graph::new();
        let mut opts = SessionOptions::new();
        if let Some(idx) = model_cfg.device.cuda_index() {
            opts.set_config(&gpu_visible_device_config(idx))
                .map_err(|e| tf_err("set GPU device config", e))?;
        }
        let bundle = SavedModelBundle::load(&opts, ["serve"], &mut graph, model_dir)
            .map_err(|e| tf_err("load SavedModel", e))?;

        let signature = model_cfg.tf_signature();
        let sig = bundle
            .meta_graph_def()
            .get_signature(signature)
            .map_err(|e| tf_err(&format!("find signature '{signature}'"), e))?;

        let resolve = |names: &[String], is_input: bool| -> Result<Vec<IoBinding>, Status> {
            let mut out = Vec::with_capacity(names.len());
            for name in names {
                let info = if is_input {
                    sig.get_input(name)
                } else {
                    sig.get_output(name)
                }
                .map_err(|e| tf_err(&format!("resolve tensor '{name}' in signature"), e))?;
                let op = graph
                    .operation_by_name_required(&info.name().name)
                    .map_err(|e| tf_err(&format!("find graph op for '{name}'"), e))?;
                out.push(IoBinding {
                    name: name.clone(),
                    operation: op,
                    index: info.name().index,
                });
            }
            Ok(out)
        };

        let input_keys = names_or_signature_keys(&input_names, sig.inputs());
        let output_keys = names_or_signature_keys(&output_names, sig.outputs());
        let inputs = resolve(&input_keys, true)?;
        let outputs = resolve(&output_keys, false)?;

        Ok((
            Box::new(TensorflowBackend {
                inner: Mutex::new(Loaded {
                    bundle,
                    _graph: graph,
                    inputs,
                    outputs,
                }),
            }),
            contract,
        ))
    }
}

impl Backend for TensorflowBackend {
    fn platform(&self) -> &'static str {
        "tensorflow_savedmodel"
    }

    fn infer(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, Status> {
        let loaded = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inputs.len() != loaded.inputs.len() {
            return Err(Status::invalid_argument(format!(
                "TensorFlow model expects {} input(s), got {}",
                loaded.inputs.len(),
                inputs.len()
            )));
        }

        let mut args = SessionRunArgs::new();
        // A TF `Tensor<T>` owns its buffer; keep each alive for the run.
        let mut feed_tensors: Vec<Box<dyn std::any::Any>> = Vec::with_capacity(inputs.len());
        for (binding, t) in loaded.inputs.iter().zip(inputs.iter()) {
            feed_input(&mut args, binding, t, &mut feed_tensors)?;
        }

        let fetch_tokens: Vec<_> = loaded
            .outputs
            .iter()
            .map(|b| args.request_fetch(&b.operation, b.index))
            .collect();

        loaded
            .bundle
            .session
            .run(&mut args)
            .map_err(|e| tf_err("run session", e))?;

        let mut result = Vec::with_capacity(loaded.outputs.len());
        for (binding, token) in loaded.outputs.iter().zip(fetch_tokens.iter()) {
            result.push(fetch_output(&mut args, binding, *token)?);
        }
        Ok(result)
    }
}

/// The declared names, or — for a single-tensor model whose flat textproto names
/// nothing that maps onto the signature — the signature's own keys (sorted).
fn names_or_signature_keys<V>(declared: &[String], map: &HashMap<String, V>) -> Vec<String> {
    // The contract always names a single-tensor input/output "input"/"output";
    // if those aren't the signature's keys, fall back to the signature's keys.
    if declared.iter().all(|n| map.contains_key(n)) && !declared.is_empty() {
        return declared.to_vec();
    }
    let mut keys: Vec<String> = map.keys().cloned().collect();
    keys.sort();
    keys
}

fn feed_input(
    args: &mut SessionRunArgs,
    binding: &IoBinding,
    t: &Tensor,
    keep_alive: &mut Vec<Box<dyn std::any::Any>>,
) -> Result<(), Status> {
    let dims: Vec<u64> = t.shape.iter().map(|d| *d as u64).collect();
    dispatch_dtype_tf!(
        t.dtype.as_str(),
        |Elem| feed_typed::<Elem>(args, binding, &dims, &t.data, &t.name, keep_alive),
        Err(Status::invalid_argument(format!(
            "TensorFlow backend: unsupported input dtype '{}'",
            t.dtype
        )))
    )
}

fn feed_typed<T: TensorType + Copy>(
    args: &mut SessionRunArgs,
    binding: &IoBinding,
    dims: &[u64],
    data: &[u8],
    model: &str,
    keep_alive: &mut Vec<Box<dyn std::any::Any>>,
) -> Result<(), Status> {
    let values = bytes_to_vec::<T>(data, model)?;
    let mut tensor = TfTensor::<T>::new(dims);
    if tensor.len() != values.len() {
        return Err(Status::invalid_argument(format!(
            "TensorFlow backend: model '{model}' input has {} elements but shape implies {}",
            values.len(),
            tensor.len()
        )));
    }
    tensor.copy_from_slice(&values);
    let boxed = Box::new(tensor);
    let tref: &TfTensor<T> = boxed.as_ref();
    // SAFETY: `keep_alive` outlives `args` at the call site, so the borrow the
    // feed holds stays valid for the whole run.
    let tref: &TfTensor<T> = unsafe { std::mem::transmute(tref) };
    args.add_feed(&binding.operation, binding.index, tref);
    keep_alive.push(boxed);
    Ok(())
}

fn fetch_output(
    args: &mut SessionRunArgs,
    binding: &IoBinding,
    token: tensorflow::FetchToken,
) -> Result<Tensor, Status> {
    let dtype = tf_dtype_to_canonical(binding.operation.output_type(binding.index as usize))?;
    let (shape, data) = dispatch_dtype_tf!(
        dtype,
        |Elem| {
            let tensor: TfTensor<Elem> = args
                .fetch(token)
                .map_err(|e| tf_err("fetch output tensor", e))?;
            let shape: Vec<i64> = tensor.dims().iter().map(|d| *d as i64).collect();
            let bytes = slice_to_bytes(&tensor[..]);
            (shape, bytes)
        },
        return Err(Status::internal(format!(
            "TensorFlow backend: unsupported output dtype '{dtype}'"
        )))
    );
    Ok(Tensor {
        name: binding.name.clone(),
        shape,
        dtype: dtype.to_string(),
        data,
    })
}

fn tf_dtype_to_canonical(dt: DataType) -> Result<&'static str, Status> {
    Ok(match dt {
        DataType::Float => "float32",
        DataType::Double => "float64",
        DataType::Int8 => "int8",
        DataType::Int16 => "int16",
        DataType::Int32 => "int32",
        DataType::Int64 => "int64",
        DataType::UInt8 => "uint8",
        DataType::UInt16 => "uint16",
        DataType::UInt32 => "uint32",
        DataType::UInt64 => "uint64",
        DataType::Bool => "bool",
        DataType::Half => "float16",
        DataType::BFloat16 => "bfloat16",
        other => {
            return Err(Status::internal(format!(
                "TensorFlow backend: unsupported output dtype {other:?}"
            )));
        }
    })
}

/// A serialized `ConfigProto` pinning TF to a single visible GPU. Hand-encoded to
/// avoid a protobuf dependency: field 6 (gpu_options) → field 5
/// (visible_device_list, string).
fn gpu_visible_device_config(idx: usize) -> Vec<u8> {
    let s = idx.to_string();
    let inner_len = 2 + s.len();
    let mut buf = Vec::with_capacity(2 + inner_len);
    buf.push(0x32);
    buf.push(inner_len as u8);
    buf.push(0x2a);
    buf.push(s.len() as u8);
    buf.extend_from_slice(s.as_bytes());
    buf
}

fn tf_err(ctx: &str, e: tensorflow::Status) -> Status {
    Status::internal(format!("TensorFlow backend: failed to {ctx}: {e}"))
}
