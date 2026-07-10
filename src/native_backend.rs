//! Native, in-process inference backends: ONNX (via `ort` / ONNX Runtime) and
//! TensorFlow (via `tensorflow` / libtensorflow SavedModels).
//!
//! Unlike the Rust `.pt` path, these engines don't use libtorch, so they operate
//! on a backend-agnostic [`NativeTensor`] (canonical dtype + raw little-endian
//! bytes) rather than a `tch::Tensor`. Each engine is compiled only when its
//! Cargo feature (`onnx` / `tensorflow`) is enabled; with neither feature this
//! module is just the [`NativeTensor`]/[`NativeModel`] contract and a loader that
//! explains how to turn the backend on.

use tonic::Status;

/// A backend-agnostic tensor handed to and returned by a [`NativeModel`].
///
/// `data` is row-major, little-endian, matching the wire contract used
/// everywhere else in nereid. `dtype` is a canonical lowercase name (see
/// `crate::dtype`).
///
/// Some fields are only *read* by the feature-gated engines, so without either
/// backend compiled in they are write-only — allow that rather than warn.
#[cfg_attr(not(any(feature = "onnx", feature = "tensorflow")), allow(dead_code))]
#[derive(Clone, Debug)]
pub struct NativeTensor {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: String,
    pub data: Vec<u8>,
}

/// A loaded native model. One boxed instance lives inside each native worker
/// thread; `run` executes a single forward pass. Inputs arrive in the order the
/// model's contract declares them; outputs must come back in the model's
/// declared output order.
pub trait NativeModel: Send {
    fn run(&mut self, inputs: Vec<NativeTensor>) -> Result<Vec<NativeTensor>, Status>;
}

/// Which native engine to load. Mirrors `DetectedBackendKind`'s native variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeKind {
    Onnx,
    Tensorflow,
}

/// Everything the loaders need that isn't the model bytes themselves.
#[cfg_attr(not(any(feature = "onnx", feature = "tensorflow")), allow(dead_code))]
pub struct NativeLoadSpec<'a> {
    /// Model path: the `.onnx` file for ONNX, or the SavedModel directory for TF.
    pub path: &'a std::path::Path,
    /// CUDA device index, or `None` for CPU.
    pub cuda_index: Option<usize>,
    /// Ordered input tensor names from the model contract (used to bind inputs
    /// by name for engines that key on names).
    pub input_names: Vec<String>,
    /// Ordered output tensor names from the model contract.
    pub output_names: Vec<String>,
    /// TensorFlow SavedModel signature key (default `"serving_default"`); unused
    /// by ONNX, so only read when the `tensorflow` backend is compiled in.
    #[cfg_attr(not(feature = "tensorflow"), allow(dead_code))]
    pub signature: String,
}

/// Load a native model of `kind`. Returns a clear error when the corresponding
/// feature wasn't compiled in, so a misconfigured deployment fails at startup
/// with actionable guidance rather than a confusing "unsupported backend".
#[cfg_attr(
    not(any(feature = "onnx", feature = "tensorflow")),
    allow(unused_variables)
)]
pub fn load_native_model(
    kind: NativeKind,
    spec: NativeLoadSpec<'_>,
) -> Result<Box<dyn NativeModel>, Status> {
    match kind {
        #[cfg(feature = "onnx")]
        NativeKind::Onnx => Ok(Box::new(onnx::OnnxModel::load(spec)?)),
        #[cfg(not(feature = "onnx"))]
        NativeKind::Onnx => Err(Status::failed_precondition(
            "model uses the ONNX backend, but this server was built without it. \
             Rebuild with `--features onnx` (or `./build.sh --onnx`).",
        )),
        #[cfg(feature = "tensorflow")]
        NativeKind::Tensorflow => Ok(Box::new(tensorflow_backend::TfModel::load(spec)?)),
        #[cfg(not(feature = "tensorflow"))]
        NativeKind::Tensorflow => Err(Status::failed_precondition(
            "model uses the TensorFlow backend, but this server was built without it. \
             Rebuild with `--features tensorflow` (or `./build.sh --tensorflow`).",
        )),
    }
}

/// Applies `$body` with `$t` bound to the Rust element type for a canonical
/// dtype name — the one place the KServe dtype set maps to concrete types for
/// the native engines. Unknown/unsupported names hit `$err`.
#[cfg(feature = "onnx")]
macro_rules! dispatch_dtype {
    ($dtype:expr, |$t:ident| $body:expr, $err:expr) => {
        match $dtype {
            "float32" => {
                type $t = f32;
                $body
            }
            "float64" => {
                type $t = f64;
                $body
            }
            "int8" => {
                type $t = i8;
                $body
            }
            "int16" => {
                type $t = i16;
                $body
            }
            "int32" => {
                type $t = i32;
                $body
            }
            "int64" => {
                type $t = i64;
                $body
            }
            "uint8" => {
                type $t = u8;
                $body
            }
            "uint16" => {
                type $t = u16;
                $body
            }
            "uint32" => {
                type $t = u32;
                $body
            }
            "uint64" => {
                type $t = u64;
                $body
            }
            "bool" => {
                type $t = bool;
                $body
            } // 1 byte; both engines take Rust `bool`
            "float16" => {
                type $t = half::f16;
                $body
            }
            "bfloat16" => {
                type $t = half::bf16;
                $body
            }
            _ => $err,
        }
    };
}

/// Like [`dispatch_dtype`] but without `bfloat16` — the `tensorflow` crate does
/// not implement `TensorType` for `half::bf16`, so a BF16 tensor is rejected on
/// the TF path rather than failing to compile.
#[cfg(feature = "tensorflow")]
macro_rules! dispatch_dtype_tf {
    ($dtype:expr, |$t:ident| $body:expr, $err:expr) => {
        match $dtype {
            "float32" => {
                type $t = f32;
                $body
            }
            "float64" => {
                type $t = f64;
                $body
            }
            "int8" => {
                type $t = i8;
                $body
            }
            "int16" => {
                type $t = i16;
                $body
            }
            "int32" => {
                type $t = i32;
                $body
            }
            "int64" => {
                type $t = i64;
                $body
            }
            "uint8" => {
                type $t = u8;
                $body
            }
            "uint16" => {
                type $t = u16;
                $body
            }
            "uint32" => {
                type $t = u32;
                $body
            }
            "uint64" => {
                type $t = u64;
                $body
            }
            "bool" => {
                type $t = bool;
                $body
            }
            "float16" => {
                type $t = half::f16;
                $body
            }
            _ => $err,
        }
    };
}

/// Reinterpret a little-endian byte buffer as `Vec<T>` (T: bytemuck-free, plain
/// old data). Validates the length is a whole number of elements.
#[cfg(any(feature = "onnx", feature = "tensorflow"))]
fn bytes_to_vec<T: Copy>(bytes: &[u8], model: &str) -> Result<Vec<T>, Status> {
    let esz = std::mem::size_of::<T>();
    if esz == 0 || !bytes.len().is_multiple_of(esz) {
        return Err(Status::invalid_argument(format!(
            "model '{model}': input byte length {} is not a multiple of element size {esz}",
            bytes.len()
        )));
    }
    let n = bytes.len() / esz;
    let mut out = Vec::<T>::with_capacity(n);
    // SAFETY: T is POD; we copy exactly n*esz bytes from a validated buffer.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
        out.set_len(n);
    }
    Ok(out)
}

/// Reinterpret a `&[T]` (POD) as its little-endian bytes.
#[cfg(any(feature = "onnx", feature = "tensorflow"))]
fn slice_to_bytes<T: Copy>(slice: &[T]) -> Vec<u8> {
    let byte_len = std::mem::size_of_val(slice);
    let mut out = vec![0u8; byte_len];
    // SAFETY: copying POD bytes out of a valid slice into an equally-sized buffer.
    unsafe {
        std::ptr::copy_nonoverlapping(slice.as_ptr() as *const u8, out.as_mut_ptr(), byte_len);
    }
    out
}

// ---------------------------------------------------------------------------
// ONNX Runtime backend
// ---------------------------------------------------------------------------
#[cfg(feature = "onnx")]
mod onnx {
    use super::{NativeLoadSpec, NativeModel, NativeTensor, bytes_to_vec, slice_to_bytes};
    use ort::execution_providers::{CPUExecutionProvider, CUDAExecutionProvider};
    use ort::session::Session;
    use ort::session::builder::GraphOptimizationLevel;
    use ort::value::{DynValue, TensorElementType, Value};
    use std::borrow::Cow;
    use tonic::Status;

    pub struct OnnxModel {
        session: Session,
        input_names: Vec<String>,
        output_names: Vec<String>,
    }

    impl OnnxModel {
        pub fn load(spec: NativeLoadSpec<'_>) -> Result<Self, Status> {
            let mut builder = Session::builder()
                .map_err(|e| ort_err("create session builder", e))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| ort_err("set optimization level", e))?;

            // Prefer CUDA when the model asks for it, falling back to CPU.
            builder = if let Some(idx) = spec.cuda_index {
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
                .commit_from_file(spec.path)
                .map_err(|e| ort_err("load ONNX model", e))?;

            // Trust the graph's own input/output names; fall back to the
            // contract's declared order if the model omits names.
            let mut input_names: Vec<String> = session
                .inputs()
                .iter()
                .map(|i| i.name().to_string())
                .collect();
            if input_names.is_empty() {
                input_names = spec.input_names.clone();
            }
            let mut output_names: Vec<String> = session
                .outputs()
                .iter()
                .map(|o| o.name().to_string())
                .collect();
            if output_names.is_empty() {
                output_names = spec.output_names.clone();
            }

            Ok(Self {
                session,
                input_names,
                output_names,
            })
        }
    }

    impl NativeModel for OnnxModel {
        fn run(&mut self, inputs: Vec<NativeTensor>) -> Result<Vec<NativeTensor>, Status> {
            if inputs.len() != self.input_names.len() {
                return Err(Status::invalid_argument(format!(
                    "ONNX model expects {} input(s), got {}",
                    self.input_names.len(),
                    inputs.len()
                )));
            }

            // Bind each input by position to the graph's input name.
            let mut ort_inputs: Vec<(Cow<'static, str>, DynValue)> =
                Vec::with_capacity(inputs.len());
            for (name, t) in self.input_names.iter().zip(inputs.iter()) {
                ort_inputs.push((Cow::Owned(name.clone()), native_to_ort_value(t)?));
            }

            let outputs = self
                .session
                .run(ort_inputs)
                .map_err(|e| ort_err("run inference", e))?;

            // Return outputs in the model's declared output order.
            let mut result = Vec::with_capacity(self.output_names.len());
            for name in &self.output_names {
                let value = outputs
                    .get(name.as_str())
                    .ok_or_else(|| Status::internal(format!("ONNX output '{name}' missing")))?;
                result.push(ort_value_to_native(name, value)?);
            }
            Ok(result)
        }
    }

    fn native_to_ort_value(t: &NativeTensor) -> Result<DynValue, Status> {
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

    fn ort_value_to_native(name: &str, value: &DynValue) -> Result<NativeTensor, Status> {
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
        Ok(NativeTensor {
            name: name.to_string(),
            shape,
            dtype: dtype.to_string(),
            data,
        })
    }

    /// Map an ort value's element type to a canonical dtype name.
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
}

// ---------------------------------------------------------------------------
// TensorFlow (libtensorflow SavedModel) backend
// ---------------------------------------------------------------------------
#[cfg(feature = "tensorflow")]
mod tensorflow_backend {
    use super::{NativeLoadSpec, NativeModel, NativeTensor, bytes_to_vec, slice_to_bytes};
    use tensorflow::{
        DataType, Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor,
        TensorType,
    };
    use tonic::Status;

    /// A resolved feed/fetch: the graph operation plus its output index.
    struct IoBinding {
        name: String,
        operation: Operation,
        index: i32,
    }

    pub struct TfModel {
        bundle: SavedModelBundle,
        _graph: Graph,
        inputs: Vec<IoBinding>,
        outputs: Vec<IoBinding>,
    }

    /// The declared names, or — for a single-tensor model whose flat textproto
    /// names nothing — the signature's own keys (sorted for determinism).
    fn names_or_signature_keys<V>(
        declared: &[String],
        map: &std::collections::HashMap<String, V>,
    ) -> Vec<String> {
        if !declared.is_empty() {
            return declared.to_vec();
        }
        let mut keys: Vec<String> = map.keys().cloned().collect();
        keys.sort();
        keys
    }

    impl TfModel {
        pub fn load(spec: NativeLoadSpec<'_>) -> Result<Self, Status> {
            let mut graph = Graph::new();
            let mut opts = SessionOptions::new();
            // Pin to a specific GPU when requested via a device visibility config.
            if let Some(idx) = spec.cuda_index {
                // config.proto: gpu_options.visible_device_list = "<idx>"
                let proto = gpu_visible_device_config(idx);
                opts.set_config(&proto)
                    .map_err(|e| tf_err("set GPU device config", e))?;
            }
            let bundle = SavedModelBundle::load(&opts, ["serve"], &mut graph, spec.path)
                .map_err(|e| tf_err("load SavedModel", e))?;

            let sig = bundle
                .meta_graph_def()
                .get_signature(&spec.signature)
                .map_err(|e| tf_err(&format!("find signature '{}'", spec.signature), e))?;

            // Resolve declared inputs/outputs to graph operations, preserving the
            // contract's ordering.
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

            let input_keys = names_or_signature_keys(&spec.input_names, sig.inputs());
            let output_keys = names_or_signature_keys(&spec.output_names, sig.outputs());
            let inputs = resolve(&input_keys, true)?;
            let outputs = resolve(&output_keys, false)?;

            Ok(Self {
                bundle,
                _graph: graph,
                inputs,
                outputs,
            })
        }
    }

    impl NativeModel for TfModel {
        fn run(&mut self, inputs: Vec<NativeTensor>) -> Result<Vec<NativeTensor>, Status> {
            if inputs.len() != self.inputs.len() {
                return Err(Status::invalid_argument(format!(
                    "TensorFlow model expects {} input(s), got {}",
                    self.inputs.len(),
                    inputs.len()
                )));
            }

            let mut args = SessionRunArgs::new();

            // Feeds. A TF `Tensor<T>` owns its buffer, so we build one per input
            // and keep them alive for the duration of the run.
            let mut feed_tensors: Vec<Box<dyn std::any::Any>> = Vec::with_capacity(inputs.len());
            for (binding, t) in self.inputs.iter().zip(inputs.iter()) {
                feed_input(&mut args, binding, t, &mut feed_tensors)?;
            }

            // Fetches, in declared order.
            let fetch_tokens: Vec<_> = self
                .outputs
                .iter()
                .map(|b| args.request_fetch(&b.operation, b.index))
                .collect();

            self.bundle
                .session
                .run(&mut args)
                .map_err(|e| tf_err("run session", e))?;

            let mut result = Vec::with_capacity(self.outputs.len());
            for (binding, token) in self.outputs.iter().zip(fetch_tokens.iter()) {
                result.push(fetch_output(&mut args, binding, *token)?);
            }
            Ok(result)
        }
    }

    /// Build a `Tensor<T>` from a NativeTensor and add it as a feed.
    fn feed_input(
        args: &mut SessionRunArgs,
        binding: &IoBinding,
        t: &NativeTensor,
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
        let mut tensor = Tensor::<T>::new(dims);
        if tensor.len() != values.len() {
            return Err(Status::invalid_argument(format!(
                "TensorFlow backend: model '{model}' input has {} elements but shape implies {}",
                values.len(),
                tensor.len()
            )));
        }
        tensor.copy_from_slice(&values);
        // add_feed borrows the tensor; store it so the borrow outlives run().
        let boxed = Box::new(tensor);
        // SAFETY-free: we hand a reference from the boxed tensor which we keep
        // alive in `keep_alive` for the whole call.
        let tref: &Tensor<T> = boxed.as_ref();
        // Extend the reference lifetime to the SessionRunArgs' lifetime; safe
        // because `keep_alive` outlives `args` at the call site.
        let tref: &Tensor<T> = unsafe { std::mem::transmute(tref) };
        args.add_feed(&binding.operation, binding.index, tref);
        keep_alive.push(boxed);
        Ok(())
    }

    fn fetch_output(
        args: &mut SessionRunArgs,
        binding: &IoBinding,
        token: tensorflow::FetchToken,
    ) -> Result<NativeTensor, Status> {
        // Determine the output dtype from the graph operation.
        let dtype = tf_dtype_to_canonical(binding.operation.output_type(binding.index as usize))?;
        let (shape, data) = dispatch_dtype_tf!(
            dtype,
            |Elem| {
                let tensor: Tensor<Elem> = args
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
        Ok(NativeTensor {
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

    /// A serialized `ConfigProto` pinning TF to a single visible GPU. Hand-encoded
    /// to avoid a protobuf dependency: field 6 (gpu_options) → field 5
    /// (visible_device_list, string).
    fn gpu_visible_device_config(idx: usize) -> Vec<u8> {
        let s = idx.to_string();
        let inner_len = 2 + s.len(); // tag(1) + len(1) + string
        let mut buf = Vec::with_capacity(2 + inner_len);
        buf.push(0x32); // field 6 (gpu_options), wire type 2 (length-delimited)
        buf.push(inner_len as u8);
        buf.push(0x2a); // field 5 (visible_device_list), wire type 2
        buf.push(s.len() as u8);
        buf.extend_from_slice(s.as_bytes());
        buf
    }

    fn tf_err(ctx: &str, e: tensorflow::Status) -> Status {
        Status::internal(format!("TensorFlow backend: failed to {ctx}: {e}"))
    }
}
