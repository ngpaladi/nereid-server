use tch::{CModule, Device, IValue, Tensor};

/// A serialized output tensor: `(shape, row-major little-endian bytes, canonical
/// dtype)`.
pub type TensorOutput = (Vec<i64>, Vec<u8>, String);

/// Serialize a tensor to `(shape, row-major little-endian bytes, canonical
/// dtype)`. The dtype is preserved from the tensor's kind, so non-float outputs
/// (e.g. int64) round-trip unchanged.
pub fn tensor_to_bytes(tensor: &Tensor) -> Result<TensorOutput, tch::TchError> {
    let tensor = tensor.to_device(Device::Cpu).contiguous();
    let shape = tensor.size();
    let kind = tensor.kind();
    let dtype = crate::dtype::canonical_from_kind(kind)
        .ok_or_else(|| tch::TchError::Torch(format!("unsupported output tensor kind {kind:?}")))?
        .to_string();

    let flat = tensor.reshape([-1]);
    let numel = flat.numel();
    let mut bytes = vec![0u8; numel * kind.elt_size_in_bytes()];
    flat.copy_data_u8(&mut bytes, numel);
    Ok((shape, bytes, dtype))
}

/// Run the model's single-input forward pass, preserving the input tensor's
/// dtype (no forced cast to float), and return the output tensor serialized by
/// [`tensor_to_bytes`].
pub fn run_forward_pass(
    model: &CModule,
    device: Device,
    input_tensor: &Tensor,
) -> Result<TensorOutput, tch::TchError> {
    let input_tensor = input_tensor.to_device(device);
    let output = model.forward_is(&[IValue::Tensor(input_tensor)])?;

    match output {
        IValue::Tensor(tensor) => tensor_to_bytes(&tensor),
        other => Err(tch::TchError::Torch(format!(
            "model output was non-tensor: {other:?}"
        ))),
    }
}

/// Run a multi-input forward pass. The model receives all `inputs` positionally
/// and may return a single tensor or a tuple of tensors; each output is
/// serialized by [`tensor_to_bytes`]. Used by the additive named-multi-tensor
/// path — the single-tensor [`run_forward_pass`] is untouched.
pub fn run_multi_forward_pass(
    model: &CModule,
    device: Device,
    inputs: Vec<Tensor>,
) -> Result<Vec<TensorOutput>, tch::TchError> {
    let ivalues: Vec<IValue> = inputs
        .into_iter()
        .map(|t| IValue::Tensor(t.to_device(device)))
        .collect();
    let output = model.forward_is(&ivalues)?;

    match output {
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
    use super::{run_forward_pass, tensor_to_bytes};
    use crate::config::load_server_config;
    use crate::model_runtime::tensor_from_input_bytes;
    use std::path::PathBuf;
    use tch::{CModule, Device, Kind, Tensor};

    /// Serialization must round-trip byte-for-byte for the non-float libtorch
    /// kinds — especially the 2-byte types with no native Rust element type
    /// (`Half`, `BFloat16`), where a wrong element size would silently corrupt
    /// output. `tensor_to_bytes` (copy_data_u8) and `tensor_from_input_bytes`
    /// (from_data_size) are both byte-generic, so this exercises 1/2/4/8-byte
    /// widths without needing a per-dtype `.pt` fixture.
    #[test]
    fn tensor_bytes_round_trip_across_kinds() {
        let values = [1.0f64, 2.0, 3.0, 4.0];
        for (kind, canonical, elt) in [
            (Kind::Int16, "int16", 2usize),
            (Kind::Int, "int32", 4),
            (Kind::Int64, "int64", 8),
            (Kind::Int8, "int8", 1),
            (Kind::Half, "float16", 2),
            (Kind::BFloat16, "bfloat16", 2),
            (Kind::Double, "float64", 8),
        ] {
            let tensor = Tensor::from_slice(&values).reshape([2, 2]).to_kind(kind);
            let (shape, bytes, dtype) = tensor_to_bytes(&tensor).expect("serialize");
            assert_eq!(dtype, canonical, "dtype label for {kind:?}");
            assert_eq!(shape, vec![2, 2]);
            assert_eq!(bytes.len(), 4 * elt, "byte width for {kind:?}");

            // Rebuild the tensor from those bytes and re-serialize: byte-identical
            // means the element size and layout survived the round trip.
            let rebuilt = tensor_from_input_bytes(&bytes, &shape, "m", kind).expect("deserialize");
            let (_, bytes2, _) = tensor_to_bytes(&rebuilt).expect("re-serialize");
            assert_eq!(bytes, bytes2, "round-trip bytes for {kind:?}");
        }
    }

    #[test]
    fn run_forward_pass_is_deterministic_for_fixed_input() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let config = load_server_config(&manifest_dir.join("nereid.yaml.example"))
            .expect("example config should load");
        let model_path = manifest_dir
            .join(config.server.ml_backends_path)
            .join("model3")
            .join("mlp.pt");

        assert!(
            model_path.is_file(),
            "expected test model to exist at {}",
            model_path.display()
        );

        let input_values: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let input_tensor = Tensor::from_slice(&input_values)
            .reshape([1, 16])
            .to_device(Device::Cpu);

        let model = CModule::load_on_device(
            model_path.to_str().expect("valid UTF-8 model path"),
            Device::Cpu,
        )
        .expect("model should load");
        let (baseline_shape, baseline_bytes, _dtype) =
            run_forward_pass(&model, Device::Cpu, &input_tensor)
                .expect("initial forward pass succeeds");

        for i in 0..1000 {
            let (shape, bytes, _dtype) = run_forward_pass(&model, Device::Cpu, &input_tensor)
                .unwrap_or_else(|err| panic!("forward pass {i} failed: {err}"));
            assert_eq!(shape, baseline_shape, "shape changed at iteration {i}");
            assert_eq!(
                bytes, baseline_bytes,
                "output bytes changed at iteration {i}"
            );
        }
    }
}
