use tch::{CModule, Device, IValue, Kind, Tensor};

pub fn run_forward_pass(
    model: &CModule,
    device: Device,
    input_tensor: &Tensor,
) -> Result<(Vec<i64>, Vec<u8>), tch::TchError> {
    let input_tensor = input_tensor.to_device(device).to_kind(Kind::Float);
    let output = model.forward_is(&[IValue::Tensor(input_tensor)])?;

    match output {
        IValue::Tensor(tensor) => {
            let output_tensor = tensor.to_device(Device::Cpu).to_kind(Kind::Float);
            let shape = output_tensor.size();
            let flat = output_tensor.reshape([-1]);
            let values = Vec::<f32>::try_from(&flat)?;
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            Ok((shape, bytes))
        }
        other => Err(tch::TchError::Torch(format!(
            "model output was non-tensor: {other:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::run_forward_pass;
    use crate::config::load_server_config;
    use std::path::PathBuf;
    use tch::{CModule, Device, Tensor};

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
        let (baseline_shape, baseline_bytes) = run_forward_pass(&model, Device::Cpu, &input_tensor)
            .expect("initial forward pass succeeds");

        for i in 0..1000 {
            let (shape, bytes) = run_forward_pass(&model, Device::Cpu, &input_tensor)
                .unwrap_or_else(|err| panic!("forward pass {i} failed: {err}"));
            assert_eq!(shape, baseline_shape, "shape changed at iteration {i}");
            assert_eq!(
                bytes, baseline_bytes,
                "output bytes changed at iteration {i}"
            );
        }
    }
}
