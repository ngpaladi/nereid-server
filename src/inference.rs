use std::error::Error;
use std::io;

use tch::{CModule, Device, IValue, Kind, Tensor};

fn tensor_to_full_string(tensor: &Tensor) -> String {
    let shape = tensor.size();
    let flat = tensor
        .to_device(Device::Cpu)
        .to_kind(Kind::Double)
        .reshape([-1]);

    match Vec::<f64>::try_from(&flat) {
        Ok(values) => format!("shape={shape:?}, values={values:?}"),
        Err(err) => format!("shape={shape:?}, failed to extract values: {err}"),
    }
}

pub fn run_forward_pass(
    model_path: &str,
    input_tensor: &Tensor,
) -> Result<(Vec<i64>, Vec<u8>), Box<dyn Error>> {
    let device = Device::Cpu;
    let model = CModule::load_on_device(model_path, device)?;

    let input_tensor = input_tensor.to_device(device).to_kind(Kind::Float);

    println!("Model path: {model_path}");
    println!(
        "Input tensor full: {}",
        tensor_to_full_string(&input_tensor)
    );

    let output = model.forward_is(&[IValue::Tensor(input_tensor)])?;

    match output {
        IValue::Tensor(tensor) => {
            let output_tensor = tensor.to_device(Device::Cpu).to_kind(Kind::Float);
            println!(
                "Model output full: {}",
                tensor_to_full_string(&output_tensor)
            );

            let shape = output_tensor.size();
            let flat = output_tensor.reshape([-1]);
            let values = Vec::<f32>::try_from(&flat)?;
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            Ok((shape, bytes))
        }
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("model output was non-tensor: {other:?}"),
        )
        .into()),
    }
}

#[cfg(test)]
mod tests {
    use super::run_forward_pass;
    use std::path::PathBuf;
    use tch::{Device, Tensor};

    #[test]
    fn run_forward_pass_is_deterministic_for_fixed_input() {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("ml-backends")
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

        let model_path_str = model_path.to_str().expect("valid UTF-8 model path");
        let (baseline_shape, baseline_bytes) =
            run_forward_pass(model_path_str, &input_tensor).expect("initial forward pass succeeds");

        for i in 0..1000 {
            let (shape, bytes) = run_forward_pass(model_path_str, &input_tensor)
                .unwrap_or_else(|err| panic!("forward pass {i} failed: {err}"));
            assert_eq!(shape, baseline_shape, "shape changed at iteration {i}");
            assert_eq!(bytes, baseline_bytes, "output bytes changed at iteration {i}");
        }
    }
}
