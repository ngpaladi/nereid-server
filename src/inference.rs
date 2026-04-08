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
