use std::error::Error;

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

pub fn run_forward_pass(model_path: &str, input_tensor: &Tensor) -> Result<(), Box<dyn Error>> {
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
            println!("Model output full: {}", tensor_to_full_string(&tensor));
        }
        other => {
            println!("Model output (non-tensor): {other:?}");
        }
    }

    Ok(())
}
