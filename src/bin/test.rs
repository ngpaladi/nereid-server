use std::error::Error;
use tch::{CModule, Device, IValue, Kind, Tensor};

const MODEL_PATH: &str = "/Users/divij_agarwal/Downloads/VSCodeFiles/nereid-server/ml-backends/model3/mlp.pt";
const INPUT_SPECS: &[InputTensorSpec] = &[
    InputTensorSpec { shape: &[1, 16] },
];

#[derive(Debug, Clone, Copy)]
struct InputTensorSpec {
    shape: &'static [i64],
}

// Producing random tensors
fn build_input_tensors_from_specs(input_specs: &[InputTensorSpec], device: Device) -> Vec<Tensor> {
    input_specs
        .iter()
        .map(|spec| Tensor::randn(spec.shape, (Kind::Float, device)))
        .collect()
}

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

fn print_ival_full(label: &str, value: &IValue) {
    match value {
        IValue::Tensor(tensor) => {
            println!("{label}: {}", tensor_to_full_string(tensor));
        }
        _ => {
            println!("{label}: {value:?}");
        }
    }
}

fn run_forward_pass(model_path: &str, input_specs: &[InputTensorSpec]) -> Result<(), Box<dyn Error>> {
    let device = Device::Cpu;
    let model = CModule::load_on_device(model_path, device)?;

    let inputs = build_input_tensors_from_specs(input_specs, device);
    if inputs.len() != input_specs.len() {
        return Err("input tensor count does not match INPUT_SPECS".into());
    }

    let input_ivalues: Vec<IValue> = inputs.into_iter().map(IValue::Tensor).collect();
    for (index, input) in input_ivalues.iter().enumerate() {
        print_ival_full(&format!("Input tensor {index} full"), input);
    }
    let output = model.forward_is(&input_ivalues)?;

    println!("Model path: {model_path}");
    println!("Input tensor count: {}", input_specs.len());
    for (index, spec) in input_specs.iter().enumerate() {
        println!("Input tensor {index} shape: {:?}", spec.shape);
    }
    print_ival_full("Model output full", &output);

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    run_forward_pass(MODEL_PATH, INPUT_SPECS)
}
