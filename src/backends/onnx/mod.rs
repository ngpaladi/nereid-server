//! ONNX backend, via ONNX Runtime (`ort`). In-process; the CUDA execution
//! provider is selected when the model's device is `cuda`.

use std::path::Path;

use tonic::Status;

use crate::backend::{Backend, BackendRegistration, Contract};
use crate::config::ModelConfig;

/// A `.onnx` model folder plus the tensor contract.
fn detect(model_dir: &Path) -> bool {
    model_dir.join("model_inference.textproto").is_file()
        && crate::backend::dir_has_ext(model_dir, "onnx")
}

fn load(model_dir: &Path, model_cfg: &ModelConfig) -> Result<(Box<dyn Backend>, Contract), Status> {
    #[cfg(feature = "onnx")]
    {
        imp::OnnxBackend::load(model_dir, model_cfg)
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = model_dir;
        Err(crate::backend::missing_feature(
            &model_cfg.name,
            "ONNX",
            "onnx",
        ))
    }
}

inventory::submit! {
    BackendRegistration {
        name: "onnx",
        version: "0.1.0",
        aliases: &[],
        describes: "a .onnx model + model_inference.textproto",
        auto_detect: true,
        detect,
        load,
    }
}

#[cfg(feature = "onnx")]
mod imp;
