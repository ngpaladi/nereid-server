//! TensorFlow backend, via libtensorflow SavedModels (`tensorflow` crate).
//! In-process; single GPU selection via `device: cuda[:idx]`.

use std::path::Path;

use tonic::Status;

use crate::backend::{Backend, BackendRegistration, Contract};
use crate::config::ModelConfig;

/// A TensorFlow SavedModel folder (`saved_model.pb` + `variables/`) plus the
/// tensor contract.
fn detect(model_dir: &Path) -> bool {
    model_dir.join("model_inference.textproto").is_file()
        && crate::backend::dir_is_saved_model(model_dir)
}

fn load(model_dir: &Path, model_cfg: &ModelConfig) -> Result<(Box<dyn Backend>, Contract), Status> {
    #[cfg(feature = "tensorflow")]
    {
        imp::TensorflowBackend::load(model_dir, model_cfg)
    }
    #[cfg(not(feature = "tensorflow"))]
    {
        let _ = model_dir;
        Err(crate::backend::missing_feature(
            &model_cfg.name,
            "TensorFlow",
            "tensorflow",
        ))
    }
}

inventory::submit! {
    BackendRegistration {
        name: "tensorflow",
        aliases: &[],
        describes: "a SavedModel (saved_model.pb + variables/) + model_inference.textproto",
        auto_detect: true,
        detect,
        load,
    }
}

#[cfg(feature = "tensorflow")]
mod imp;
