//! TorchScript (`.pt`) backend, via libtorch (`tch`).
//!
//! The detection predicate and the [`BackendRegistration`] are always compiled
//! (pure file inspection, no libtorch dependency); everything that touches
//! `tch` lives in the feature-gated `imp` module. So the core discovers this
//! backend at link time — no central enum or match — and a `.pt` folder still
//! gets a precise "rebuild with `--features torch`" message when the backend
//! isn't built in.

use std::path::Path;

use tonic::Status;

use crate::backend::{Backend, BackendRegistration, Contract};
use crate::config::ModelConfig;

/// A `.pt` model folder: a TorchScript file plus the tensor contract.
fn detect(model_dir: &Path) -> bool {
    model_dir.join("model_inference.textproto").is_file()
        && crate::backend::dir_has_ext(model_dir, "pt")
}

fn load(model_dir: &Path, model_cfg: &ModelConfig) -> Result<(Box<dyn Backend>, Contract), Status> {
    #[cfg(feature = "torch")]
    {
        imp::TorchBackend::load(model_dir, model_cfg)
    }
    #[cfg(not(feature = "torch"))]
    {
        let _ = model_dir;
        Err(crate::backend::missing_feature(
            &model_cfg.name,
            "TorchScript (.pt)",
            "torch",
        ))
    }
}

inventory::submit! {
    BackendRegistration {
        name: "torch",
        // "rust" is the historical name for this backend.
        aliases: &["rust"],
        describes: "a .pt model + model_inference.textproto",
        auto_detect: true,
        detect,
        load,
    }
}

#[cfg(feature = "torch")]
mod imp;
