//! Compile-time C++ backend, via the `cxx-models` crate.
//!
//! Unlike every other backend here, a `cxx` model's code isn't on disk at all —
//! it's compiled into the server (bridged type-safely with [`cxx`], no
//! hand-rolled FFI) and keyed by the model's name, so "loading" is a registry
//! lookup rather than a file read and the model folder only supplies the tensor
//! contract.
//!
//! That is exactly what `auto_detect: false` is for: there is no file signature
//! to look for, so this backend is never guessed from a folder's contents and
//! has to be named with `backend: "cxx"` in `nereid.yaml`. A folder holding just
//! a `model_inference.textproto` would otherwise be a shape every compiled-in
//! backend matched equally well.

use std::path::Path;

use tonic::Status;

use crate::backend::{Backend, BackendRegistration, Contract};
use crate::config::ModelConfig;

/// All a `cxx` model's folder carries is the tensor contract; the code itself is
/// linked into the server. Registered with `auto_detect: false`, so this is
/// never used to guess a folder's backend — it only says what the folder needs
/// to hold once you've declared `backend: "cxx"`.
fn detect(model_dir: &Path) -> bool {
    model_dir.join("model_inference.textproto").is_file()
}

fn load(model_dir: &Path, model_cfg: &ModelConfig) -> Result<(Box<dyn Backend>, Contract), Status> {
    #[cfg(feature = "cxx")]
    {
        imp::CxxBackend::load(model_dir, model_cfg)
    }
    #[cfg(not(feature = "cxx"))]
    {
        let _ = model_dir;
        Err(crate::backend::missing_feature(
            &model_cfg.name,
            "compile-time C++ (cxx)",
            "cxx",
        ))
    }
}

inventory::submit! {
    BackendRegistration {
        name: "cxx",
        version: "0.1.0",
        aliases: &[],
        describes: "a model_inference.textproto for a C++ model compiled into the server",
        // Compiled-in code, not files on disk — declaration-only.
        auto_detect: false,
        detect,
        load,
    }
}

#[cfg(feature = "cxx")]
mod imp;
