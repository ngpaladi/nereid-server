//! C++ subprocess backend.
//!
//! A C++ model ships a `main.cpp` that is compiled into a `model` executable at
//! startup — mirroring how the Python backend builds a venv — and then run as a
//! child process per request, speaking the language-agnostic tensor contract in
//! [`subprocess`] (raw tensor on stdin, framed tensor out via
//! `NEREID_OUTPUT_PATH`). No recompiling the server, no unsafe FFI, and a crash
//! in the model is contained to its own process.
//!
//! The detection predicate and the registration below are always compiled (pure
//! file inspection, no engine dependency); the engine itself lives in the
//! feature-gated `imp` module.

use std::path::Path;

use tonic::Status;

use crate::backend::{Backend, BackendRegistration, Contract};
use crate::config::ModelConfig;

/// A C++ model folder: the tensor contract plus something to run — `main.cpp`
/// (compiled at startup), a `build.sh` that produces the executable, or a
/// prebuilt `model` binary.
fn detect(model_dir: &Path) -> bool {
    model_dir.join("model_inference.textproto").is_file()
        && (model_dir.join("main.cpp").is_file()
            || model_dir.join("build.sh").is_file()
            || model_dir.join("model").is_file())
}

fn load(model_dir: &Path, model_cfg: &ModelConfig) -> Result<(Box<dyn Backend>, Contract), Status> {
    #[cfg(feature = "cpp")]
    {
        imp::CppBackend::load(model_dir, model_cfg)
    }
    #[cfg(not(feature = "cpp"))]
    {
        let _ = model_dir;
        Err(crate::backend::missing_feature(
            &model_cfg.name,
            "C++ (main.cpp)",
            "cpp",
        ))
    }
}

inventory::submit! {
    BackendRegistration {
        name: "cpp",
        version: "0.1.0",
        aliases: &[],
        describes: "main.cpp (or a build.sh / prebuilt model) + model_inference.textproto",
        auto_detect: true,
        detect,
        load,
    }
}

#[cfg(feature = "cpp")]
mod imp;
#[cfg(feature = "cpp")]
mod subprocess;
