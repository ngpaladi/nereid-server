//! Python (`main.py`) backend. Each model runs in a per-model virtualenv (built
//! at startup from `requirements.txt`) as a subprocess per request, over the
//! stdin / framed-`NEREID_OUTPUT_PATH` tensor contract. Out of process, so a
//! crash in the model is contained.

use std::path::Path;

use tonic::Status;

use crate::backend::{Backend, BackendRegistration, Contract};
use crate::config::ModelConfig;

/// A `main.py` + `requirements.txt` folder.
fn detect(model_dir: &Path) -> bool {
    model_dir.join("main.py").is_file() && model_dir.join("requirements.txt").is_file()
}

fn load(model_dir: &Path, model_cfg: &ModelConfig) -> Result<(Box<dyn Backend>, Contract), Status> {
    #[cfg(feature = "python")]
    {
        imp::PythonBackend::load(model_dir, model_cfg)
    }
    #[cfg(not(feature = "python"))]
    {
        let _ = model_dir;
        Err(crate::backend::missing_feature(
            &model_cfg.name,
            "Python (main.py)",
            "python",
        ))
    }
}

inventory::submit! {
    BackendRegistration {
        name: "python",
        version: "0.1.0",
        aliases: &[],
        describes: "main.py + requirements.txt",
        auto_detect: true,
        detect,
        load,
    }
}

#[cfg(feature = "python")]
mod imp;
