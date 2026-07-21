//! The compile-time C++ engine: look the model up by name in the `cxx-models`
//! registry, then run it in process, serialized behind a `Mutex`. It implements
//! the same [`Backend`] trait as every other engine. See the module docs in
//! `mod.rs` for why this backend is declaration-only.

use std::path::Path;
use std::sync::Mutex;

use tonic::Status;

use crate::backend::{Backend, Contract, Tensor};
use crate::config::ModelConfig;

pub struct CxxBackend {
    // `cxx_models::CxxModel` is `Send` but not `Sync`; the `Mutex` serializes
    // inference (as the dedicated worker thread did before) and makes the
    // backend shareable across the async runtime.
    model: Mutex<Box<dyn cxx_models::CxxModel>>,
}

impl CxxBackend {
    pub fn load(
        model_dir: &Path,
        model_cfg: &ModelConfig,
    ) -> Result<(Box<dyn Backend>, Contract), Status> {
        let contract = Contract::parse(model_dir)?;
        if !contract.has_input() {
            return Err(Status::failed_precondition(format!(
                "C++ model '{}' must declare input_shape in model_inference.textproto",
                model_cfg.name
            )));
        }
        // The C++ is compiled in and keyed by the model name; the folder just
        // supplies the contract.
        let model = cxx_models::create(&model_cfg.name).ok_or_else(|| {
            Status::failed_precondition(format!(
                "no compiled-in C++ model named '{}'. Register it in the cxx-models crate and \
                 rebuild with `--features cxx`.",
                model_cfg.name
            ))
        })?;

        Ok((
            Box::new(CxxBackend {
                model: Mutex::new(model),
            }),
            contract,
        ))
    }
}

impl Backend for CxxBackend {
    fn platform(&self) -> &'static str {
        "nereid_cxx"
    }

    fn infer(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, Status> {
        let input = inputs.into_iter().next().ok_or_else(|| {
            Status::invalid_argument("cxx model expects exactly one input tensor")
        })?;
        let model = self.model.lock().unwrap_or_else(|e| e.into_inner());
        let (shape, dtype, data) = model
            .run(&input.dtype, &input.shape, &input.data)
            .map_err(|e| Status::internal(format!("cxx model failed: {e}")))?;
        Ok(vec![Tensor {
            name: "output".to_string(),
            shape,
            dtype,
            data,
        }])
    }
}
