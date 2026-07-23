//! The C++ engine: compile a model's `main.cpp` at startup, then run it as a
//! child process per request. It implements the same [`Backend`] trait as every
//! other engine, so dispatch, backpressure, and the `Checkpoint` stream come
//! from the core with no per-backend plumbing. See the module docs in `mod.rs`
//! for the folder shape and the tensor contract.

use std::path::{Path, PathBuf};
use std::process::Command;

use tonic::Status;

use super::subprocess::{self, TensorInput};
use crate::backend::{Backend, Contract, Tensor};
use crate::config::ModelConfig;

pub struct CppBackend {
    model_name: String,
    model_dir: PathBuf,
}

impl CppBackend {
    pub fn load(
        model_dir: &Path,
        model_cfg: &ModelConfig,
    ) -> Result<(Box<dyn Backend>, Contract), Status> {
        // Compile main.cpp -> `model` on startup (mirrors venv setup).
        prepare_cpp_model(model_dir, &model_cfg.name)?;

        // Like Python, a C++ model's every reply is a typed tensor, so it must
        // declare output_shape in model_inference.textproto.
        if !model_dir.join("model_inference.textproto").is_file() {
            return Err(Status::failed_precondition(format!(
                "C++ model '{}' must include a model_inference.textproto declaring output_shape",
                model_cfg.name
            )));
        }
        let contract = Contract::parse(model_dir)?;
        if contract.outputs.is_empty() {
            return Err(Status::failed_precondition(format!(
                "C++ model '{}' must declare output_shape in model_inference.textproto (every reply is a typed tensor)",
                model_cfg.name
            )));
        }

        Ok((
            Box::new(CppBackend {
                model_name: model_cfg.name.clone(),
                model_dir: model_dir.to_path_buf(),
            }),
            contract,
        ))
    }
}

impl Backend for CppBackend {
    fn platform(&self) -> &'static str {
        "cpp_subprocess"
    }

    fn infer(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, Status> {
        let input = inputs.into_iter().next().map(|t| TensorInput {
            shape: t.shape,
            bytes: t.data,
            dtype: t.dtype,
        });
        let (shape, data, dtype) = run_cpp_inference(&self.model_name, &self.model_dir, input)?;
        Ok(vec![Tensor {
            name: "output".to_string(),
            shape,
            dtype,
            data,
        }])
    }
}

/// The compiled model executable inside a C++ model folder.
fn cpp_binary_path(model_dir: &Path) -> PathBuf {
    model_dir.join(if cfg!(windows) { "model.exe" } else { "model" })
}

/// Compile a C++ model's `main.cpp` into a `model` executable when one isn't
/// already present, mirroring the Python backend's venv creation. A folder that
/// ships an executable `build.sh` uses that instead (for models with extra
/// sources or link flags); a folder that ships a prebuilt `model` binary is
/// reused as-is. Called once at load.
fn prepare_cpp_model(model_dir: &Path, name: &str) -> Result<(), Status> {
    let binary = cpp_binary_path(model_dir);
    if binary.is_file() {
        return Ok(()); // reuse a shipped or previously-built binary
    }

    let build_sh = model_dir.join("build.sh");
    let output = if build_sh.is_file() {
        Command::new("bash")
            .arg("build.sh")
            .current_dir(model_dir)
            .output()
    } else {
        let main_cpp = model_dir.join("main.cpp");
        if !main_cpp.is_file() {
            return Err(Status::failed_precondition(format!(
                "C++ model '{name}' has none of: a prebuilt 'model' binary, a build.sh, or main.cpp"
            )));
        }
        Command::new("c++")
            .args(["-O2", "-std=c++17", "main.cpp", "-o", "model"])
            .current_dir(model_dir)
            .output()
    };

    let output = output.map_err(|err| {
        Status::failed_precondition(format!(
            "failed to invoke the C++ toolchain for model '{name}': {err} \
             (is a C++ compiler installed and on PATH?)"
        ))
    })?;
    if !output.status.success() {
        return Err(Status::failed_precondition(format!(
            "failed to compile C++ model '{name}':\n{}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    if !binary.is_file() {
        return Err(Status::failed_precondition(format!(
            "C++ model '{name}' build did not produce a 'model' executable"
        )));
    }
    Ok(())
}

/// Run a C++ model for one inference request and return its framed output
/// tensor. Blocking — the `ModelManager` runs it off the async runtime.
fn run_cpp_inference(
    model_name: &str,
    model_dir: &Path,
    input: Option<TensorInput>,
) -> Result<(Vec<i64>, Vec<u8>, String), Status> {
    let binary = cpp_binary_path(model_dir);
    if !binary.is_file() {
        return Err(Status::not_found(format!(
            "compiled 'model' executable not found for C++ model '{model_name}'"
        )));
    }
    let mut command = Command::new(&binary);
    command.current_dir(model_dir);
    subprocess::run_subprocess_inference(command, model_name, input)
}
