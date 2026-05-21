// Parses and validates the server YAML configuration.
// Defines server/model config types, maps configured devices to `tch::Device`,
// and enforces basic checks (required fields, non-empty names, unique model names,
// and positive queue capacity).

use std::collections::HashSet;
#[cfg(target_os = "linux")]
use std::ffi::CString;
use std::fs;
#[cfg(target_os = "linux")]
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;

use serde::Deserialize;
use tch::{Cuda, Device};
use tonic::Status;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    pub server: ServerSection,
    pub models: Vec<ModelConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerSection {
    pub bind_addr: String,
    pub ml_backends_path: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub name: String,
    pub device: ModelDevice,
    pub queue_capacity: usize,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelDevice {
    Cpu,
    Cuda,
}

impl ModelDevice {
    pub fn to_tch_device(self) -> Result<Device, Status> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            Self::Cuda => {
                preload_libtorch_cuda();

                if Cuda::is_available() {
                    Ok(Device::Cuda(0))
                } else {
                    Err(Status::failed_precondition(
                        "CUDA model configured but CUDA is not available",
                    ))
                }
            }
        }
    }
}

fn preload_libtorch_cuda() {
    #[cfg(target_os = "linux")]
    {
        for library in ["libc10_cuda.so", "libtorch_cuda.so"] {
            if let Err(err) = dlopen_library(library) {
                eprintln!("failed to preload {library}: {err}");
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn dlopen_library(library: &str) -> Result<(), String> {
    const RTLD_NOW: c_int = 2;
    const RTLD_GLOBAL: c_int = 0x100;

    let path = CString::new(library).map_err(|err| err.to_string())?;
    let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
    if handle.is_null() {
        Err(dlopen_error())
    } else {
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn dlopen_error() -> String {
    let error = unsafe { dlerror() };
    if error.is_null() {
        "unknown dlopen error".to_owned()
    } else {
        unsafe { std::ffi::CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(target_os = "linux")]
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

pub fn load_server_config(path: &Path) -> Result<ServerConfig, Box<dyn std::error::Error>> {
    let raw = fs::read_to_string(path)?;
    let config: ServerConfig = serde_yaml::from_str(&raw)?;
    validate_server_config(&config)?;
    Ok(config)
}

pub fn validate_server_config(config: &ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    if config.server.bind_addr.trim().is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "server.bind_addr must be non-empty",
        )
        .into());
    }

    if config.server.ml_backends_path.trim().is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "server.ml_backends_path must be non-empty",
        )
        .into());
    }

    if config.models.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "config must contain at least one model",
        )
        .into());
    }

    let mut seen = HashSet::with_capacity(config.models.len());
    for model in &config.models {
        if model.name.trim().is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "model name must be non-empty",
            )
            .into());
        }

        if model.queue_capacity == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("queue_capacity must be > 0 for model '{}'", model.name),
            )
            .into());
        }

        if !seen.insert(model.name.clone()) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("duplicate model name in config: '{}'", model.name),
            )
            .into());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{validate_server_config, ServerConfig};

    fn parse_config(raw: &str) -> Result<ServerConfig, serde_yaml::Error> {
        serde_yaml::from_str(raw)
    }

    #[test]
    fn config_parses_when_valid() {
        let config = parse_config(
            r#"
server:
  bind_addr: "[::1]:50051"
  ml_backends_path: "ml-backends"
models:
  - name: "model3"
    device: "cpu"
    queue_capacity: 16
"#,
        )
        .expect("config should parse");

        validate_server_config(&config).expect("config should validate");
    }

    #[test]
    fn config_rejects_duplicate_model_names() {
        let config = parse_config(
            r#"
server:
  bind_addr: "[::1]:50051"
  ml_backends_path: "ml-backends"
models:
  - name: "same"
    device: "cpu"
    queue_capacity: 16
  - name: "same"
    device: "cuda"
    queue_capacity: 8
"#,
        )
        .expect("config should parse");

        assert!(validate_server_config(&config).is_err());
    }

    #[test]
    fn config_rejects_zero_queue_capacity() {
        let config = parse_config(
            r#"
server:
  bind_addr: "[::1]:50051"
  ml_backends_path: "ml-backends"
models:
  - name: "model3"
    device: "cpu"
    queue_capacity: 0
"#,
        )
        .expect("config should parse");

        assert!(validate_server_config(&config).is_err());
    }

    #[test]
    fn config_rejects_invalid_device_value() {
        let result = parse_config(
            r#"
server:
  bind_addr: "[::1]:50051"
  ml_backends_path: "ml-backends"
models:
  - name: "model3"
    device: "metal"
    queue_capacity: 8
"#,
        );

        assert!(result.is_err());
    }

    #[test]
    fn config_requires_models_field() {
        let result = parse_config(
            r#"
server:
  bind_addr: "[::1]:50051"
  ml_backends_path: "ml-backends"
"#,
        );

        assert!(result.is_err());
    }

    #[test]
    fn config_requires_bind_addr() {
        let result = parse_config(
            r#"
server:
  ml_backends_path: "ml-backends"
models:
  - name: "model3"
    device: "cpu"
    queue_capacity: 16
"#,
        );

        assert!(result.is_err());
    }

    #[test]
    fn config_requires_ml_backends_path() {
        let result = parse_config(
            r#"
server:
  bind_addr: "[::1]:50051"
models:
  - name: "model3"
    device: "cpu"
    queue_capacity: 16
"#,
        );

        assert!(result.is_err());
    }
}
