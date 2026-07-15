// Parses and validates the server YAML configuration.
// Defines the backend-agnostic server/model config types (device resolution and
// any backend-specific handling live in the `backend` modules) and enforces
// basic checks (required fields, non-empty names, unique model names, and
// positive queue capacity).

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use serde::Deserialize;

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
    // Read by the device-aware backends (torch/onnx/tensorflow); a build with
    // only the Python backend never consults it.
    #[cfg_attr(
        not(any(feature = "torch", feature = "onnx", feature = "tensorflow")),
        allow(dead_code)
    )]
    pub device: ModelDevice,
    pub queue_capacity: usize,
    /// Optional explicit backend selector — the `name` a backend registers under
    /// (e.g. `"torch"`, `"onnx"`). When set, it decides the model's backend (and
    /// disambiguates a folder that contains files for more than one — e.g. a
    /// `.pt` alongside `main.py`). When absent, the backend is auto-detected from
    /// the folder contents. Validated against the backend registry at load time,
    /// so adding a backend needs no change here.
    #[serde(default)]
    pub backend: Option<String>,
    /// TensorFlow SavedModel signature key. Only meaningful for the `tensorflow`
    /// backend; defaults to `"serving_default"` when absent.
    #[serde(default)]
    #[cfg_attr(not(feature = "tensorflow"), allow(dead_code))]
    pub signature: Option<String>,
}

impl ModelConfig {
    /// The TensorFlow signature to serve, defaulting to `"serving_default"`.
    #[cfg_attr(not(feature = "tensorflow"), allow(dead_code))]
    pub fn tf_signature(&self) -> &str {
        self.signature.as_deref().unwrap_or("serving_default")
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ModelDevice {
    Cpu,
    Cuda(usize),
}

impl<'de> Deserialize<'de> for ModelDevice {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        match raw.as_str() {
            "cpu" => Ok(ModelDevice::Cpu),
            "cuda" => Ok(ModelDevice::Cuda(0)),
            other => other
                .strip_prefix("cuda:")
                .and_then(|index| index.parse::<usize>().ok())
                .map(ModelDevice::Cuda)
                .ok_or_else(|| {
                    serde::de::Error::custom(format!(
                        "invalid device '{other}': expected \"cpu\", \"cuda\", or \"cuda:<index>\""
                    ))
                }),
        }
    }
}

impl ModelDevice {
    /// The CUDA device index for GPU-capable backends, or `None` for CPU. Device
    /// resolution and any libtorch/CUDA availability checks live in each backend
    /// module (e.g. `backend::torch`), keeping this core config type
    /// backend-agnostic.
    #[cfg_attr(not(any(feature = "onnx", feature = "tensorflow")), allow(dead_code))]
    pub fn cuda_index(self) -> Option<usize> {
        match self {
            Self::Cpu => None,
            Self::Cuda(index) => Some(index),
        }
    }
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
    use super::{ModelDevice, ServerConfig, validate_server_config};

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
    fn config_parses_cuda_with_explicit_device_index() {
        let config = parse_config(
            r#"
server:
  bind_addr: "[::1]:50051"
  ml_backends_path: "ml-backends"
models:
  - name: "model3"
    device: "cuda:1"
    queue_capacity: 8
"#,
        )
        .expect("config should parse");

        assert!(matches!(config.models[0].device, ModelDevice::Cuda(1)));
    }

    #[test]
    fn config_defaults_bare_cuda_to_device_zero() {
        let config = parse_config(
            r#"
server:
  bind_addr: "[::1]:50051"
  ml_backends_path: "ml-backends"
models:
  - name: "model3"
    device: "cuda"
    queue_capacity: 8
"#,
        )
        .expect("config should parse");

        assert!(matches!(config.models[0].device, ModelDevice::Cuda(0)));
    }

    #[test]
    fn config_rejects_non_numeric_cuda_index() {
        let result = parse_config(
            r#"
server:
  bind_addr: "[::1]:50051"
  ml_backends_path: "ml-backends"
models:
  - name: "model3"
    device: "cuda:abc"
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
