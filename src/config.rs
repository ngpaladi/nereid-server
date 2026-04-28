use std::collections::HashSet;
use std::fs;
use std::path::Path;

use serde::Deserialize;
use tch::Device;
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
                if tch::Cuda::is_available() {
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
    use super::{ServerConfig, validate_server_config};

    fn parse_config(raw: &str) -> Result<ServerConfig, serde_yaml::Error> {
        serde_yaml::from_str(raw)
    }

    #[test]
    fn config_parses_when_valid() {
        let config = parse_config(
            r#"
server:
  bind_addr: "[::1]:50051"
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
"#,
        );

        assert!(result.is_err());
    }

    #[test]
    fn config_requires_bind_addr() {
        let result = parse_config(
            r#"
server: {}
models:
  - name: "model3"
    device: "cpu"
    queue_capacity: 16
"#,
        );

        assert!(result.is_err());
    }
}
