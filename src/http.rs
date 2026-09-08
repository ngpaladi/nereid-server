//! The plain-HTTP surface, served on `server.http_addr` (Triton's convention:
//! port 8002), separate from the gRPC bind address:
//!
//! - `GET /metrics` — Triton's inference request metrics for Prometheus (see
//!   `crate::metrics`).
//! - `GET /v2/health/live`, `GET /v2/health/ready` — the KServe v2 HTTP health
//!   endpoints, so a Kubernetes liveness/readiness probe can be pointed at
//!   nereid exactly as it would be at Triton. Both answer `200` with an empty
//!   body once the server is up: nereid loads every configured model before it
//!   starts listening (a model that fails to load fails startup), so a running
//!   server is a ready server — the same answer the gRPC `ServerLive` and
//!   `ServerReady` RPCs give.
//! - `GET /v2/models/{name}/ready` and `GET /v2/models/{name}/versions/{version}/ready`
//!   — the per-model readiness probe: `200` for a configured model (and its
//!   single version, `"1"`), else `400`, as Triton answers.
//!
//! HTTP/1.1 via axum, which is already compiled in as tonic's router. Nothing
//! here does inference: the HTTP/REST `/v2` inference mirror remains deferred.

use std::sync::Arc;

use axum::Router;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::http::header::CONTENT_TYPE;
use axum::routing::get;
use tokio::net::TcpListener;

use crate::backend::ModelManager;
use crate::metrics::TEXT_FORMAT;
use crate::triton::version_available;

/// The routes above, over the shared `ModelManager`.
pub fn router(model_manager: Arc<ModelManager>) -> Router {
    Router::new()
        .route("/metrics", get(metrics))
        .route("/v2/health/live", get(live))
        .route("/v2/health/ready", get(ready))
        .route("/v2/models/{name}/ready", get(model_ready))
        .route(
            "/v2/models/{name}/versions/{version}/ready",
            get(model_version_ready),
        )
        .with_state(model_manager)
}

/// Serve the HTTP surface on an already-bound listener until shut down.
pub async fn serve_on(
    listener: TcpListener,
    model_manager: Arc<ModelManager>,
) -> std::io::Result<()> {
    axum::serve(listener, router(model_manager)).await
}

async fn metrics(
    State(manager): State<Arc<ModelManager>>,
) -> ([(axum::http::HeaderName, &'static str); 1], String) {
    ([(CONTENT_TYPE, TEXT_FORMAT)], manager.metrics().render())
}

async fn live() -> StatusCode {
    StatusCode::OK
}

async fn ready() -> StatusCode {
    StatusCode::OK
}

async fn model_ready(
    State(manager): State<Arc<ModelManager>>,
    Path(name): Path<String>,
) -> StatusCode {
    readiness(&manager, &name, "")
}

async fn model_version_ready(
    State(manager): State<Arc<ModelManager>>,
    Path((name, version)): Path<(String, String)>,
) -> StatusCode {
    readiness(&manager, &name, &version)
}

/// The KServe v2 readiness answer for a model (and optional version): `200`
/// when it is served, `400` when not — the code Triton returns.
fn readiness(manager: &ModelManager, name: &str, version: &str) -> StatusCode {
    if manager.is_configured(name) && version_available(version) {
        StatusCode::OK
    } else {
        StatusCode::BAD_REQUEST
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelConfig, ModelDevice, ServerConfig, ServerSection};
    use std::net::SocketAddr;
    use std::path::PathBuf;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Spawn the HTTP surface over a `ModelManager` serving the committed
    /// `model3` fixture, on an ephemeral port.
    async fn spawn_http() -> (SocketAddr, Arc<ModelManager>) {
        let ml_backends = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ml-backends");
        assert!(
            ml_backends.join("model3").is_dir(),
            "model3 fixture missing"
        );
        let config = ServerConfig {
            server: ServerSection {
                bind_addr: "127.0.0.1:0".to_string(),
                ml_backends_path: ml_backends.to_string_lossy().into_owned(),
                http_addr: None,
            },
            models: vec![ModelConfig {
                name: "model3".to_string(),
                device: ModelDevice::Cpu,
                queue_capacity: 4,
                backend: None,
                signature: None,
            }],
        };
        let manager =
            Arc::new(ModelManager::from_config(&config).expect("model manager should build"));
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().unwrap();
        tokio::spawn(serve_on(listener, manager.clone()));
        (addr, manager)
    }

    /// Talk plain HTTP/1.1 the way a Prometheus scraper or a kubelet probe
    /// does; returns `(status line, headers, body)`.
    async fn http_get(addr: SocketAddr, path: &str) -> (String, String, String) {
        let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
        stream
            .write_all(
                format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
                    .as_bytes(),
            )
            .await
            .expect("write request");
        let mut response = Vec::new();
        stream
            .read_to_end(&mut response)
            .await
            .expect("read response");
        let response = String::from_utf8(response).expect("utf-8 response");
        let (head, body) = response.split_once("\r\n\r\n").expect("headers and body");
        let (status, headers) = head.split_once("\r\n").unwrap_or((head, ""));
        (
            status.to_string(),
            headers.to_ascii_lowercase(),
            body.to_string(),
        )
    }

    #[tokio::test]
    async fn metrics_endpoint_serves_prometheus_text() {
        let (addr, manager) = spawn_http().await;
        manager.begin_request("model3").unwrap().succeed();

        let (status, headers, body) = http_get(addr, "/metrics").await;
        assert_eq!(status, "HTTP/1.1 200 OK");
        assert!(
            headers.contains("content-type: text/plain; version=0.0.4; charset=utf-8"),
            "{headers}"
        );
        assert!(
            body.contains("nv_inference_request_success{model=\"model3\",version=\"1\"} 1"),
            "{body}"
        );

        let (status, _, _) = http_get(addr, "/nope").await;
        assert_eq!(status, "HTTP/1.1 404 Not Found");
    }

    /// The KServe v2 health endpoints a Kubernetes probe is pointed at: live
    /// and ready answer 200 for a running server; a configured model (and its
    /// version "1") is ready, anything else is 400.
    #[tokio::test]
    async fn kserve_health_endpoints() {
        let (addr, _manager) = spawn_http().await;

        for path in ["/v2/health/live", "/v2/health/ready"] {
            let (status, _, body) = http_get(addr, path).await;
            assert_eq!(status, "HTTP/1.1 200 OK", "{path}");
            assert!(body.is_empty(), "{path} body must be empty, got {body:?}");
        }

        for path in [
            "/v2/models/model3/ready",
            "/v2/models/model3/versions/1/ready",
        ] {
            let (status, _, _) = http_get(addr, path).await;
            assert_eq!(status, "HTTP/1.1 200 OK", "{path}");
        }
        for path in [
            "/v2/models/ghost/ready",
            "/v2/models/model3/versions/2/ready",
        ] {
            let (status, _, _) = http_get(addr, path).await;
            assert_eq!(status, "HTTP/1.1 400 Bad Request", "{path}");
        }
    }
}
