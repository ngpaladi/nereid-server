# syntax=docker/dockerfile:1
#
# nereid-server container image.
#
# Two stages. The builder produces a *relocatable* bundle with
# `./build.sh --release --link bundled`: libtorch's shared objects are copied
# next to the binary and the rpath is patched to `$ORIGIN/lib`, so the binary
# runs with no `LD_LIBRARY_PATH` (see docs/building.md). The runtime stage then
# carries only that bundle plus what the Python backend needs at runtime —
# neither the Rust toolchain nor the multi-GB `target/` tree survives into it.
#
# The image ships the default backends (TorchScript `.pt` + Python `main.py`),
# which is what makes it large (~600 MB): the bundled libtorch is ~470 MB of
# that. Pass BUILD_ARGS to build a leaner or different set, e.g.
#   docker build --build-arg BUILD_ARGS="--backends onnx" .
# for an ONNX-only server that links no libtorch at all.
#
# Runtime contract: the server reads `./nereid.yaml` from its working directory
# and resolves `server.ml_backends_path` relative to it, so mount both into
# /nereid:
#   docker run --rm -p 50051:50051 -p 8002:8002 -v "$PWD:/nereid" ghcr.io/ngpaladi/nereid-server
# and set `bind_addr: "[::]:50051"` (plus `http_addr: "[::]:8002"` for the
# Prometheus /metrics and /v2/health probe endpoints) in that config — the example config's `[::1]` is
# loopback-only and is unreachable from outside the container.

ARG RUST_VERSION=1
ARG DEBIAN_SUITE=bookworm

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
FROM rust:${RUST_VERSION}-${DEBIAN_SUITE} AS builder

# patchelf — `--link bundled` uses it to set the `$ORIGIN/lib` rpath; without it
#            build.sh falls back to an LD_LIBRARY_PATH wrapper script and the
#            bundle stops being a single self-contained executable.
# curl/unzip — required by `--fetch-libtorch`, which downloads the official
#            libtorch and verifies its pinned sha256 instead of letting `tch`
#            pull it in opaquely.
# protoc is deliberately absent: build.rs uses `protoc-bin-vendored`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends patchelf curl unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Extra flags forwarded verbatim to build.sh (backend selection, device, ...).
ARG BUILD_ARGS=""

COPY . .

RUN ./build.sh --release --link bundled --fetch-libtorch ${BUILD_ARGS}

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM debian:${DEBIAN_SUITE}-slim AS runtime

# python3 + python3-venv — the Python backend builds a per-model virtualenv at
#   load time and runs `main.py` inside it.
# ca-certificates — that venv's `pip install -r requirements.txt` needs TLS roots.
# libgomp1 — OpenMP runtime libtorch links against (belt and braces: the bundle
#   usually carries its own copy).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-venv ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --create-home --home-dir /home/nereid nereid \
    && mkdir -p /nereid \
    && chown nereid:nereid /nereid

# The self-contained bundle: /opt/nereid/grpc-test + /opt/nereid/lib/*.so.
COPY --from=builder /src/dist/grpc-test /opt/nereid

# These are overridden by the labels the release workflow injects; they exist so
# a plain `docker build` still produces a labelled image.
LABEL org.opencontainers.image.source="https://github.com/ngpaladi/nereid-server" \
      org.opencontainers.image.description="nereid-server — a nifty little Rust inference server (gRPC, KServe v2 compatible)" \
      org.opencontainers.image.licenses="MIT"

# Unprivileged by default. A mounted model directory must therefore be writable
# by uid 10001 (the Python backend creates `venv/` inside it) — or run the
# container with `--user "$(id -u):$(id -g)"` to match the host owner.
USER nereid

# The config contract: `nereid.yaml` and the model folders are read from here.
WORKDIR /nereid

# gRPC, and the optional HTTP surface (server.http_addr): /metrics, /v2/health/*.
EXPOSE 50051 8002

ENTRYPOINT ["/opt/nereid/grpc-test"]
