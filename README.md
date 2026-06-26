# nereid-server
A nifty little rust inference server

Goal is to build a simple replacement for a ML inference server.
Right now, this project contains a Rust gRPC server built with `tonic`.
The server supports PyTorch models exported in TorchScript (.pt) format, as well as
arbitrary Python scripts (`main.py`) run inside a per-model virtualenv.

## Proto
- `proto/inference.proto`

## Server behavior
- `Nereid/HealthCheck` returns status `ok`.
- `Nereid/ViewModels` returns model names configured in `nereid.yaml`.
- `Nereid/Checkpoint` only accepts `model_name` values configured in `nereid.yaml`. The server
  detects each model's backend kind from its folder contents:
  - `main.py` + `requirements.txt` -> runs `main.py` in that model's venv and streams its
    stdout/stderr lines as output chunks.
  - `model_inference.textproto` + `.pt` model -> runs Rust inference for that model and streams
    the output tensor.

## Model folder contract
Each model must be a folder under `<server.ml_backends_path>/<model_name>/` with:
- `requirements.txt`
- `main.py`
OR
- `model_inference.textproto` (for Rust `.pt` inference models)
- `.pt` model

A model folder must satisfy exactly one of these contracts; the server fails to start if a
configured model's folder matches both or neither.

On server startup (for models using main.py):
- If `venv/` exists for a model, it is reused.
- If `venv/` is missing, server creates it and installs from `requirements.txt`.
- If `requirements.txt` is missing, startup fails with an error.

`model_inference.textproto` shape examples:
```text
input_shape: [16]
max_batch_size: 8

# A request shape of [4, 16] is accepted: batch size 4, input shape [16].
# A request shape of [9, 16] is rejected because it exceeds max_batch_size.

# input_shape: [-1, 16]  # -1 means this dimension is client-provided.
```

`input_shape` excludes batch size. If `max_batch_size` is greater than `0`, the request
shape from the client must include a leading batch dimension and the server enforces that it is at most
`max_batch_size`. If `max_batch_size` is `0` or omitted, the request shape must match the
`input_shape` rank directly. Fixed dimensions must match exactly; `-1` dimensions may be
any positive value.

## `nereid.yaml` configuration (required)
`nereid.yaml` is loaded at server startup from the repository root (`./nereid.yaml`).
If this file is missing or invalid, the server does not start.

Create a local config from the versioned example:
```bash
cp nereid.yaml.example nereid.yaml
```

`nereid.yaml` is ignored by Git so each developer can keep local model and device settings.

It is required because the server uses it to:
- decide which models are exposed via gRPC (`ViewModels` and `Checkpoint`)
- choose execution device per model (`cpu`, `cuda`, or `cuda:<index>` on multi-GPU systems)
- size each model request queue (`queue_capacity`)
- choose the server bind address (`server.bind_addr`)
- find model folders (`server.ml_backends_path`)

Example (`nereid.yaml.example`):
```yaml
server:
  bind_addr: "[::1]:50051"
  ml_backends_path: "ml-backends"

models:
  - name: "model3"
    device: "cpu"       # "cpu", "cuda" (GPU 0), or "cuda:<index>" (e.g. "cuda:1")
    queue_capacity: 16  # must be > 0
```

## Server installation
Prerequisites:
- Rust (stable) with Cargo: [Rust Installation](https://rust-lang.org/tools/install/)

Build:
```bash
cargo build
```

Run:
```bash
cargo run
```

Run (if using CUDA as device):
```bash
TORCH_CUDA_VERSION=cu124 cargo run
```

Server binds to `server.bind_addr` from `nereid.yaml` (default example: `[::1]:50051`).
Model folders are loaded from `server.ml_backends_path` (default example: `ml-backends`).
This folder must exist in the project root and contain all ML model folders.

### Running the built binary directly (not via `cargo run`)
`cargo run`/`cargo test` automatically set `LD_LIBRARY_PATH` so the binary can find the
libtorch shared libraries (`libtorch_cpu.so`, etc.) downloaded by `torch-sys`. If you run a
built binary directly (`./target/debug/grpc-test`, a Docker container, systemd, etc.), you
will hit:
```
error while loading shared libraries: libtorch_cpu.so: cannot open shared object file
```
Fix by pointing `LD_LIBRARY_PATH` at the extracted libtorch `lib/` directory before running.
A cargo clean/rebuild can leave more than one `torch-sys-*` build directory on disk
(stale ones from prior profiles or interrupted builds), so pick the directory with the
largest (i.e. fully extracted, not truncated) `libtorch_cpu.so`:
```bash
export LD_LIBRARY_PATH=$(dirname "$(find target -name libtorch_cpu.so -printf '%s %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)"):$LD_LIBRARY_PATH
./target/debug/grpc-test
```
This is a known `tch-rs`/`torch-sys` characteristic (libtorch ships as shared libraries with
no rpath embedded in the final binary), not specific to this project. See the
[`tch-rs` FAQ](https://github.com/LaurentMazare/tch-rs#faq) for background.

## Python mock ED client
The Python client is a YAML-configured mock ED producer runner. It supports fixed shapes, a list of possible shapes, and random shape generation for variable-shape models.

See `python-client/README.md` for installation, configuration, and examples.

## grpcurl installation (Alternate way to test server)

See the official [`grpcurl` installation guide](https://github.com/fullstorydev/grpcurl#installation).

## grpcurl usage
Health check:
```bash
grpcurl -plaintext -import-path ./proto -proto inference.proto -d '{}' '[::1]:50051' inference.Nereid/HealthCheck
```

View models:
```bash
grpcurl -plaintext -import-path ./proto -proto inference.proto -d '{}' '[::1]:50051' inference.Nereid/ViewModels
```

## Project structure
- `src/main.rs`: gRPC service implementation and server bootstrap.
- `src/model_runtime.rs`: per-model backend detection, Rust `.pt` inference workers.
- `src/python_backend.rs`: venv setup and `main.py` process streaming for Python models.
- `build.rs`: compiles `.proto` files at build time.
- `proto/inference.proto`: service and message definitions.
- `python-client/client.py`: YAML-configured mock ED client.
