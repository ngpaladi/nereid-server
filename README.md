# nereid-server
A nifty little rust inference server

Goal is to build a simple replacement for an inference server.
Right now, this project contains a Rust gRPC server built with `tonic`.
It can handle client requests and spawn a python process to run a sample model accordingly

## Proto
- `proto/inference.proto`

## Server behavior
- `Nereid/HealthCheck` returns status `ok`.
- `Nereid/ViewModels` returns model names configured in `nereid.yaml`.
- `Nereid/Checkpoint` only accepts `model_name` values configured in `nereid.yaml`, runs Rust inference for that model, and streams output chunks.

Note: Functionality for models containing main.py is temporarily disabled right now.

## Model folder contract
Each model must be a folder under `<server.ml_backends_path>/<model_name>/` with:
- `requirements.txt`
- `main.py`
OR
- `model_inference.textproto` (for Rust `.pt` inference models)
- `.pt` model

On server startup (if using main.py):
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
- choose execution device per model (`cpu` or `cuda`)
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
    device: "cpu"       # "cpu" or "cuda"
    queue_capacity: 16  # must be > 0
```

## Server installation
Prerequisites:
- Rust (stable) with Cargo: [Rust Installation](https://rust-lang.org/tools/install/)
- `protoc`: [Protocol Buffers Installation](https://protobuf.dev/installation/)

Build:
```bash
cargo build
```

Run:
```bash
cargo run
```

Server binds to `server.bind_addr` from `nereid.yaml` (default example: `[::1]:50051`).
Model folders are loaded from `server.ml_backends_path` (default example: `ml-backends`).
This folder must exist in the project root and contain all ML model folders.

## Python mock ED client
The Python client is a YAML-configured mock ED producer runner. It supports fixed shapes, a list of possible shapes, and random shape generation for variable-shape models.

See `python-client/README.md` for installation, configuration, and examples.

## grpcurl installation

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
- `build.rs`: compiles `.proto` files at build time.
- `proto/inference.proto`: service and message definitions.
- `python-client/client.py`: YAML-configured mock ED client.
