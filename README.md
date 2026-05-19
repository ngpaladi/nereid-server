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
Each model must be a folder under `ml-backends/<model_name>/` with:
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
input_shape: [1, 16]
# also supported:
# input_shape: 1
# input_shape: 16
# input_shape: "1,16"
# input_shape: "1x16"
```

## `nereid.yaml` configuration (required)
`nereid.yaml` is loaded at server startup from the repository root (`./nereid.yaml`).
If this file is missing or invalid, the server does not start.

It is required because the server uses it to:
- decide which models are exposed via gRPC (`ViewModels` and `Checkpoint`)
- choose execution device per model (`cpu` or `cuda`)
- size each model request queue (`queue_capacity`)
- choose the server bind address (`server.bind_addr`)

Example:
```yaml
server:
  bind_addr: "[::1]:50051"

models:
  - name: "model3"
    device: "cuda"      # "cpu" or "cuda"
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

## Python mock ED client
The Python client is a small YAML-configured mock ED producer runner. It creates random float32 input tensors and sends them to `Nereid/Checkpoint`.

Install dependencies:
```bash
python3 -m venv python-client/.venv
source python-client/.venv/bin/activate
pip install -r python-client/requirements.txt
```

Create a local config:
```bash
cp python-client/client.yaml.example python-client/client.yaml
```

Example config (all values are explained in client.yaml.example):
```yaml
host: "[::1]"
port: 50051
model: "model3"
shape: [1, 16]
producers: 1
inputs_per_producer: 1
chunk_bytes: 65536
sleep_seconds: 0
```

Run from the repository root:
```bash
python3 python-client/client.py
```

To use a different config file:
```bash
python3 python-client/client.py --config path/to/client.yaml
```

Client behavior:
- starts `producers` separate OS processes
- opens one persistent gRPC channel per producer
- sends `inputs_per_producer` separate tensors per producer
- sends one `Checkpoint` stream per tensor
- prints status only, without decoding output tensors

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
