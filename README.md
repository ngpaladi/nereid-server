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

## Dummy ED producers
Dummy ED producers act as test clients for the server. They create random input tensors and call `python-client/client.py` repeatedly.

Files:
- `scripts/spawn_edproducers.py`: can spawn one or multiple producer processes

Run from repository root:
```bash
python3 scripts/spawn_edproducers.py
```

Producer config is currently set by constants in `scripts/spawn_edproducers.py`:
- `PRODUCERS`: number of parallel producer processes
- `ITERATIONS_PER_PRODUCER`: number of requests (tensors) each producer sends
- `MODEL`: target `model_name` sent to server (must exist in `nereid.yaml`)
- `HOST`/`PORT`: server address

Iterations:
- one iteration = one generated tensor + one `Checkpoint` request
- total requests sent = `PRODUCERS * ITERATIONS_PER_PRODUCER`
- example: `PRODUCERS=4` and `ITERATIONS_PER_PRODUCER=25` sends 100 requests in total

Producer role:
- simulate concurrent clients
- verify routing to a configured model
- stress queueing/backpressure behavior using `queue_capacity`

## Client installation (`grpcurl`)

See the official [`grpcurl` installation guide](https://github.com/fullstorydev/grpcurl#installation).

## Client usage
Health check:
```bash
grpcurl -plaintext -import-path ./proto -proto inference.proto -d '{}' '[::1]:50051' inference.Nereid/HealthCheck
```

View models:
```bash
grpcurl -plaintext -import-path ./proto -proto inference.proto -d '{}' '[::1]:50051' inference.Nereid/ViewModels
```

Checkpoint (streaming):
Note: Right now the data and output files are ignored here
```bash
grpcurl -plaintext \
  -import-path ./proto \
  -proto inference.proto \
  -d '{
        "model_name": "model2",
        "data": "input.csv",
        "output_file": "out.txt"
      }' \
  '[::1]:50051' \
  inference.Nereid/Checkpoint
```

## Project structure
- `src/main.rs`: gRPC service implementation and server bootstrap.
- `build.rs`: compiles `.proto` files at build time.
- `proto/inference.proto`: service and message definitions.
