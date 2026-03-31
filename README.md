# nereid-server
A nifty little rust inference server

Goal is to build a simple replacement for an inference server.
Right now, this project contains a Rust gRPC server built with `tonic`.
It can handle client requests and spawn a python process to run a sample model accordingly

## Proto
- `proto/inference.proto`

## Server behavior
- `Health/HealthCheck` returns status `ok`.
- `Sonic/ViewModels` returns folder names under `ml-backends` (or `["No models found"]` when empty).
- `Sonic/Checkpoint` runs `<model>/main.py` using that model's existing `venv` Python and streams output chunks.

## Model folder contract
Each model must be a folder under `ml-backends/<model_name>/` with:
- `requirements.txt`
- `main.py`

On server startup:
- If `venv/` exists for a model, it is reused.
- If `venv/` is missing, server creates it and installs from `requirements.txt`.
- If `requirements.txt` is missing, startup fails with an error.

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

Server listens on `[::1]:50051`.

## Client installation (`grpcurl`)

See the official [`grpcurl` installation guide](https://github.com/fullstorydev/grpcurl#installation).

## Client usage
Health check:
```bash
grpcurl -plaintext -import-path ./proto -proto inference.proto -d '{}' '[::1]:50051' inference.Health/HealthCheck
```

View models:
```bash
grpcurl -plaintext -import-path ./proto -proto inference.proto -d '{}' '[::1]:50051' inference.Sonic/ViewModels
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
  inference.Sonic/Checkpoint
```

## Project structure
- `src/main.rs`: gRPC service implementation and server bootstrap.
- `build.rs`: compiles `.proto` files at build time.
- `proto/inference.proto`: service and message definitions.
