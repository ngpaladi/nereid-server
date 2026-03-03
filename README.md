Goal is to build a simple replacement for nvidia triton server.
Right now this only contains a simple Rust gRPC server built with `tonic` and `prost`.

## What it does
- Exposes `Greeter/SayHello` to return a greeting message.
- Exposes `Greeter/AddTensors` to add two tensors element-wise.

Proto definition: `proto/helloworld.proto`.

## Installation
Prerequisites:
- Rust (stable) with Cargo installed: [Rust Installation](https://rust-lang.org/tools/install/)
- `protoc`: [Protocol Buffers Installation](https://protobuf.dev/installation/)

Install dependencies and build:
```bash
cargo build
```

## Run
```bash
cargo run
```

Server starts on `127.0.0.1:50051`

## Project structure
- `src/main.rs`: gRPC service implementation and server bootstrap.
- `build.rs`: compiles `.proto` files at build time.
- `proto/helloworld.proto`: service and message definitions.
