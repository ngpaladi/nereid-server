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
    stdout/stderr lines as output chunks. If the folder also includes a
    `model_inference.textproto`, the server validates the request tensor against that contract
    and pipes it into `main.py` on stdin (see [Python input contract](#python-input-contract)).
  - `model_inference.textproto` + `.pt` model -> runs Rust inference for that model and streams
    the output tensor.

## Model folder contract
Each model must be a folder under `<server.ml_backends_path>/<model_name>/` with:
- `requirements.txt`
- `main.py`
- `model_inference.textproto` (optional — adds input validation, see below)
OR
- `model_inference.textproto` (for Rust `.pt` inference models)
- `.pt` model

By default the backend is auto-detected: a Python model by `main.py` + `requirements.txt`,
a Rust model by `model_inference.textproto` + a `.pt` file. If a folder matches both or
neither, auto-detection fails at startup. To ship files for both backends in one folder
(e.g. a `.pt` model alongside `main.py`), set `backend: "python"` or `backend: "rust"` on
that model in `nereid.yaml`; the declared backend is authoritative and only its own
required files are checked.

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

`input_shape` excludes batch size. If `max_batch_size` is greater than `0`, the request shape
from the client may include a leading batch dimension, which the server enforces is at most
`max_batch_size`. A request that omits the batch dimension (sending just the `input_shape` rank)
is accepted and auto-expanded to batch size 1 — e.g. with `input_shape: [16]` a request shape of
`[16]` is treated as `[1, 16]`. If `max_batch_size` is `0` or omitted, the request shape must
match the `input_shape` rank directly. Fixed dimensions must match exactly; `-1` dimensions may
be any positive value.

### Python input contract
A Python model folder may include a `model_inference.textproto` (the same format as Rust models).
When present, the server validates each request tensor against it — rejecting shape/rank/batch
mismatches with an `InvalidArgument` status *before* launching `main.py` — and then delivers the
validated tensor to the process:
- **stdin**: the raw tensor bytes, little-endian `float32`, in row-major order.
- **`NEREID_INPUT_SHAPE`**: the (batch-normalized) shape as comma-separated dimensions, e.g. `1,16`.
- **`NEREID_INPUT_DTYPE`**: the element dtype, currently always `float32`.

A minimal `main.py` reading the tensor (no third-party dependencies):
```python
import os, struct, sys

shape = [int(d) for d in os.environ["NEREID_INPUT_SHAPE"].split(",")]
raw = sys.stdin.buffer.read()
values = struct.unpack("<%df" % (len(raw) // 4), raw)
# ... reshape `values` to `shape` and run inference ...
```
Without a `model_inference.textproto`, a Python model receives no tensor input (stdin is closed)
and the request stream is simply drained — the original behavior. Note that *output* from a
Python model is always its stdout/stderr text; the contract governs input only.

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

### Using an external libtorch installation
During a Cargo build, `tch-rs` defaults to downloading a CPU-only build of libtorch, unless
certain environment variables are set (see
[the getting started notes](https://github.com/LaurentMazare/tch-rs#getting-started)). It may
be desirable to instead use an external installation of libtorch, by setting the `LIBTORCH`
environment variable (see
[Libtorch Manual Install](https://github.com/LaurentMazare/tch-rs#libtorch-manual-install)) to
the unpacked location prior to running `cargo build`. Be aware that this must use the precise
version of libtorch that `tch-rs` expects (v2.5.1 for `tch-rs` v0.18.1), and must use the
cxx11 ABI (PyTorch additionally provides builds with the pre-cxx11 ABI for older versions,
including 2.5.1), and must have been built against a CUDA version compatible with the system's
CUDA libraries and drivers.

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
