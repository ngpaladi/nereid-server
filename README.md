# nereid-server
A nifty little rust inference server

Goal is to build a simple replacement for a ML inference server.
Right now, this project contains a Rust gRPC server built with `tonic`.
The server supports PyTorch models exported in TorchScript (.pt) format, arbitrary
Python scripts (`main.py`) run inside a per-model virtualenv, and — behind opt-in
build features — **ONNX** and **TensorFlow** models served natively in-process (see
[Native ONNX and TensorFlow backends](#native-onnx-and-tensorflow-backends)).

📖 **Documentation:** the [`docs/`](docs/) folder is a full docs site (MkDocs) covering the
architecture, backends, model contract, Triton compatibility, and the build system. Build it
locally with `pip install -r docs-requirements.txt && mkdocs build` (or `mkdocs serve`).

## Proto
- `proto/inference.proto` — the native Nereid service.
- `proto/grpc_service.proto` — the Triton-compatible KServe v2 surface (see below).

## Triton compatibility (KServe v2)
The server also exposes NVIDIA Triton's `inference.GRPCInferenceService` on the same bind
address, so a stock `tritonclient` (or any KServe v2 speaker) can drive nereid without code
changes. The package, service name, RPC names, and message field numbers in
`proto/grpc_service.proto` are vendored verbatim from
[Triton's `grpc_service.proto`](https://github.com/triton-inference-server/common/blob/main/protobuf/grpc_service.proto),
so the wire format is byte-compatible. This is **wire**-parity, not authoring-parity: models
are still written against nereid's own contracts (a `.pt` file or a `main.py`).

**Implemented RPCs:** `ServerLive`, `ServerReady`, `ModelReady`, `ServerMetadata`,
`ModelMetadata`, unary `ModelInfer`, and streaming `ModelStreamInfer`. Both backends are
servable:
- **Rust `.pt`** — single-tensor and multi-tensor (nested `input {}`/`output {}` blocks in
  the textproto); datatypes are the libtorch kinds (`FP16/32/64`, `INT8/16/32/64`, `UINT8`,
  `BOOL`, `BF16`). `FP32` and `INT64` are verified end-to-end against a stock `tritonclient`;
  tensor (de)serialization is byte-generic and unit-tested for every listed kind (including
  the 2-byte `FP16`/`BF16`).
  - For a **multi-tensor** model, the request's named inputs are bound in the order the
    textproto lists them and passed positionally to the model's `forward()` — so the
    `input {}` blocks must be ordered to match `forward(a, b, ...)`'s parameter order.
- **Python `main.py`** — every reply is a typed tensor (the same `NEREID_OUTPUT_PATH` framed
  contract the native `Nereid/Checkpoint` path uses); byte-passthrough over any fixed-width
  KServe dtype.

The request datatype must match the model's declared `data_type` (default `FP32`). nereid
serves a single implicit model version, `"1"`. **Not implemented (deferred):** Rust
`UINT16/32/64` and `BYTES`; the HTTP/REST `/v2` mirror, Prometheus metrics, and the
repository/config/statistics RPCs.

### Verifying compatibility
Wire compatibility is established by a **stock `tritonclient`** (built from Triton's own proto
stubs, not nereid's vendored copy) — not by nereid's own client/server round-trip, which only
proves self-consistency. The committed checker `scripts/triton_compat_check.py` runs that
cross-implementation check across the example models:
```bash
pip install -r scripts/requirements.txt
# with a nereid.yaml exposing pymul, pyaddint, rustint, multi and model3, server running:
python scripts/triton_compat_check.py --url 127.0.0.1:50051 \
    --model pymul:mul --model pyaddint:addint --model rustint:addint64 --model model3 \
    --multi multi --stream pymul
# -> ... rustint: output == input+1 (int64) ✓ ... multi (sum, prod) == (a+b, a*b) ✓ ... TRITON_COMPAT_OK
```

## Server behavior
- `Nereid/HealthCheck` returns status `ok`.
- `Nereid/ViewModels` returns model names configured in `nereid.yaml`.
- `Nereid/Checkpoint` only accepts `model_name` values configured in `nereid.yaml`. The server
  detects each model's backend kind from its folder contents (or from an explicit `backend` in
  `nereid.yaml`):
  - `main.py` + `requirements.txt` -> runs `main.py` in that model's venv, streams its
    stdout/stderr lines as text chunks, and streams the model's typed output tensor as
    `output_chunk`s. A Python model must ship a `model_inference.textproto` declaring
    `output_shape` (see [Python tensor contract](#python-tensor-contract)); if it also declares
    `input_shape`, the validated request tensor is piped into `main.py` on stdin.
  - `model_inference.textproto` + `.pt` model -> runs Rust inference for that model and streams
    the output tensor.

## Model folder contract
Each model must be a folder under `<server.ml_backends_path>/<model_name>/` with:
- `requirements.txt`
- `main.py`
- `model_inference.textproto` declaring `output_shape` (required — see below)
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

### Python tensor contract
Every Python model **must** ship a `model_inference.textproto` declaring `output_shape`: each
reply is a typed output tensor. `input_shape` is optional — a model may consume no tensor input.

**Input (optional).** When `input_shape` is declared, the server validates each request tensor
against it — rejecting shape/rank/batch mismatches with `InvalidArgument` *before* launching
`main.py` — and delivers the validated tensor to the process:
- **stdin**: the raw tensor bytes, little-endian `float32`, row-major.
- **`NEREID_INPUT_SHAPE`**: the (batch-normalized) shape, comma-separated, e.g. `1,16`.
- **`NEREID_INPUT_DTYPE`**: the element dtype, currently always `float32`.

**Output (required).** `main.py` writes its output tensor to the file named by
**`NEREID_OUTPUT_PATH`** in a self-describing framed format: a UTF-8 header line
`"float32 d0,d1,...\n"` followed by the raw little-endian `float32` bytes. The server validates
it against the declared `output_shape` (and the request batch size) and streams it back as
`CheckpointResponse.output_chunk`s. stdout/stderr remain available as streamed text log chunks.

A minimal `main.py` (no third-party dependencies) reading an input tensor and replying with one:
```python
import os, struct, sys

raw = sys.stdin.buffer.read()  # present when input_shape is declared
values = struct.unpack("<%df" % (len(raw) // 4), raw) if raw else ()
result = [float(sum(values))]  # ... run inference ...

with open(os.environ["NEREID_OUTPUT_PATH"], "wb") as f:
    f.write(("float32 %d\n" % len(result)).encode("utf-8"))
    f.write(struct.pack("<%df" % len(result), *result))
```
See `ml-backends/model1` and `ml-backends/model2` for complete examples. A Python model that
exits 0 without writing a valid output tensor is a contract violation and the request fails.

## Native ONNX and TensorFlow backends
Beyond `.pt` and Python, nereid can serve **ONNX** and **TensorFlow** models *natively* —
in-process, no Python subprocess — behind opt-in build features. They're off by default so the
base build stays lean; enable them with `./build.sh --onnx` / `--tensorflow` (or
`cargo build --features onnx,tensorflow`).

**Every backend is a Cargo feature, so you build only the ones you want.** The default build
keeps the two original backends (`torch` + `python`); pass `--no-default-features` with just the
features you need. An **ONNX-only** or **TensorFlow-only** server links **no libtorch at all** —
the whole [libtorch build story](#the-buildsh-libtorch-aware-build-driver) simply doesn't apply
to it.

| Feature | Backend | Heavy dependency |
| --- | --- | --- |
| `torch` (default) | TorchScript `.pt` | libtorch (via `tch`) |
| `python` (default) | Python `main.py` | none |
| `onnx` | ONNX | ONNX Runtime (via `ort`) |
| `tensorflow` | TensorFlow SavedModel | libtensorflow |

```bash
./build.sh --backends onnx                 # ONNX only — no libtorch, no python
./build.sh --backends torch,onnx           # .pt + ONNX
./build.sh --release --link bundled --backends onnx,tensorflow   # native-only, self-contained
cargo build --no-default-features --features onnx                # the same, via cargo
```
A model whose files need a backend the server wasn't built with fails at startup with a clear
"rebuild with `--features …`" message.

### Selecting backends the manifest doesn't know about
Cargo features select the *in-tree* backends, because a feature can only name a backend
`Cargo.toml` already lists. Backends are discovered, though — anyone can drop a folder (or a git
submodule) into `src/backends/`, and such a backend has no feature to switch it off. So the
build takes a second, independent selection **by name pattern**, `$NEREID_BACKENDS` or a
`backends.conf` beside `Cargo.toml` (the env var wins; both use the same syntax):

```bash
NEREID_BACKENDS='onnx,tensorflow' cargo build --no-default-features --features onnx,tensorflow
NEREID_BACKENDS='!torch'          cargo build      # everything discovered except torch
NEREID_BACKENDS='*,!vendor-*'     cargo build      # drop a whole family of vendored backends
```
Patterns are comma- or newline-separated, `*` matches any run of characters, a leading `!`
excludes (and beats any include), and `#` starts a comment in `backends.conf`. Unset or empty
selects every discovered backend — the default. A build prints what the selection excluded, so a
dropped backend is never silent.

The two axes compose: the pattern decides which backend folders are *compiled in at all*, the
feature decides whether an in-tree backend's engine (and its heavy dependency) comes with it.

- **ONNX** — a folder with one `.onnx` file + `model_inference.textproto`. Runs on
  [ONNX Runtime](https://onnxruntime.ai/) via the `ort` crate, with the **CUDA execution
  provider** selected when the model's `device` is `cuda`.
- **TensorFlow** — a folder with a **SavedModel** (`saved_model.pb` + `variables/`) +
  `model_inference.textproto`. Runs on libtensorflow via the `tensorflow` crate; the SavedModel
  signature defaults to `serving_default` (override per model with `signature:` in `nereid.yaml`).
  GPU needs the libtensorflow GPU build (`./build.sh --tensorflow-gpu`).

Both are served over the same surfaces as the `.pt` backend: the Triton `ModelInfer` path
(single- **and** multi-tensor, the full KServe dtype set) and, for single-tensor models,
nereid's native `Checkpoint`. Backend detection is automatic from the folder contents; set
`backend: "onnx"` / `"tensorflow"` in `nereid.yaml` to be explicit or to disambiguate. A model
whose files need a backend the server wasn't built with fails at startup with a clear
"rebuild with `--features …`" message.

Generate the example models (`onnxadd`, `tfadd` — each computes `input + 1`) with:
```bash
python scripts/make_example_models.py   # needs `onnx` (and `tensorflow` for tfadd)
```

**Dtype note:** the native path additionally supports `UINT16/32/64` (which the Rust `.pt` path
cannot, as libtorch lacks those kinds); TensorFlow does not support `BF16`.

## Compile-time C++ models (feature `cxx`)
If you want a C++ model running in process with type-safe interop, and no subprocess or unsafe FFI
in the way, nereid can link the C++ straight in through the [`cxx`](https://cxx.rs) crate. The C++
lives in the `cxx-models` crate (`crates/cxx-models/`, meant to be vendored as a git submodule or
workspace member): implement the `nereid::Model` interface in `cpp/`, register it by name in
`create_model`, and rebuild with `./build.sh --cxx` (or `cargo build --features cxx`).

Since the C++ is compiled into the binary and keyed by name, a model's folder holds nothing but a
`model_inference.textproto`, and you select it with `backend: "cxx"` in `nereid.yaml`. It can't be
auto-detected, because there is no file to detect — which is what the registry's
`auto_detect: false` is for: the backend registers itself like any other, but is only ever
selected by being named. It's served over the Triton `ModelInfer` path, single-tensor for now. See
`crates/cxx-models` for the `cxxadd` example (`output = input + 1`) and `ml-backends/cxxadd` for
its model folder.

So this is the build-time counterpart to the other backends. Adding or changing a C++ model means
recompiling the server, which is a real cost and worth being honest about; what you get for it is
direct in-process interop with no process or loader machinery, and a boundary the compiler checks
for you rather than one you hand-write and hope about.

## C++ subprocess backend (feature `cpp`)
A C++ model is served the same way a Python `main.py` model is: out of process. The folder ships a
`main.cpp` alongside its `model_inference.textproto`, the server compiles it into a `model`
executable at startup (the same idea as building a Python model's venv), and then runs it as a
child process per request. Turn it on with `./build.sh --cpp` (or `cargo build --features cpp`).
It needs a C++ compiler on `PATH` at runtime, since that's when the compiling happens.

The model speaks the same language-agnostic tensor contract the Python backend uses, so there is
no unsafe FFI anywhere and you never rebuild the server to add a model:
- **input** (optional): raw little-endian bytes on stdin, with `NEREID_INPUT_SHAPE` /
  `NEREID_INPUT_DTYPE` in the environment;
- **output**: a framed tensor written to `NEREID_OUTPUT_PATH` — a header line
  `"<dtype> <d0>,<d1>,...\n"` followed by the raw little-endian bytes.

Running out of process is the whole point: a model that segfaults takes down its own process and
nothing else, which is not something an in-process dynamically-loaded plugin can promise you. If
`main.cpp` and one compiler invocation aren't enough — extra sources, link flags — a folder can
ship an executable `build.sh` instead, or just a prebuilt `model` binary. `ml-backends/cppadd` is a
complete example (`output = input + 1`); starting the server builds it for you, or you can compile
it yourself with `c++ -O2 -std=c++17 main.cpp -o model`.

It's served over the Triton `ModelInfer` path, single-tensor for now; the native `Checkpoint`
streaming path and multi-tensor C++ models are both deferred. Detection is automatic from
`main.cpp`, and you can set `backend: "cpp"` in `nereid.yaml` to be explicit about it.

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

### `./build.sh` — libtorch-aware build driver
`cargo build` links libtorch *dynamically* and leaves the binary needing
`LD_LIBRARY_PATH` at runtime (see [Running the built binary directly](#running-the-built-binary-directly-not-via-cargo-run)).
`./build.sh` wraps `cargo build`, resolves the libtorch dependency, and can produce
a self-contained (or statically linked) binary. It also folds in the pieces an HPC
build needs — `module load`, an external libtorch install, or a conda/pyenv/venv
environment. Run `./build.sh --help` for the full option list; the
**[docs site](docs/building.md)** has the detailed walkthrough. The three linking
modes:

- **`dynamic`** (default) — ordinary build. Locates the libtorch `lib/` directory
  and writes a `target/<profile>/run-grpc-test.sh` wrapper that sets
  `LD_LIBRARY_PATH` for you.
  ```bash
  ./build.sh --release            # then: target/release/run-grpc-test.sh
  ```
- **`bundled`** — relocatable, "ship it anywhere" build. Copies libtorch's shared
  objects next to the binary under `dist/grpc-test/` and patches its rpath to
  `$ORIGIN/lib`, so it runs with **no `LD_LIBRARY_PATH`** and moves as a unit — a
  good fit for containers, systemd, or a tarball. Needs `patchelf`
  (`apt-get install patchelf`); without it a `run.sh` wrapper is written instead.
  ```bash
  ./build.sh --release --link bundled --fetch-libtorch
  # -> dist/grpc-test/{grpc-test, lib/*.so}; just run dist/grpc-test/grpc-test
  ```
- **`static`** — true static link of libtorch (`LIBTORCH_STATIC=1`). PyTorch no
  longer ships a prebuilt static libtorch, so `./build.sh --build-libtorch` builds
  one from source (`scripts/build-libtorch.sh`) and links against it; or point
  `--libtorch` at a static build you already have. The script validates the
  archives up front and confirms the result has no libtorch runtime dependency.
  ```bash
  ./build.sh --release --link static --build-libtorch   # long; builds libtorch.a
  ```

**libtorch source** (precedence: `--build-libtorch` > `--fetch-libtorch` >
`--libtorch`/`$LIBTORCH` > `tch`'s own download). `--fetch-libtorch` downloads the
official libtorch and **verifies its sha256** before use, instead of the opaque
download `tch` does by default.

**Other flags:** `--device cpu|cuda|cuda:<ver>` (sets `TORCH_CUDA_VERSION`),
`--module <spec>` (repeatable, `module load` for HPC), `--conda <env>` /
`--pyenv <ver>` / `--venv <dir>` to build inside a managed environment, `--out <dir>`
for the bundle location, `--run` to launch after building, and `-- <cargo args>` to
pass flags straight through to cargo.

The previous cluster-specific `build.sh` is now the default `--module ... --libtorch ...`
path — e.g. `./build.sh --module cuda/12.6.0 --device cuda --libtorch <shared-install>`.

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
- `src/backend/`: the `Backend` trait, the `ModelManager`, and the registry that detects a
  model's backend and loads it — all backend-agnostic.
- `src/backends/<name>/`: one self-contained folder per backend (`mod.rs` registers it and
  detects its folder shape; the feature-gated `imp.rs` is the engine).
- `build.rs`: compiles `.proto` files and discovers the backend folders at build time.
- `proto/inference.proto`: service and message definitions.
- `python-client/client.py`: YAML-configured mock ED client.
