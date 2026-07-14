# Building & running

## Prerequisites

- **Rust (stable) + Cargo** — the only hard requirement for a default build.
- A backend's extra tools, only if you enable it: a CUDA toolkit for `--device cuda`; `python3-venv`
  for the Python backend at runtime; a C++ compiler for the ONNX/TF build scripts.

The quickest start:

```bash
cargo build
cargo run          # reads ./nereid.yaml
```

## `./build.sh` — the libtorch-aware build driver

`cargo build` links libtorch *dynamically* and leaves the binary needing `LD_LIBRARY_PATH` at
runtime. `./build.sh` wraps `cargo build`, resolves the libtorch dependency, selects backends, and
can produce a self-contained or statically linked binary. Run `./build.sh --help` for the full
option list.

### Linking modes (`--link`)

**`dynamic`** (default) — ordinary build. Finds the libtorch it linked and writes a
`run-grpc-test.sh` wrapper that sets `LD_LIBRARY_PATH` for you.

```bash
./build.sh --release      # then: target/release/run-grpc-test.sh
```

**`bundled`** — copies libtorch's shared objects next to the binary and sets an `$ORIGIN` rpath, so
it runs with **no `LD_LIBRARY_PATH`** and relocates as a unit — good for containers and tarballs.

```bash
./build.sh --release --link bundled --fetch-libtorch
# -> dist/grpc-test/{grpc-test, lib/*.so}
```

**`static`** — statically links libtorch (`LIBTORCH_STATIC=1`). PyTorch no longer ships a prebuilt
static libtorch, so `--build-libtorch` builds one from source and links against it.

```bash
./build.sh --release --link static --build-libtorch
```

### Where libtorch comes from

In precedence order:

- `--build-libtorch` — build a static libtorch from source (`scripts/build-libtorch.sh`).
- `--fetch-libtorch` — download the official libtorch and **verify its sha256** before use.
- `--libtorch <dir>` / `$LIBTORCH` — use an existing install (the HPC path).
- *(default)* — let `tch` download a CPU libtorch itself.

### Selecting backends

```bash
./build.sh --onnx --tensorflow          # add native backends to the default torch+python
./build.sh --backends onnx              # ONNX only — links no libtorch at all
cargo build --no-default-features --features onnx,tensorflow   # the same, via cargo
```

`--backends <csv>` picks an exact set (turning off the `torch`+`python` defaults);
`--onnx` / `--tensorflow` add on top. `--link bundled` bundles whichever runtimes are linked.

### HPC & managed environments

```bash
./build.sh --module cuda/12.6.0 --device cuda --libtorch /depot/group/torch/libtorch --release
./build.sh --conda nereid --release       # or --pyenv <ver> / --venv <dir>
```

`--module` (repeatable) runs `module load` before building; `--conda` / `--pyenv` / `--venv`
activate a managed environment first — useful on clusters with a module system or a non-default
Python.

## Running the built binary directly

`cargo run` / `cargo test` set `LD_LIBRARY_PATH` for you, but a binary you run directly
(`./target/debug/grpc-test`, a container, systemd) will hit:

```
error while loading shared libraries: libtorch_cpu.so: cannot open shared object file
```

Fix it, cleanest first:

1. Run the generated wrapper: `target/<profile>/run-grpc-test.sh`.
2. Build `--link bundled` and run `dist/grpc-test/grpc-test` — the path is baked in.
3. Set it yourself: `export LD_LIBRARY_PATH=<libtorch>/lib:$LD_LIBRARY_PATH`.

A native-only build (`--backends onnx`) links no libtorch, so none of this applies to it.

## Configure and run

Copy the example config and start the server:

```bash
cp nereid.yaml.example nereid.yaml
cargo run    # or ./build.sh --run
```

The server binds `server.bind_addr` and loads models from `server.ml_backends_path` — see the
[model contract](model-contract.md).
