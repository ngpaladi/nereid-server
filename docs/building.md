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

`cargo build` links libtorch dynamically and hands you a binary that needs `LD_LIBRARY_PATH` set
before it will run. That's fine on your laptop and a nuisance everywhere else, so `./build.sh`
wraps `cargo build`: it works out where libtorch is coming from, selects your backends, and can
produce a self-contained or statically linked binary. `./build.sh --help` has the full option
list.

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

`--backends <csv>` picks an exact set, which turns off the `torch`+`python` defaults, while
`--onnx` / `--tensorflow` add to whatever is already selected. `--link bundled` bundles whichever
runtimes ended up linked.

Those are Cargo features, so they can only name backends that `Cargo.toml` already knows about. If
you've dropped your own backend into `src/backends/` (or pointed a git submodule at one), there is
no feature for it, so the build takes a second selection by name pattern instead — `$NEREID_BACKENDS`
or a `backends.conf` file:

```bash
NEREID_BACKENDS="onnx,tensorflow" cargo build --no-default-features --features onnx,tensorflow
NEREID_BACKENDS="!torch" ./build.sh          # everything discovered except torch
```

`*` matches any run of characters and a leading `!` excludes; unset means everything that was
discovered. [Backends](backends.md#adding-your-own-backend) has the details.

### HPC & managed environments

```bash
./build.sh --module cuda/12.6.0 --device cuda --libtorch /depot/group/torch/libtorch --release
./build.sh --conda nereid --release       # or --pyenv <ver> / --venv <dir>
```

`--module` (repeatable) runs `module load` before building; `--conda` / `--pyenv` / `--venv`
activate a managed environment first — useful on clusters with a module system or a non-default
Python.

## Running the built binary directly

`cargo run` and `cargo test` set `LD_LIBRARY_PATH` for you, which is why this only bites once you
run the binary yourself — from a container, from systemd, or just `./target/debug/grpc-test`:

```
error while loading shared libraries: libtorch_cpu.so: cannot open shared object file
```

Fix it, cleanest first:

1. Run the generated wrapper: `target/<profile>/run-grpc-test.sh`.
2. Build `--link bundled` and run `dist/grpc-test/grpc-test` — the path is baked in.
3. Set it yourself: `export LD_LIBRARY_PATH=<libtorch>/lib:$LD_LIBRARY_PATH`.

A native-only build (`--backends onnx`) links no libtorch at all, so none of this applies to it.

## Configure and run

Copy the example config and start the server:

```bash
cp nereid.yaml.example nereid.yaml
cargo run    # or ./build.sh --run
```

The server binds `server.bind_addr` and loads models from `server.ml_backends_path` — see the
[model contract](model-contract.md).
