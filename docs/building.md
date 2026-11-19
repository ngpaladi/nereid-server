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

`--backends <csv>` is how you pick an exact set with `build.sh`: it turns off the `torch`+`python`
defaults and builds only what you name. (Under the hood it's `build.sh` that passes Cargo's
`--no-default-features` for you — that's a Cargo flag, not one `build.sh` accepts, so reach for it
only when you drive `cargo` directly, as in the third line above.) `--onnx` / `--tensorflow` add to
whatever's already selected, and `--link bundled` bundles whichever runtimes ended up linked.

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

## Container image

Every published GitHub release builds a `linux/amd64` image and pushes it to the GitHub
Container Registry:

```bash
docker pull ghcr.io/ngpaladi/nereid-server:latest
```

Tags follow the release tag: `{version}`, `{major}.{minor}`, `{major}`, plus a long
`sha-<commit>` tag on every build. `latest` moves only for a full release — a prerelease
never claims it.

### Running it

The server reads `nereid.yaml` from its working directory and resolves
`server.ml_backends_path` relative to it. The image sets `WORKDIR /nereid`, so that is where
the config and the model folders belong:

```bash
docker run --rm -p 50051:50051 -p 8002:8002 -v "$PWD:/nereid" ghcr.io/ngpaladi/nereid-server:latest
```

Two things differ from a host run:

- **Bind a non-loopback address.** `nereid.yaml.example` uses `[::1]:50051`, which inside a
  container is reachable only from that container. Use `[::]:50051` (or `0.0.0.0:50051`) — and
  likewise `http_addr: "[::]:8002"` if you publish the [metrics / health](metrics.md) port.
- **The container runs unprivileged**, as uid 10001. The Python backend builds a `venv/`
  *inside* each model folder, so a mounted model directory has to be writable by that uid —
  otherwise run with `--user "$(id -u):$(id -g)"` to match the host owner.

### What's in it

A two-stage build. The builder runs `./build.sh --release --link bundled --fetch-libtorch`,
so libtorch is downloaded with its sha256 verified and its shared objects end up next to the
binary under an `$ORIGIN/lib` rpath. The runtime stage is `debian:bookworm-slim` plus
`python3`/`python3-venv` (for the Python backend's per-model virtualenv) and that bundle at
`/opt/nereid` — no Rust toolchain, no `target/`.

The image ships the default backends (TorchScript `.pt` + Python), which is what makes it
roughly 600 MB — the bundled libtorch is ~470 MB of that. The `BUILD_ARGS` build argument is
passed
straight through to `build.sh`, so a leaner or differently-equipped image is one flag away:

```bash
docker build --build-arg BUILD_ARGS="--backends onnx" -t nereid-server:onnx .   # links no libtorch
docker build --build-arg BUILD_ARGS="--onnx --tensorflow" -t nereid-server:all .
```

### Building or publishing by hand

The **Container image** workflow also takes a `workflow_dispatch`. Left alone it builds and
smoke-tests without publishing; tick `push` to publish the result under the branch and commit
tags. The smoke test checks the two things a container gets wrong first — that every shared
object the binary needs resolves inside the image, and that the server starts and reads its
config — not that any model serves.
