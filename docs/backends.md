# Backends

A **backend** is how a model actually runs. nereid works out which one a model needs from the
contents of its folder, or you can just say so with `backend:` in `nereid.yaml`.

| Backend | Feature | Model folder | Heavy dependency | Isolation |
| --- | --- | --- | --- | --- |
| Torch (TorchScript) | `torch` *(default)* | `.pt` + textproto | libtorch (via `tch`) | in-process |
| Python | `python` *(default)* | `main.py` + `requirements.txt` | none | subprocess |
| ONNX | `onnx` | `.onnx` + textproto | ONNX Runtime (via `ort`) | in-process |
| TensorFlow | `tensorflow` | SavedModel + textproto | libtensorflow | in-process |

You only build the ones you want. The default build keeps `torch` + `python`; pass
`--no-default-features` with just the features you need, and an ONNX-only server never links
libtorch. See [Building & running](building.md).

## Rust TorchScript (`.pt`)

A folder with one `.pt` file (a TorchScript export) and a `model_inference.textproto`. The model
is loaded into libtorch through [`tch`](https://github.com/LaurentMazare/tch-rs) and run in
process. Single- and multi-tensor models both work, across the libtorch dtypes (`FP16/32/64`,
`INT8/16/32/64`, `UINT8`, `BOOL`, `BF16`), and each model's `device` (`cpu`, `cuda`, or
`cuda:<index>`) picks the CPU or a particular GPU.

## Python (`main.py`)

A folder with `main.py` + `requirements.txt`. At startup nereid builds a virtualenv for the model
from `requirements.txt` (and reuses it if one is already there), then runs `main.py` as a
subprocess per request. The model reads its input tensor on stdin and writes a typed output tensor
to a scratch file — the [subprocess tensor contract](model-contract.md#subprocess-tensor-contract)
has the details. Since it runs out of process, a model that crashes takes down only itself.

## ONNX

A folder with one `.onnx` file + textproto, run in process on
[ONNX Runtime](https://onnxruntime.ai/) via the [`ort`](https://ort.pyke.io/) crate. If a model's
`device` is `cuda`, ort's CUDA execution provider is selected for it. This path covers the full
KServe dtype set, including `UINT16/32/64`, which the `.pt` path can't do because libtorch has no
kind for them.

## TensorFlow

A folder with a SavedModel (`saved_model.pb` + `variables/`) + textproto, run on libtensorflow via
the [`tensorflow`](https://github.com/tensorflow/rust) crate. The SavedModel signature defaults to
`serving_default`, which you can override per model with `signature:` in `nereid.yaml`. GPU
support needs the libtensorflow GPU build, and `BF16` isn't available on this path.

## Choosing a backend

If a folder matches exactly one backend, that's the one you get. Set `backend:` in `nereid.yaml`
when you want to be explicit, or when a folder ships files for more than one backend and the
server can't pick for you:

```yaml
models:
  - name: "my_onnx_model"
    device: "cuda"
    queue_capacity: 16
    backend: "onnx"    # "python" | "torch" (or "rust") | "onnx" | "tensorflow"
```

A model whose files need a backend the server wasn't built with fails at startup and tells you
which `--features` to rebuild with. It is never quietly mislabeled or skipped, which matters more
than it sounds: a model that silently runs on the wrong engine is a much worse day than one that
refuses to start.

## How the server finds a backend

Nothing in the server core knows the four backends above exist. There's no enum of backend kinds
and no `match` that dispatches to them, which is deliberate — every one of those would be a
central file you'd have to edit to add a backend.

Instead each backend lives in its own folder under `src/backends/<name>/` and submits a
registration at link time:

```rust
inventory::submit! {
    BackendRegistration {
        name: "tensorflow",             // the `backend:` value in nereid.yaml
        version: "0.1.0",               // this backend's own version
        aliases: &[],                   // other accepted spellings
        describes: "a SavedModel (saved_model.pb + variables/) + model_inference.textproto",
        auto_detect: true,              // false = only selectable by declaring it
        detect,                         // does this folder look like my model?
        load,                           // build the backend, or say which feature is missing
    }
}
```

The core iterates those registrations to detect and load, so it never names a backend. Detection
is pure file inspection with no dependency on the engine itself, which means the registry is
complete even when a backend's feature is switched off — a `.pt` folder in an ONNX-only build
still gets a precise "rebuild with `--features torch`" instead of a confusing "no backend matches
this folder".

Two fields are worth calling out. `version` is the backend's own version rather than the server's,
so a backend that evolves on its own schedule can say where it is; bump the major when a revision
changes the folder shape, the contract, or what a model has to declare. It's reported in the
startup log for every model loaded, so you can trace a deployment back to the exact revision that
served it. `auto_detect: false` is for a backend whose code is compiled into the server rather
than sitting on disk — there's no file signature to look for, so it's selectable only by naming it
in `nereid.yaml`.

## Adding your own backend

Drop a folder into `src/backends/` with two files in it:

- `mod.rs` — the detection predicate and the `inventory::submit!` above. This is always compiled,
  and must not depend on the engine.
- `imp.rs` — the engine itself, behind `#[cfg(feature = "...")]`, implementing the `Backend` trait
  (`platform()`, `infer()`, and optionally `checkpoint_stream()`).

Then build. `build.rs` globs the subfolders of `src/backends/` and emits the module declarations,
so there is no `mod` line to add, no enum arm, no detection list, and no registration call
anywhere else in the tree. Because a backend is just a directory, it can be a git submodule
pointing at your own repository, and one that hasn't been initialized yet (so it has no `mod.rs`)
is skipped rather than breaking the build.

Which discovered backends get compiled in is a separate question from Cargo features, because a
feature can only name a backend that `Cargo.toml` already lists — which an out-of-tree backend, by
definition, doesn't. So the build also takes a selection by name pattern, from `$NEREID_BACKENDS`
or a `backends.conf` file:

```bash
NEREID_BACKENDS="onnx,tensorflow" cargo build --no-default-features --features onnx,tensorflow
NEREID_BACKENDS="!torch"          cargo build      # everything discovered except torch
NEREID_BACKENDS="*,!vendor-*"     cargo build      # drop a family of vendored backends
```

Patterns are separated by commas or newlines, `*` matches any run of characters, a leading `!`
excludes (and beats any include), and `#` starts a comment in the file. Leave it unset and you get
everything that was discovered; whatever the selection drops is printed as a build warning, so a
missing backend is never a mystery. The two knobs compose rather than overlap: the pattern decides
which backend folders are compiled in at all, and the feature decides whether an in-tree backend's
engine and its heavy dependency come along with it.

One caveat, since it will come up the first time you edit one of these files: `cargo fmt` and
rust-analyzer both walk the module tree, and that tree stops at the generated `include!` that
wires in `src/backends/`. CI therefore runs `rustfmt` on those files directly in addition to
`cargo fmt --all --check`. rust-analyzer does run build scripts and should resolve the generated
modules, though its support for `include!`-wired modules can be flaky.
