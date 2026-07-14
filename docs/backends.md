# Backends

A **backend** is how a model actually runs. nereid detects a model's backend from its folder
contents, or you set it explicitly with `backend:` in `nereid.yaml`. Every backend is an opt-in
**Cargo feature**, so you build only the runtimes you need.

| Backend | Feature | Model folder | Heavy dependency | Isolation |
| --- | --- | --- | --- | --- |
| Rust TorchScript | `torch` *(default)* | `.pt` + textproto | libtorch (via `tch`) | in-process |
| Python | `python` *(default)* | `main.py` + `requirements.txt` | none | subprocess |
| ONNX | `onnx` | `.onnx` + textproto | ONNX Runtime (via `ort`) | in-process |
| TensorFlow | `tensorflow` | SavedModel + textproto | libtensorflow | in-process |

> **Build only what you want.** The default build keeps `torch` + `python`. Use
> `--no-default-features` with just the features you need — e.g.
> `cargo build --no-default-features --features onnx` for an **ONNX-only** server that never links
> libtorch. See [Building & running](building.md).

## Rust TorchScript (`.pt`)

A folder with one `.pt` file (a TorchScript-exported model) and a `model_inference.textproto`. The
model is loaded into libtorch via [`tch`](https://github.com/LaurentMazare/tch-rs) and run on the
per-model worker thread. Supports single- and multi-tensor models and the libtorch dtypes
(`FP16/32/64`, `INT8/16/32/64`, `UINT8`, `BOOL`, `BF16`); a per-model `device` (`cpu`, `cuda`, or
`cuda:<index>`) selects CPU or a specific GPU.

## Python (`main.py`)

A folder with `main.py` + `requirements.txt`. On startup nereid builds a per-model virtualenv from
`requirements.txt` (reused if it already exists) and runs `main.py` as a **subprocess per request**.
The model reads its input tensor on stdin and writes a typed output tensor to a scratch file — see
the [subprocess tensor contract](model-contract.md#subprocess-tensor-contract). Because it runs out
of process, a crash in the model is contained to its own process.

## ONNX

A folder with one `.onnx` file + textproto. Runs on [ONNX Runtime](https://onnxruntime.ai/) via the
[`ort`](https://ort.pyke.io/) crate, in-process on the model's worker thread. The **CUDA execution
provider** is selected when a model's `device` is `cuda`. Served over the Triton `ModelInfer` path,
single- and multi-tensor, across the full KServe dtype set — including `UINT16/32/64`, which the
Rust `.pt` path can't (libtorch has no kind for them).

## TensorFlow

A folder with a **SavedModel** (`saved_model.pb` + `variables/`) + textproto. Runs on libtensorflow
via the [`tensorflow`](https://github.com/tensorflow/rust) crate. The SavedModel signature defaults
to `serving_default` (override per model with `signature:` in `nereid.yaml`); GPU needs the
libtensorflow GPU build. TensorFlow does not support `BF16` on this path.

## Choosing a backend

Detection is automatic when a folder unambiguously matches one backend. Set `backend:` in
`nereid.yaml` to be explicit, or to disambiguate a folder that ships files for more than one:

```yaml
models:
  - name: "my_onnx_model"
    device: "cuda"
    queue_capacity: 16
    backend: "onnx"    # "python" | "rust" | "onnx" | "tensorflow"
```

A model whose files need a backend the server **wasn't built with** fails at startup with a clear
message telling you to rebuild with the right `--features` — the server never silently mislabels or
skips a model.
