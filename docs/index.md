# nereid-server

A nifty little Rust inference server. nereid loads ML models from a config file and serves them
over gRPC — both its own native `Nereid` service and a **Triton-compatible** KServe v2 surface on
the same address, so a stock `tritonclient` can drive it without code changes.

Models are served by pluggable **backends**, and each backend is an opt-in build feature, so you
compile only the runtimes you need.

## At a glance

```text
   gRPC client                 nereid-server
 (tritonclient,      ┌───────────────────────────────────────────┐
  grpcurl, ...)      │  gRPC surfaces                             │
      │   ModelInfer │    Nereid  +  Triton (KServe v2)           │
      └─────────────►│         │                                 │
        Checkpoint   │         ▼                                 │
                     │    ModelManager                           │
                     │         │                                 │
                     │         ▼                                 │
                     │   per-model worker + bounded queue        │
                     │         │                                 │
                     │         ▼                                 │
                     │      backend ──► Rust .pt   (libtorch)     │
                     │              ──► Python     (main.py)      │
                     │              ──► ONNX       (ONNX Runtime) │
                     │              ──► TensorFlow (SavedModel)   │
                     └───────────────────────────────────────────┘
```

## What to read next

- **[Architecture](architecture.md)** — the gRPC surfaces, the `ModelManager`, per-model workers
  and backpressure, and how a request flows to a backend.
- **[Backends](backends.md)** — the Rust `.pt`, Python, ONNX, and TensorFlow backends, and the
  modular feature system that lets you build only what you want.
- **[Model contract](model-contract.md)** — what goes in a model folder, the
  `model_inference.textproto`, batching rules, and the subprocess tensor contract.
- **[Triton compatibility](triton.md)** — how nereid speaks KServe v2 on the wire, which RPCs are
  implemented, and how to verify it.
- **[Building & running](building.md)** — `build.sh`, the libtorch dependency, linking modes, HPC
  builds, and selecting backends.

## Core ideas

- **Config-driven.** `nereid.yaml` lists the models to expose, each model's execution device, and
  its request-queue size. A model that isn't in the config isn't served.
- **Folder-per-model.** Each model is a directory under `server.ml_backends_path`; its backend is
  detected from the folder's contents (or set explicitly with `backend:` in `nereid.yaml`).
- **Wire-compatible with Triton.** The `inference.GRPCInferenceService` surface is vendored from
  Triton's own proto, so the wire format is byte-compatible — see
  [Triton compatibility](triton.md).
- **Modular by build feature.** `torch`, `python`, `onnx`, and `tensorflow` are Cargo features;
  an ONNX-only or TensorFlow-only build links no libtorch at all.
