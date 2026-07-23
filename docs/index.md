# nereid-server

A nifty little Rust inference server. nereid reads a config file, loads the models you list in it,
and serves them over gRPC — both its own native `Nereid` service and the standard KServe v2 gRPC
surface on the same address, so any KServe v2 client (a stock `tritonclient` included) can drive it
without changing a line of client code.

How a model actually runs is the job of a **backend**. Backends are self-contained and
self-registering, so the server core has no idea which ones exist, and you compile in only the
ones you want.

## At a glance

```text
   gRPC client                 nereid-server
 (tritonclient,      ┌───────────────────────────────────────────┐
  grpcurl, ...)      │  gRPC surfaces                            │
      │   ModelInfer │    Nereid  +  KServe v2                   │
      └─────────────►│         │                                 │
        Checkpoint   │         ▼                                 │
                     │    ModelManager                           │
                     │         │  (per-model permits)            │
                     │         ▼                                 │
                     │      backend ──► Torch .pt  (libtorch)    │
                     │              ──► Python     (main.py)     │
                     │              ──► ONNX       (ONNX Runtime)│
                     │              ──► TensorFlow (SavedModel)  │
                     └───────────────────────────────────────────┘
```

## What to read next

- **[Architecture](architecture.md)** — the two gRPC surfaces, the `ModelManager`, how requests are
  bounded, and how one gets to a backend.
- **[Backends](backends.md)** — the four backends that ship today, how the server finds them, and
  how you add one of your own.
- **[Model contract](model-contract.md)** — what goes in a model folder, what
  `model_inference.textproto` says, the batching rules, and the subprocess tensor contract.
- **[KServe v2 compatibility](triton.md)** — how nereid speaks KServe v2 on the wire, which RPCs
  are implemented, and how to check that for yourself.
- **[Building & running](building.md)** — `build.sh`, the libtorch dependency, linking modes, HPC
  builds, and choosing your backends.

## Core ideas

- **Config-driven.** `nereid.yaml` lists the models to expose, each model's device, and how many
  requests it will hold at once. If a model isn't in the config, it isn't served.
- **Folder-per-model.** Every model is a directory under `server.ml_backends_path`, and the server
  works out its backend from what's in the folder (or from an explicit `backend:` in the config).
- **Speaks the KServe v2 standard.** The `inference.GRPCInferenceService` surface is vendored from
  the KServe v2 spec, so what goes over the wire is byte-compatible with any client of it. See
  [KServe v2 compatibility](triton.md).
- **Backends are discovered, not listed.** Each one lives in its own folder and registers itself at
  link time, so nothing in the core enumerates them, and adding a backend doesn't mean editing the
  core. An ONNX-only build links no libtorch at all.
