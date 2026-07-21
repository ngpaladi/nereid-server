# Architecture

## Two gRPC surfaces, one address

nereid binds a single address (`server.bind_addr`) and serves two gRPC services on it:

- **`inference.Nereid`** — nereid's own service: `HealthCheck`, `ViewModels`, and the streaming
  `Checkpoint` inference RPC.
- **`inference.GRPCInferenceService`** — NVIDIA Triton's KServe v2 surface, vendored verbatim so
  that a stock `tritonclient` can drive nereid unchanged (see
  [Triton compatibility](triton.md)).

They aren't two systems. Both are backed by the same models and the same `ModelManager`; they're
just two front doors into it.

## The ModelManager

At startup the server reads `nereid.yaml` and builds a `ModelManager` that owns every configured
model. For each one it:

1. Resolves the model folder under `server.ml_backends_path`.
2. Works out the backend from what's in the folder, or honors an explicit `backend:` in the config
   (see [Backends](backends.md)).
3. Hands the folder to that backend to load — creating a virtualenv, loading a `.pt` into
   libtorch, opening an ONNX Runtime session — and reads the model's `model_inference.textproto`
   contract.
4. Stores the loaded backend alongside its contract and a bounded pool of request permits, and
   logs which backend, at which version, is serving the model.

If any model can't be prepared — a missing file, a bad contract, a backend the server wasn't built
with — startup fails with an actionable error instead of coming up with a broken model in it. This
is one place where being strict is clearly right: a server that starts happily and then fails on
the first real request has just moved the problem somewhere much less convenient.

## Backpressure

Each model gets a semaphore with `queue_capacity` permits. A request takes a permit, runs the
blocking backend off the async runtime, and gives the permit back when it's done; if no permit is
available, the call is rejected right away with `ResourceExhausted`.

```text
  Client            gRPC handler          permits          backend
    │  ModelInfer        │                   │                │
    ├───────────────────►│                   │                │
    │                    │ validate dtype/   │                │
    │                    │ shape/batch       │                │
    │                    ├─── try_acquire ──►│                │
    │                    │                   │ (none free?)   │
    │  RESOURCE_EXHAUSTED│◄── unavailable ───┤                │
    │◄───────────────────┤                   │                │
    │                    │                   ├──── infer ────►│
    │                    │◄──────── output tensor ────────────┤
    │  ModelInferResponse│                   │                │
    │◄───────────────────┤                   │                │
```

So nereid sheds load rather than growing a backlog it has no real intention of getting through.
That's the more useful behavior for a client that is going to retry anyway: a fast rejection is
information, while a queue that keeps accepting work just converts overload into latency and hides
it. The subprocess backends get this for free, since the same permit bounds how many child
processes a model can have running at once.

## Request lifecycle

- **`ModelInfer` (Triton, unary + streaming).** The richest path: single- and multi-tensor, the
  full KServe dtype set. The handler checks the request datatype against the model's declared
  datatype, normalizes the batch dimension, builds the input tensors, takes a permit, and streams
  the typed output back.
- **`Nereid/Checkpoint` (streaming).** nereid's own path: the client streams a tensor in chunks,
  the server validates it, runs the model, and streams the output tensor back. A backend can also
  provide a richer stream — the Python backend uses this to interleave the model's stdout and
  stderr as log chunks while it runs.

## Backend dispatch

Everything above is backend-agnostic. The core talks to one `Backend` trait — a `platform()`
string, a blocking `infer()`, and an optional `checkpoint_stream()` — and never asks what kind of
engine is on the other side. Tensors cross that boundary as raw bytes plus a shape and a dtype,
rather than as any one framework's tensor type, which is what keeps the trait from quietly
becoming libtorch-shaped.

The backends aren't listed anywhere in the core either. Each one lives in its own folder under
`src/backends/`, registers itself at link time, and is discovered by the build; the core just
iterates those registrations to detect and load. Adding a backend means implementing the trait and
dropping in a folder, and the gRPC surfaces don't change at all. [Backends](backends.md) walks
through what goes in that folder.
