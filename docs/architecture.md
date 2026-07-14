# Architecture

## Two gRPC surfaces, one address

nereid binds a single address (`server.bind_addr`) and serves two gRPC services on it:

- **`inference.Nereid`** — nereid's native service: `HealthCheck`, `ViewModels`, and the streaming
  `Checkpoint` inference RPC.
- **`inference.GRPCInferenceService`** — NVIDIA Triton's KServe v2 surface, vendored verbatim so a
  stock `tritonclient` can drive nereid unchanged (see [Triton compatibility](triton.md)).

Both are backed by the same models and the same `ModelManager`; they're just two front doors.

## The ModelManager

At startup the server reads `nereid.yaml`, then builds a `ModelManager` that owns every configured
model. For each model it:

1. Resolves the model folder under `server.ml_backends_path`.
2. **Detects the backend** from the folder contents — or honours an explicit `backend:` in the
   config (see [Backends](backends.md)).
3. Prepares the backend (e.g. creates a Python venv, loads a `.pt` into libtorch, opens an ONNX
   Runtime session), reading the model's `model_inference.textproto` contract.
4. Registers a **handle** plus a per-model **worker** and a **bounded queue**.

If any model can't be prepared — a missing file, a bad contract, or a backend the server wasn't
built with — startup fails with an actionable error rather than serving a broken model.

## Per-model workers and backpressure

Each in-process model (Rust `.pt`, ONNX, TensorFlow) runs on its **own OS worker thread** with a
bounded `sync_channel` sized by the model's `queue_capacity`. A request is enqueued as a job with a
oneshot reply channel; the worker runs the forward pass and sends the result back.

```text
  Client            gRPC handler          bounded queue        model worker
    │  ModelInfer        │                     │                    │
    ├───────────────────►│                     │                    │
    │                    │ validate dtype/     │                    │
    │                    │ shape/batch         │                    │
    │                    ├──── try_send(job) ─►│                    │
    │                    │                     │  (full?)           │
    │  RESOURCE_EXHAUSTED│◄─── full ───────────┤                    │
    │◄───────────────────┤                     │                    │
    │                    │                     ├──── job ──────────►│
    │                    │                     │                    │ forward pass
    │                    │◄──────── output tensor ──────────────────┤
    │  ModelInferResponse│                     │                    │
    │◄───────────────────┤                     │                    │
```

When the queue is full the call is rejected immediately with `ResourceExhausted` — nereid sheds
load rather than growing an unbounded backlog. The Python and C++ subprocess backends use the same
idea, bounding concurrent subprocesses with a semaphore of `queue_capacity` permits.

## Request lifecycle

- **`ModelInfer` (Triton, unary + streaming).** The richest path: single- **and** multi-tensor,
  the full KServe dtype set. The handler validates the request datatype against the model's
  declared datatype, batch-normalizes the shape, builds the input tensor(s), enqueues, and streams
  the typed output back.
- **`Nereid/Checkpoint` (streaming).** nereid's native path: the client streams a tensor in chunks;
  the server validates it, runs the model, and streams the output tensor (and, for Python models,
  stdout/stderr as log chunks) back.

## Backend dispatch

Dispatch is by model kind. In-process backends share a small `NativeModel` seam (raw-bytes tensors,
not any one framework's tensor type), so ONNX, TensorFlow, and other in-process engines reuse the
same worker, queue, and validation. Subprocess backends (Python, C++) share a common
stdin/framed-output tensor contract. Adding a backend is implementing one of those two shapes and
registering it — the gRPC surfaces don't change.
