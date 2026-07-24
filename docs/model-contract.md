# Model contract

Every model is a folder under `<server.ml_backends_path>/<model_name>/`. What files the folder must
contain depends on the backend — the **Model folder** column of the
[backends table](backends.md) is the canonical list, so it isn't repeated here.

Two things every folder has in common, whatever the backend:

- a `model_inference.textproto` (below) declaring the model's tensor shapes and datatype — the
  subprocess backends (Python and `cpp`) additionally require it to declare `output_shape`, since
  every reply is a typed tensor;
- exactly one backend's worth of files. The server works out which backend from the folder's
  contents; if a folder matches more than one (or none), set `backend:` in `nereid.yaml` to settle
  it and only that backend's files are checked. The `cxx` backend is the exception that always
  needs declaring — its folder is just a textproto, which carries no signature to detect (see
  [Backends](backends.md#compile-time-c-cxx)).

## `model_inference.textproto`

Declares the model's tensor shapes and datatype. The simplest, single-tensor form:

```text
input_shape: [16]
output_shape: [16]
max_batch_size: 8
data_type: "FP32"   # optional; defaults to FP32
```

- **`input_shape`** excludes the batch dimension.
- **`max_batch_size`** — if greater than `0`, a request may include a leading batch dimension, which
  the server enforces is at most `max_batch_size`. A request that omits the batch dimension (sending
  just the `input_shape` rank) is auto-expanded to batch size 1 — e.g. with `input_shape: [16]`, a
  request shape of `[16]` is treated as `[1, 16]`. If `max_batch_size` is `0` or omitted, the
  request shape must match the `input_shape` rank directly.
- Fixed dimensions must match exactly; a `-1` dimension is client-provided and may be any positive
  value.

**Batching rules** — with `input_shape: [16]` and `max_batch_size: 8`:

- a request shape of `[4, 16]` is accepted (batch 4);
- a request shape of `[9, 16]` is **rejected** (exceeds `max_batch_size`);
- a request shape of `[16]` is accepted and treated as `[1, 16]`.

### Multi-tensor models

A model that consumes or produces more than one named tensor uses nested `input {}` / `output {}`
blocks:

```text
input  { name: "a" data_type: "FP32" dims: [16] }
input  { name: "b" data_type: "FP32" dims: [16] }
output { name: "sum"  data_type: "FP32" dims: [16] }
output { name: "prod" data_type: "FP32" dims: [16] }
max_batch_size: 8
```

The request's named inputs are bound in the order the textproto lists them and passed positionally
to the model, so the `input {}` blocks must match the model's expected input order.

## Subprocess tensor contract

The Python backend — and any other backend that runs a model as a child process — speaks one
language-agnostic contract. Nothing about it is Python-specific, so a model can be written in
anything that can read stdin and write a file.

**Input (optional).** When `input_shape` is declared, the server validates the request tensor and
delivers it to the process:

- **stdin** — the raw tensor bytes, little-endian, row-major.
- **`NEREID_INPUT_SHAPE`** — the (batch-normalized) shape, comma-separated, e.g. `1,16`.
- **`NEREID_INPUT_DTYPE`** — the element dtype, e.g. `float32`.

**Output (required).** The model writes its output tensor to the file named by
**`NEREID_OUTPUT_PATH`** in a self-describing framed format: a UTF-8 header line
`"<dtype> d0,d1,...\n"` followed by the raw little-endian bytes. The server validates it against the
declared `output_shape` (and the request batch size) before returning it.

A minimal `main.py` (no third-party dependencies) that reads an input tensor and replies with one:

```python
import os, struct, sys

raw = sys.stdin.buffer.read()  # present when input_shape is declared
values = struct.unpack("<%df" % (len(raw) // 4), raw) if raw else ()
result = [float(sum(values))]  # ... run inference ...

with open(os.environ["NEREID_OUTPUT_PATH"], "wb") as f:
    f.write(("float32 %d\n" % len(result)).encode("utf-8"))
    f.write(struct.pack("<%df" % len(result), *result))
```

A model that exits `0` without writing a valid output tensor has broken the contract, so the
request fails. Exiting cleanly is not the same as having produced an answer, and the server would
otherwise have nothing to send back.

## `nereid.yaml`

`nereid.yaml` is loaded at startup from the repository root. It selects which models are exposed,
each model's device and queue size, and the server bind address:

```yaml
server:
  bind_addr: "[::1]:50051"
  ml_backends_path: "ml-backends"

models:
  - name: "model3"
    device: "cpu"       # "cpu", "cuda" (GPU 0), or "cuda:<index>"
    queue_capacity: 16  # must be > 0
```
