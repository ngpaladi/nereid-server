# Design: ingesting Triton model repositories

**Status:** draft / for discussion
**Scope:** let nereid serve an existing [Triton](https://github.com/triton-inference-server/server)
model repository — concretely, a repo laid out like
[`fastmachinelearning/sonic-models`](https://github.com/fastmachinelearning/sonic-models) —
with as little hand-editing as possible.

## Motivation

nereid already speaks the KServe v2 wire protocol, so a `tritonclient` can drive it unchanged.
The gap is on the *other* side: the models themselves. Today a model has to be laid out
nereid's way — a flat folder with a `model_inference.textproto` and the model file at its root.
Triton repositories are laid out differently, and there are a lot of them already in the wild.
`sonic-models` is the concrete target: ~dozens of models used across CMS/HEP SONIC deployments,
already packaged as a Triton repo.

The goal is for someone holding such a repo to point nereid at it and have the models nereid's
backends can actually run come up served — and for the ones they *can't* (yet) run to fail fast
with a message that says exactly why, in keeping with nereid's "strict at startup" stance.

## Background: the two layouts

### nereid today

```
<ml_backends_path>/<model_name>/
    model_inference.textproto     # the contract, at the folder root
    <model file(s)>               # .pt | .onnx | saved_model.pb + variables/ | main.py + requirements.txt | ...
```

- The **contract** (`src/backend/contract.rs`) is owned by the core and parsed from
  `model_inference.textproto`, in either a flat single-tensor form (`input_shape`, `output_shape`,
  `max_batch_size`, `data_type`) or a nested multi-tensor form (`input {}` / `output {}` blocks with
  `name`, `data_type`, `dims`). `dims` exclude the batch dimension; `max_batch_size > 0` enables it.
- The **backend** is auto-detected from file signatures at the folder root, or pinned with
  `backend:` in `nereid.yaml`. Detection predicates (from `src/backends/*/mod.rs`):

  | backend      | detects on                                         |
  |--------------|----------------------------------------------------|
  | `onnx`       | textproto + a `.onnx` file                         |
  | `torch`      | textproto + a `.pt` file                           |
  | `tensorflow` | textproto + `saved_model.pb` **and** `variables/`  |
  | `python`     | `main.py` + `requirements.txt`                     |
  | `cpp`        | textproto + `main.cpp` / `build.sh` / `model`      |
  | `cxx`        | textproto only (declaration-only, `backend: cxx`)  |

- `nereid.yaml` is authoritative for *which* models are served and for each one's `device`
  (`cpu` / `cuda[:idx]`) and `queue_capacity`. A model not listed is not served.

### A Triton repository (e.g. `sonic-models`)

```
models/<model_name>/
    config.pbtxt                  # Triton ModelConfig (text proto)
    1/                            # numbered version directories
        model.onnx                # onnxruntime_onnx
        model.graphdef            # tensorflow_graphdef
        model.savedmodel/         # tensorflow_savedmodel  (saved_model.pb + variables/)
        model.pt                  # pytorch_libtorch
        model.py                  # python (TritonPythonModel)
    2/
    <name>_labels.txt             # optional, referenced by output.label_filename
```

`config.pbtxt` is a superset of what nereid's contract carries. A representative one
(`models/particlenet/config.pbtxt`):

```protobuf
name: "particlenet"
platform: "onnxruntime_onnx"
max_batch_size: 160
dynamic_batching { preferred_batch_size: [ 80 ] }
input [
  { name: "pf_points"   data_type: TYPE_FP32 dims: [ 2, -1 ] },
  { name: "pf_features" data_type: TYPE_FP32 dims: [ 25, -1 ] },
  ...
]
output [
  { name: "softmax" data_type: TYPE_FP32 dims: [ 20 ] label_filename: "particlenet_labels.txt" }
]
optimization { graph: { level: -1 } }
```

## Mapping Triton → nereid

### Platform → backend

| Triton `platform`        | model file             | nereid backend | status |
|--------------------------|------------------------|----------------|--------|
| `onnxruntime_onnx`       | `model.onnx`           | `onnx`         | **direct** |
| `pytorch_libtorch`       | `model.pt`             | `torch`        | **direct** |
| `tensorflow_savedmodel`  | `model.savedmodel/`    | `tensorflow`   | **direct** |
| `tensorflow_graphdef`    | `model.graphdef`       | —              | **gap**: nereid's TF backend loads SavedModels only |
| `python`                 | `model.py`             | —              | **gap**: Triton's `TritonPythonModel` (`initialize`/`execute`, `pb_utils`) is a different contract from nereid's stdin/`NEREID_OUTPUT_PATH` subprocess model |
| `ensemble`               | (no weights)           | —              | **gap**: nereid has no ensemble/DAG scheduler |

In `sonic-models` all six appear — e.g. `particlenet*` (onnx), `particlenet_AK4_PT` (torch),
`facile_all_v5` (savedmodel), `deepmet`/`deeptau_*` (graphdef), `deeptau_python` (python),
`deeptau_ensemble` (ensemble). So the "direct" set is real and useful on its own, and the gaps
are real and must be reported, not silently skipped.

### Fields → `Contract`

| `config.pbtxt`                        | nereid `Contract`                     | notes |
|---------------------------------------|---------------------------------------|-------|
| `max_batch_size`                      | `max_batch_size`                      | same meaning; `0` ⇒ no batch dim |
| `input[] { name, data_type, dims }`   | `inputs: [TensorSpec { name, dtype, dims }]` | `dims` exclude batch in both |
| `output[] { name, data_type, dims }`  | `outputs: [TensorSpec { ... }]`       | authoritative dtypes ⇒ `strict_output_dtype: true` |
| `data_type: TYPE_FP32`                | `dtype: "FP32"`                       | strip the `TYPE_` prefix |
| `dims: [-1, ...]`                     | `dims: [-1, ...]`                     | `-1` = variable, identical semantics |

Everything the contract needs is present in `config.pbtxt`. dtype normalization is just dropping
`TYPE_`; only KServe fixed-width dtypes are supported today (`kserve_fixed_width`), so `TYPE_STRING`
/ `TYPE_BYTES` is a **gap** (fails fast).

### Fields nereid has no home for

- `dynamic_batching`, `optimization`, `warmup` — advisory/perf; **ignore with a log line**.
- `instance_group` — Triton's replica/device knob. nereid gets concurrency from
  `queue_capacity` and device from `nereid.yaml`. Could *seed defaults* from it (`KIND_GPU` ⇒
  `cuda`, `count` ⇒ `queue_capacity`); see open questions.
- `version_policy` — drives version resolution (below).
- `label_filename` / labels — nereid doesn't post-process to labels; **ignore for now** (carry the
  file along so it can be re-attached later).
- `reshape`, `optional`, `sequence_batching` — **gaps**; reject with an explicit message.

### Version directories

Triton keeps weights in numbered subdirs; nereid expects the model file at the folder root.
`version_policy` selects which are live (`latest { num_versions }` default = highest 1, `all {}`,
`specific { versions }`). nereid serves one artifact per model, so ingestion must **resolve a single
version** — default to the highest-numbered directory, honoring `specific` when set, and treating
`all {}` as "highest" (with a log line, since nereid won't expose multiple versions).

## Design options

**A. Offline import/convert tool.** A `nereid import-triton <repo>` subcommand (or script) that
reads each `config.pbtxt`, writes a nereid-native folder — generated `model_inference.textproto`,
the resolved version's model file linked/copied to the root, labels carried along — and emits a
starter `nereid.yaml`. Runtime is untouched; existing detection/loading just works.

- *Pros:* zero core-runtime change; output is inspectable and diffable; gaps surface at import as
  plain text; reuses everything.
- *Cons:* a materialization step and a second copy that can drift from the source repo; "point at a
  repo and go" needs a build step first.

**B. Native runtime ingestion.** Teach the core to read a Triton repo directly: a `config.pbtxt`
→ `Contract` reader alongside the textproto reader, version-dir resolution, and a `platform` →
backend map that bypasses signature detection and points the chosen backend at
`<version>/model.<ext>`.

- *Pros:* single source of truth; genuine "point nereid at the repo" drop-in; no generated copies.
- *Cons:* larger core change (second contract source, version resolution, backend file-path
  indirection, a broad and evolving Triton config surface to parse defensively).

**C. Hybrid (recommended).** Do the *reader* natively but keep it small, and keep an import path as
the escape hatch:

1. A `config.pbtxt` → `Contract` parser in the core — `config.pbtxt` becomes a second contract
   source, reusing the existing `Contract` types and all downstream dtype/shape/batch validation.
2. Version-dir-aware loading: when a model folder has a `config.pbtxt` and numbered version dirs,
   resolve the version and hand the backend the file inside it. Detection becomes: *if `config.pbtxt`
   is present, map `platform` → backend explicitly*; otherwise fall back to today's signature
   detection. This keeps "backends are discovered, not listed" intact — the platform map lives in a
   small core adapter, and each backend still just receives a directory + a file to load.
3. `import-triton` remains available for users who prefer to materialize + hand-tune, and as the
   pragmatic workaround for gap platforms (e.g. converting a graphdef to a SavedModel).

This gets the drop-in experience for the direct set while keeping the blast radius contained and the
strict-startup ethos (gaps ⇒ actionable error) unchanged.

## Recommended plan (phased)

- **Phase 0 — this doc.** Agree the approach, the supported set, and the gap policy.
- **Phase 1 — read + resolve.** `config.pbtxt` → `Contract`; `TYPE_` normalization; version
  resolution; `platform` → backend map for the direct set (onnx / torch / savedmodel). Fail fast on
  graphdef / python / ensemble / string dtype / `sequence_batching` with a message naming the
  platform and the model. Fixtures: a trimmed copy of a few `sonic-models` configs.
- **Phase 2 — serve a whole repo.** Point `ml_backends_path` at a Triton `models/` dir and serve all
  discoverable-and-supported models, with a "serve all" mode so a minimal `nereid.yaml` (or none) is
  needed; per-model `device`/`queue_capacity` overrides still come from `nereid.yaml`. Optionally
  seed defaults from `instance_group`.
- **Phase 3 — close gaps as warranted.** `tensorflow_graphdef` support (or a documented convert
  step); labels surfaced on the KServe metadata; revisit Triton-Python and ensembles as separate
  designs.

## What we explicitly do **not** do (yet)

Each of these fails at startup with a message naming the model and the unsupported feature, rather
than coming up half-working:

- `tensorflow_graphdef`, Triton-style `python` (`model.py`), and `ensemble` platforms.
- `TYPE_STRING` / `TYPE_BYTES` tensors, `sequence_batching`, `reshape`, `optional` inputs.
- Serving multiple versions of a model concurrently.

## Open questions

1. **Model selection.** Triton serves every model in the repo; nereid serves what
   `nereid.yaml` lists. Do we add a "serve all discovered" mode, auto-generate the model list, or
   keep an explicit list as the gate?
2. **Device & concurrency.** Keep `nereid.yaml` authoritative, or derive defaults from
   `instance_group` (`KIND_GPU`/`KIND_CPU`, `count`) and let `nereid.yaml` override?
3. **Reader native vs. import (A/B/C).** Confirm the hybrid, or start import-only to de-risk.
4. **Config parsing.** Hand-write a minimal `config.pbtxt` reader for the fields we use, or pull in
   the Triton `model_config.proto` and parse it as textproto (bigger dependency, full fidelity)?
5. **Gap handling granularity.** Hard-fail the whole startup on any unsupported model, or serve the
   supported ones and refuse the rest with a warning? (nereid's default is strict; a "serve a whole
   repo" flow may want the latter.)
