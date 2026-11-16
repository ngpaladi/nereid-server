# Prometheus metrics

nereid serves the **inference request** metrics of NVIDIA Triton — same names, same labels, same
units — on a plain-HTTP `/metrics` endpoint. A Prometheus scrape job, Grafana dashboard, or alert
rule written against Triton's `nv_inference_*` series reads nereid without changes.

Only the request metrics are implemented. Triton's GPU/CPU/memory utilization gauges, its
response-cache metrics, and its opt-in latency summaries and histograms are not.

## Enabling

Set `server.http_addr`; when it is absent no HTTP is served. It must differ from
`server.bind_addr`, which speaks gRPC. Triton's convention is port `8002`. The same address also
serves the [KServe v2 health endpoints](triton.md#http-health-endpoints) for Kubernetes probes.

```yaml
server:
  bind_addr: "[::]:50051"
  http_addr: "[::]:8002"
  ml_backends_path: "ml-backends"
```

Then:

```bash
curl -s http://localhost:8002/metrics
```

A minimal Prometheus job:

```yaml
scrape_configs:
  - job_name: nereid
    static_configs:
      - targets: ["nereid-host:8002"]
```

In the container, publish the port alongside the gRPC one:
`docker run -p 50051:50051 -p 8002:8002 …`.

## What is exported

Every series carries Triton's `model` and `version` labels. nereid serves a single implicit model
version, so `version` is always `"1"`. All series exist from startup, at zero, for every model in
`nereid.yaml` — so `rate()` has a baseline before the first request, and a model that has never
been called is visibly idle rather than missing.

### Counts

| Metric | Type | Meaning |
|---|---|---|
| `nv_inference_request_success` | counter | Successful requests. A batched request counts once. |
| `nv_inference_request_failure` | counter | Failed requests, with a `reason` label (below). |
| `nv_inference_count` | counter | Inferences performed: a request with batch size *n* adds *n*. |
| `nv_inference_exec_count` | counter | Backend executions. nereid runs one execution per request, so this tracks `request_success`; the ratio `count / exec_count` is the average batch size. |
| `nv_inference_pending_request_count` | gauge | Requests accepted for a model but not yet executing. |

The `reason` label on `nv_inference_request_failure` takes Triton's four values:

| `reason` | When |
|---|---|
| `REJECTED` | The model's queue was full (`queue_capacity`) and the request was shed with `RESOURCE_EXHAUSTED`. |
| `CANCELED` | The client went away before the response — the call was cancelled, or it stopped reading a `Checkpoint` stream. |
| `BACKEND` | The model failed: the backend's `infer` returned an error, a Python model exited non-zero or wrote a malformed output, or the output contradicted the contract. |
| `OTHER` | Everything else — chiefly a request that failed validation (wrong shape, datatype, byte length, or tensor name). |

### Latencies

Cumulative microsecond counters, as in Triton. Divide the `rate()` of a duration by the `rate()`
of `nv_inference_request_success` for the average per request. They accumulate only for successful
requests.

| Metric | Covers |
|---|---|
| `nv_inference_request_duration_us` | End to end, from the request naming a model to the response being ready. |
| `nv_inference_queue_duration_us` | Waiting for a worker thread after a permit was granted. nereid sheds load rather than queueing (see [Backpressure](architecture.md#backpressure)), so this is normally tiny. |
| `nv_inference_compute_input_duration_us` | Inside the backend: turning the request's tensors into the framework's inputs (and moving them to the model's device). |
| `nv_inference_compute_infer_duration_us` | Inside the backend: executing the model. |
| `nv_inference_compute_output_duration_us` | Inside the backend: extracting the framework's outputs into response tensors (and copying them back from the device). |

### How the phases are measured

The phases follow Triton's definitions and order, *queue → compute_input → compute_infer →
compute_output*, and — as in Triton — the three compute phases are reported by the backend and lie
entirely inside it. What the server does around the backend is in `request_duration_us` only:
decoding and validating the request (on the `Checkpoint` path, receiving the streamed chunks), and
validating and serializing the response. So the four phases sum to somewhat less than
`request_duration_us`, and the difference is the frontend's overhead.

Where each backend draws the lines:

| Backend | `compute_input` | `compute_infer` | `compute_output` |
|---|---|---|---|
| Torch | Building libtorch tensors, moving them to the device | The forward pass (on CUDA, synchronized before the clock stops) | Copying outputs to host bytes |
| ONNX | Building ONNX Runtime values | `Session::run` | Extracting output tensors |
| TensorFlow | Building and feeding TF tensors | `Session::run` | Fetching output tensors |
| Python, C++ (`main.cpp`) | — | The child process, spawn to exit (the input is piped to it while it runs) | Reading and parsing the output tensor file |
| C++ (compiled-in) | — | The whole call: the model takes and returns raw bytes | — |

A Python model's `Checkpoint` stream runs as one opaque step, so its whole span is `compute_infer`.

Two differences from Triton remain, both because nereid has no scheduler queue or instance groups:

- `queue` is only the wait for a worker thread. A request that waits for another request to the
  same in-process model to finish (models run one execution at a time; `queue_capacity` above 1
  admits more) spends that wait in `compute_infer`, where Triton would count it as `queue`.
- There is no dynamic batcher, so every execution serves exactly one request.

## Which requests count

Both front doors are accounted the same way: a KServe v2 `ModelInfer` (unary or streamed) and a
native `Nereid/Checkpoint` call each become one request against their model. A request is
attributed the moment it names a configured model; from then on it ends as exactly one success or
one failure. A request for a model that isn't in `nereid.yaml` fails with `NOT_FOUND` and is
counted nowhere, since there is no model series to put it under.

## Useful queries

```promql
# Requests per second, per model
sum by (model) (rate(nv_inference_request_success[1m]))

# Failure ratio, per model and reason
sum by (model, reason) (rate(nv_inference_request_failure[5m]))
  / ignoring (reason) group_left
  sum by (model) (rate(nv_inference_request_success[5m]) + rate(nv_inference_request_failure[5m]))

# Average model execution time (ms), per model
rate(nv_inference_compute_infer_duration_us[5m]) / rate(nv_inference_request_success[5m]) / 1000

# Average batch size
rate(nv_inference_count[5m]) / rate(nv_inference_exec_count[5m])
```
