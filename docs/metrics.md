# Prometheus metrics

nereid serves the **inference request** metrics of NVIDIA Triton — same names, same labels, same
units — on a plain-HTTP `/metrics` endpoint. A Prometheus scrape job, Grafana dashboard, or alert
rule written against Triton's `nv_inference_*` series reads nereid without changes.

Only the request metrics are implemented. Triton's GPU/CPU/memory utilization gauges, its
response-cache metrics, and its opt-in latency summaries and histograms are not.

## Enabling

Set `server.metrics_addr`; when it is absent nothing is served. It must differ from
`server.bind_addr`, which speaks gRPC. Triton's convention is port `8002`:

```yaml
server:
  bind_addr: "[::]:50051"
  metrics_addr: "[::]:8002"
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
| `nv_inference_compute_input_duration_us` | Validating and assembling the input tensors. On the `Checkpoint` path this includes receiving the streamed chunks. |
| `nv_inference_queue_duration_us` | Waiting for a worker thread after a permit was granted. nereid sheds load rather than queueing (see [Backpressure](architecture.md#backpressure)), so this is normally tiny. |
| `nv_inference_compute_infer_duration_us` | The backend's `infer` — the model itself. For a Python model's `Checkpoint` stream, the whole subprocess run. |
| `nv_inference_compute_output_duration_us` | Validating and serializing the outputs. |

Note that `compute_input` and `compute_output` are measured where nereid does that work — at the
gRPC boundary — whereas Triton measures them inside the backend. Backends that convert bytes to
framework tensors (libtorch, ONNX Runtime) do so inside `infer`, so that time lands in
`compute_infer` here.

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
