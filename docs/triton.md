# Triton compatibility (KServe v2)

nereid exposes NVIDIA Triton's `inference.GRPCInferenceService` on the same bind address as its
native service, so a stock `tritonclient` (or any KServe v2 speaker) can drive nereid without code
changes.

## Wire parity, not authoring parity

The package, service name, RPC names, and message field numbers in `proto/grpc_service.proto` are
vendored **verbatim** from
[Triton's `grpc_service.proto`](https://github.com/triton-inference-server/common/blob/main/protobuf/grpc_service.proto),
so the wire format is byte-compatible. That's parity on the wire and nothing more: models are still
written against nereid's own contracts (a `.pt` file, a `main.py`, a SavedModel, …), not against
Triton's model repository layout.

## Implemented RPCs

`ServerLive`, `ServerReady`, `ModelReady`, `ServerMetadata`, `ModelMetadata`, unary `ModelInfer`,
and streaming `ModelStreamInfer`.

- **Datatypes** are the KServe fixed-width kinds: `FP16/32/64`, `INT8/16/32/64`, `UINT8`, `BOOL`,
  `BF16`, plus `UINT16/32/64` on the native (ONNX/TensorFlow) path.
- **Inputs** may be sent as `raw_input_contents` (raw little-endian bytes, preferred) or, for FP32,
  the typed `contents`.
- nereid serves a single implicit model version, `"1"`.

> **Not implemented (deferred):** Rust `UINT16/32/64` and `BYTES`; the HTTP/REST `/v2` mirror;
> Prometheus metrics; and the repository / config / statistics RPCs.

## Verifying compatibility

It would be easy to convince ourselves of compatibility by having nereid's own client talk to
nereid's own server, but that only proves the two agree with each other. So the check uses a stock
`tritonclient`, built from Triton's proto stubs rather than our vendored copy, which makes it a
real cross-implementation test. The committed checker `scripts/triton_compat_check.py` runs it
across the example models:

```bash
pip install -r scripts/requirements.txt
# with a nereid.yaml exposing pymul, pyaddint, rustint, multi and model3, and the server running:
python scripts/triton_compat_check.py --url 127.0.0.1:50051 \
    --model pymul:mul --model pyaddint:addint --model rustint:addint64 --model model3 \
    --multi multi --stream pymul
# -> ... rustint: output == input+1 (int64) ✓ ... multi (sum, prod) == (a+b, a*b) ✓ ... TRITON_COMPAT_OK
```

## Poking it with `grpcurl`

The native `Nereid` service is easy to hit directly:

```bash
# Health check
grpcurl -plaintext -import-path ./proto -proto inference.proto -d '{}' \
    '[::1]:50051' inference.Nereid/HealthCheck

# List configured models
grpcurl -plaintext -import-path ./proto -proto inference.proto -d '{}' \
    '[::1]:50051' inference.Nereid/ViewModels
```
