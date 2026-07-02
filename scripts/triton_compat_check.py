#!/usr/bin/env python3
"""Cross-implementation Triton (KServe v2) compatibility check for nereid.

This drives a running nereid server with a STOCK ``tritonclient`` — built from
NVIDIA Triton's own generated stubs, NOT from nereid's vendored
``proto/grpc_service.proto``. That is the point: the in-repo Rust tests pair
nereid's client with nereid's server, which only proves self-consistency. Only a
client generated from Triton's own proto can prove the wire format actually
matches Triton (a mis-copied field number is wrong identically on both of
nereid's sides and round-trips cleanly, but fails here).

Usage:
    pip install -r scripts/requirements.txt
    # start the server first, e.g. with a nereid.yaml exposing the models below
    python scripts/triton_compat_check.py --url 127.0.0.1:50051 \
        --model pymul:mul --model pyaddint:addint --model model3 \
        --stream pymul

Each ``--model`` is ``NAME`` or ``NAME:MODE``. For every model the script checks
readiness, metadata, and one ``infer()``. Modes assert real arithmetic:
``mul`` -> ``input*2+1`` in FP32 (the ``ml-backends/pymul`` fixture);
``addint`` -> ``input+1`` in INT32 (``ml-backends/pyaddint``, the non-float
datatype path). ``--stream NAME`` additionally drives the streaming
``ModelStreamInfer`` RPC against a ``mul`` model and matches each response to its
request by id.
"""
import argparse
import queue
import sys

import numpy as np
import tritonclient.grpc as grpcclient


def _concrete_shape(meta_shape):
    """Variable (-1) dims become 1 so we can build a concrete request."""
    return [d if d > 0 else 1 for d in meta_shape]


def check_model(client, spec: str) -> None:
    name, _, mode = spec.partition(":")
    if not client.is_model_ready(name):
        raise SystemExit(f"FAIL: model {name!r} is not ready")

    meta = client.get_model_metadata(name)
    inp_meta = meta.inputs[0]
    out_name = meta.outputs[0].name
    shape = _concrete_shape(inp_meta.shape)
    count = int(np.prod(shape))

    if mode == "addint":
        values = np.arange(10, 10 + count, dtype=np.int32).reshape(shape)
        datatype = "INT32"
    elif mode == "addint64":
        values = np.arange(10, 10 + count, dtype=np.int64).reshape(shape)
        datatype = "INT64"
    else:
        values = np.arange(1, count + 1, dtype=np.float32).reshape(shape)
        datatype = "FP32"

    inp = grpcclient.InferInput(inp_meta.name, shape, datatype)
    inp.set_data_from_numpy(values)
    out = client.infer(model_name=name, inputs=[inp], request_id=f"compat-{name}")
    arr = out.as_numpy(out_name)

    if arr is None:
        raise SystemExit(f"FAIL: model {name!r} returned no output")
    print(f"  {name}: in {shape} {datatype} -> out {list(arr.shape)} {arr.dtype} ok", flush=True)

    if mode == "mul":
        expected = values * 2.0 + 1.0
        if not np.array_equal(arr, expected):
            raise SystemExit(f"FAIL: {name!r} expected input*2+1 {expected!r}, got {arr!r}")
        print(f"  {name}: output == input*2+1 (FP32) ✓", flush=True)
    elif mode in ("addint", "addint64"):
        want_dtype = np.int32 if mode == "addint" else np.int64
        expected = values + 1
        if arr.dtype != want_dtype or not np.array_equal(arr, expected):
            raise SystemExit(f"FAIL: {name!r} expected input+1 {want_dtype} {expected!r}, got {arr!r}")
        print(f"  {name}: output == input+1 ({arr.dtype}) ✓", flush=True)


def check_multi(client, name: str) -> None:
    """Drive a two-input/two-output model (sum, prod) = (a+b, a*b)."""
    if not client.is_model_ready(name):
        raise SystemExit(f"FAIL: multi model {name!r} is not ready")
    a = np.array([[1, 2, 3, 4]], dtype=np.float32)
    b = np.array([[10, 20, 30, 40]], dtype=np.float32)
    ta = grpcclient.InferInput("a", [1, 4], "FP32"); ta.set_data_from_numpy(a)
    tb = grpcclient.InferInput("b", [1, 4], "FP32"); tb.set_data_from_numpy(b)
    outs = [grpcclient.InferRequestedOutput("sum"), grpcclient.InferRequestedOutput("prod")]
    res = client.infer(model_name=name, inputs=[ta, tb], outputs=outs, request_id=f"multi-{name}")
    got_sum, got_prod = res.as_numpy("sum"), res.as_numpy("prod")
    if not np.array_equal(got_sum, a + b) or not np.array_equal(got_prod, a * b):
        raise SystemExit(f"FAIL: {name!r} multi outputs wrong: sum={got_sum!r} prod={got_prod!r}")
    print(f"  {name}: multi (sum, prod) == (a+b, a*b) ✓", flush=True)


def check_stream(client, name: str) -> None:
    """Drive ModelStreamInfer against a `mul` model and verify each response."""
    if not client.is_model_ready(name):
        raise SystemExit(f"FAIL: stream model {name!r} is not ready")
    shape = _concrete_shape(client.get_model_metadata(name).inputs[0].shape)
    count = int(np.prod(shape))

    results: "queue.Queue" = queue.Queue()
    client.start_stream(callback=lambda result, error: results.put((result, error)))
    n = 3
    inputs_by_id = {}
    for i in range(n):
        values = np.full(shape, float(i + 1), dtype=np.float32).reshape([count])[:count]
        values = values.reshape(shape)
        inputs_by_id[str(i)] = values
        inp = grpcclient.InferInput("input", shape, "FP32")
        inp.set_data_from_numpy(values)
        client.async_stream_infer(model_name=name, inputs=[inp], request_id=str(i))
    client.stop_stream()

    got = 0
    while not results.empty():
        result, error = results.get()
        if error is not None:
            raise SystemExit(f"FAIL: stream error: {error}")
        req_id = result.get_response().id
        arr = result.as_numpy("output")
        expected = inputs_by_id[req_id] * 2.0 + 1.0
        if not np.array_equal(arr, expected):
            raise SystemExit(f"FAIL: stream id {req_id} expected {expected!r}, got {arr!r}")
        got += 1
    if got != n:
        raise SystemExit(f"FAIL: expected {n} stream responses, got {got}")
    print(f"  {name}: stream returned {n} correct responses ✓", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="127.0.0.1:50051")
    parser.add_argument("--model", dest="models", action="append", default=[],
                        help="NAME or NAME:mul|addint; repeatable")
    parser.add_argument("--stream", dest="streams", action="append", default=[],
                        help="NAME of a mul model to stream-check; repeatable")
    parser.add_argument("--multi", dest="multis", action="append", default=[],
                        help="NAME of a (sum,prod)=(a+b,a*b) model; repeatable")
    args = parser.parse_args()
    models = args.models or ["pymul:mul"]

    client = grpcclient.InferenceServerClient(url=args.url)
    if not client.is_server_live() or not client.is_server_ready():
        raise SystemExit("FAIL: server is not live/ready")
    print(f"server live & ready at {args.url}", flush=True)

    for spec in models:
        check_model(client, spec)
    for name in args.multis:
        check_multi(client, name)
    for name in args.streams:
        check_stream(client, name)

    print("TRITON_COMPAT_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
