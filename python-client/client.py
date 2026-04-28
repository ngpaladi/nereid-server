#!/usr/bin/env python3
import argparse
import importlib
import math
import random
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import grpc
from grpc_tools import protoc


def generate_proto_modules(repo_root: Path, out_dir: Path) -> None:
    proto_dir = repo_root / "proto"
    proto_file = proto_dir / "inference.proto"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{proto_dir}",
            f"--python_out={out_dir}",
            f"--grpc_python_out={out_dir}",
            str(proto_file),
        ]
    )
    if result != 0:
        raise RuntimeError("failed to generate Python gRPC code from proto/inference.proto")


def load_proto_modules(repo_root: Path):
    out_dir = repo_root / "python-client" / "_generated"
    generate_proto_modules(repo_root, out_dir)

    sys.path.insert(0, str(out_dir))
    pb2 = importlib.import_module("inference_pb2")
    pb2_grpc = importlib.import_module("inference_pb2_grpc")
    return pb2, pb2_grpc


def parse_shape(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("shape cannot be empty")
    shape = [int(p) for p in parts]
    if any(d <= 0 for d in shape):
        raise ValueError("all shape dimensions must be positive")
    return shape


def parse_values(raw: str) -> List[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("values cannot be empty")
    return [float(p) for p in parts]


def encode_f32_le(values: Sequence[float]) -> bytes:
    return b"".join(struct.pack("<f", v) for v in values)


def decode_f32_le(data: bytes) -> List[float]:
    if len(data) % 4 != 0:
        raise ValueError(f"output byte length {len(data)} is not divisible by 4")
    return [x[0] for x in struct.iter_unpack("<f", data)]


def reshape(flat: Sequence[float], shape: Sequence[int]):
    if not shape:
        return list(flat)

    total = 1
    for dim in shape:
        total *= dim
    if total != len(flat):
        raise ValueError(f"shape {list(shape)} expects {total} values but got {len(flat)}")

    def build(offset: int, dims: Sequence[int]):
        if len(dims) == 1:
            width = dims[0]
            return list(flat[offset : offset + width]), offset + width

        width = math.prod(dims[1:])
        out = []
        cursor = offset
        for _ in range(dims[0]):
            part, cursor = build(cursor, dims[1:])
            out.append(part)
        return out, cursor

    nested, _ = build(0, shape)
    return nested


def request_stream(pb2, model_name: str, shape: Sequence[int], data: bytes, chunk_bytes: int):
    yield pb2.CheckpointRequest(
        meta=pb2.CheckpointMeta(model_name=model_name, output_file="")
    )

    if not data:
        yield pb2.CheckpointRequest(
            chunk=pb2.TensorChunk(
                tensor_name="input",
                shape=shape,
                data=b"",
                chunk_index=0,
                end_of_tensor=True,
            )
        )
        return

    total_chunks = (len(data) + chunk_bytes - 1) // chunk_bytes
    for i, start in enumerate(range(0, len(data), chunk_bytes)):
        payload = data[start : start + chunk_bytes]
        yield pb2.CheckpointRequest(
            chunk=pb2.TensorChunk(
                tensor_name="input",
                shape=shape,
                data=payload,
                chunk_index=i,
                end_of_tensor=(i + 1 == total_chunks),
            )
        )


def build_random_values(count: int, lo: float, hi: float, precision: int) -> List[float]:
    values = [random.uniform(lo, hi) for _ in range(count)]
    if precision >= 0:
        return [round(v, precision) for v in values]
    return values


def log(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream an input tensor to Nereid/Checkpoint and print output tensor values"
    )
    parser.add_argument("--host", default="[::1]", help="gRPC host")
    parser.add_argument("--port", type=int, default=50051, help="gRPC port")
    parser.add_argument("--model", default="model3", help="model name")
    parser.add_argument("--shape", default="1,16", help="input shape, e.g. 1,16")
    parser.add_argument(
        "--values",
        default=",".join(str(i) for i in range(1, 17)),
        help="comma-separated float values",
    )
    parser.add_argument(
        "--chunk-bytes",
        type=int,
        default=64 * 1024,
        help="bytes per streamed tensor chunk",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress verbose output")
    parser.add_argument("--iterations", type=int, default=1, help="number of requests to send")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="sleep between requests")
    parser.add_argument("--randomize-values", action="store_true", help="generate random values per request")
    parser.add_argument("--random-min", type=float, default=0.0, help="random generation lower bound")
    parser.add_argument("--random-max", type=float, default=1.0, help="random generation upper bound")
    parser.add_argument("--random-precision", type=int, default=6, help="rounding digits; <0 disables rounding")
    parser.add_argument("--seed", type=int, default=None, help="optional random seed")

    args = parser.parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0")
    if args.random_min > args.random_max:
        raise ValueError("--random-min cannot be greater than --random-max")

    repo_root = Path(__file__).resolve().parent.parent
    pb2, pb2_grpc = load_proto_modules(repo_root)

    shape = parse_shape(args.shape)
    expected = math.prod(shape)
    base_values: List[float] = []
    if not args.randomize_values:
        base_values = parse_values(args.values)
        if len(base_values) != expected:
            raise ValueError(
                f"input values count mismatch: shape {shape} expects {expected}, got {len(base_values)}"
            )
    if args.seed is not None:
        random.seed(args.seed)

    target = f"{args.host}:{args.port}"
    log(f"connecting to {target}", args.quiet)

    with grpc.insecure_channel(target) as channel:
        stub = pb2_grpc.NereidStub(channel)

        for i in range(args.iterations):
            iteration = i + 1
            values = (
                build_random_values(
                    expected,
                    args.random_min,
                    args.random_max,
                    args.random_precision,
                )
                if args.randomize_values
                else base_values
            )
            data = encode_f32_le(values)

            output_shape: List[int] = []
            output_chunks: Dict[int, bytes] = {}
            responses = stub.Checkpoint(
                request_stream(pb2, args.model, shape, data, args.chunk_bytes)
            )

            for resp in responses:
                if resp.chunk:
                    log(f"server: {resp.chunk}", args.quiet)

                if resp.HasField("output_chunk"):
                    oc = resp.output_chunk
                    if oc.shape:
                        output_shape = list(oc.shape)
                    output_chunks[int(oc.chunk_index)] = bytes(oc.data)

                if resp.done:
                    log(f"done=true exit_code={resp.exit_code}", args.quiet)

            if not output_chunks:
                log("no output tensor chunks received", args.quiet)
                continue

            if not args.quiet:
                output_bytes = b"".join(output_chunks[j] for j in sorted(output_chunks.keys()))
                output_values = decode_f32_le(output_bytes)
                pretty = reshape(output_values, output_shape) if output_shape else output_values
                print(f"iteration {iteration}/{args.iterations}")
                print(f"output shape: {output_shape if output_shape else '[unknown]'}")
                print("output values:")
                print(pretty)

            if args.sleep_seconds > 0 and iteration < args.iterations:
                time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
