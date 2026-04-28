#!/usr/bin/env python3
# This module is configured via globals and invoked by spawn_edproducers.py.
# It runs one producer loop over a single persistent gRPC connection.
import importlib
import math
import random
import struct
import sys
import time
from pathlib import Path
from typing import Iterator, List, Sequence

import grpc
from grpc_tools import protoc

# Global config (overridden by spawn_edproducers.py before each producer starts).
ITERATIONS = 1
PRODUCER_ID = "edproducer-1"
SHAPE = "1,16"
RANDOM_MIN = 0.0
RANDOM_MAX = 1.0
RANDOM_PRECISION = 6

SEED = None
SLEEP_SECONDS = 0.0
HOST = "[::1]"
PORT = 50051
MODEL = "model3"
CHUNK_BYTES = 64 * 1024


def log(msg: str) -> None:
    print(f"[ed-producer][{PRODUCER_ID}] {msg}")


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
    dims = [int(p) for p in parts]
    if any(d <= 0 for d in dims):
        raise ValueError("all shape dimensions must be positive")
    return dims


def build_random_values(count: int, lo: float, hi: float, precision: int) -> List[float]:
    values = [random.uniform(lo, hi) for _ in range(count)]
    if precision >= 0:
        return [round(v, precision) for v in values]
    return values


def encode_f32_le(values: Sequence[float]) -> bytes:
    return b"".join(struct.pack("<f", v) for v in values)


def request_stream(pb2, model_name: str, shape: Sequence[int], data: bytes, chunk_bytes: int) -> Iterator:
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


def run_producer() -> int:
    if ITERATIONS <= 0:
        raise ValueError("ITERATIONS must be > 0")
    if RANDOM_MIN > RANDOM_MAX:
        raise ValueError("RANDOM_MIN cannot be greater than RANDOM_MAX")
    if SEED is not None:
        random.seed(SEED)

    here = Path(__file__).resolve()
    repo_root = here.parent.parent
    pb2, pb2_grpc = load_proto_modules(repo_root)

    shape = parse_shape(SHAPE)
    expected = math.prod(shape)
    target = f"{HOST}:{PORT}"

    log(
        f"start model={MODEL} target={target} iterations={ITERATIONS} "
        f"shape={shape} mode=persistent-connection"
    )

    with grpc.insecure_channel(target) as channel:
        stub = pb2_grpc.NereidStub(channel)
        for i in range(ITERATIONS):
            iteration = i + 1
            values = build_random_values(expected, RANDOM_MIN, RANDOM_MAX, RANDOM_PRECISION)
            data = encode_f32_le(values)

            try:
                responses = stub.Checkpoint(
                    request_stream(pb2, MODEL, shape, data, CHUNK_BYTES)
                )
                saw_done = False
                for resp in responses:
                    if resp.done:
                        saw_done = True
                if not saw_done:
                    log(f"request failed iteration={iteration} reason=missing_done")
                    return 1
            except grpc.RpcError as err:
                log(
                    f"request failed iteration={iteration} code={err.code()} details={err.details()}"
                )
                return 1

            if iteration % 100 == 0 or iteration == ITERATIONS:
                log(f"request ok iteration={iteration}/{ITERATIONS}")

            if SLEEP_SECONDS > 0 and iteration < ITERATIONS:
                time.sleep(SLEEP_SECONDS)

    log(f"complete iterations={ITERATIONS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_producer())
