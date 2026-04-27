#!/usr/bin/env python3
import argparse
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence


def parse_shape(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("shape cannot be empty")
    dims = [int(p) for p in parts]
    if any(d <= 0 for d in dims):
        raise ValueError("all shape dimensions must be positive")
    return dims


def parse_values(raw: str) -> List[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("values cannot be empty")
    return [float(p) for p in parts]


def build_random_values(count: int, lo: float, hi: float, precision: int) -> List[float]:
    values = [random.uniform(lo, hi) for _ in range(count)]
    if precision >= 0:
        return [round(v, precision) for v in values]
    return values


def csv(values: Sequence[float]) -> str:
    return ",".join(f"{v:g}" for v in values)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Dummy EDproducer: invoke python-client/client.py repeatedly with a configurable "
            "shape and generated/random tensor values."
        )
    )
    parser.add_argument("--iterations", type=int, default=1, help="number of client runs")
    parser.add_argument("--producer-id", default="edproducer-1", help="label for log lines")
    parser.add_argument("--shape", default="1,16", help="input tensor shape, e.g. 1,16")
    parser.add_argument(
        "--values",
        default="",
        help="fixed comma-separated values (overrides random generation)",
    )
    parser.add_argument("--random-min", type=float, default=0.0, help="minimum random value")
    parser.add_argument("--random-max", type=float, default=1.0, help="maximum random value")
    parser.add_argument(
        "--random-precision",
        type=int,
        default=6,
        help="decimal digits for random values; use negative to disable rounding",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="delay between iterations",
    )
    parser.add_argument("--host", default="[::1]", help="gRPC host")
    parser.add_argument("--port", type=int, default=50051, help="gRPC port")
    parser.add_argument("--model", default="model3", help="model name")
    parser.add_argument("--chunk-bytes", type=int, default=64 * 1024, help="chunk bytes")
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="python executable used to invoke client.py",
    )
    parser.add_argument(
        "--client-path",
        default="",
        help="path to python-client/client.py (default: auto-detect from repo root)",
    )
    args = parser.parse_args()

    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0")
    if args.random_min > args.random_max:
        raise ValueError("--random-min cannot be greater than --random-max")

    if args.seed is not None:
        random.seed(args.seed)

    shape = parse_shape(args.shape)
    expected = math.prod(shape)

    here = Path(__file__).resolve()
    repo_root = here.parent.parent
    client_path = Path(args.client_path) if args.client_path else (repo_root / "python-client" / "client.py")
    client_path = client_path.resolve()
    if not client_path.exists():
        raise FileNotFoundError(f"client script not found at {client_path}")

    fixed_values: List[float] = []
    if args.values.strip():
        fixed_values = parse_values(args.values)
        if len(fixed_values) != expected:
            raise ValueError(
                f"--values count mismatch: shape {shape} expects {expected}, got {len(fixed_values)}"
            )

    for i in range(args.iterations):
        iteration = i + 1
        values = fixed_values or build_random_values(
            expected, args.random_min, args.random_max, args.random_precision
        )

        cmd = [
            args.python_bin,
            str(client_path),
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--model",
            args.model,
            "--shape",
            args.shape,
            "--values",
            csv(values),
            "--chunk-bytes",
            str(args.chunk_bytes),
        ]

        print(
            f"[{args.producer_id}] iteration {iteration}/{args.iterations} "
            f"shape={shape} values={values}"
        )
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"[{args.producer_id}] iteration {iteration} failed with exit code {result.returncode}"
            )
            return result.returncode

        if args.sleep_seconds > 0 and iteration < args.iterations:
            time.sleep(args.sleep_seconds)

    print(f"[{args.producer_id}] complete ({args.iterations} iterations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
