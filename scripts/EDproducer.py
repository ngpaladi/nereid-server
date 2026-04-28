#!/usr/bin/env python3
# This module is configured via globals and invoked by spawn_edproducers.py.
# It runs one producer loop and calls python-client/client.py each iteration.
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence

# Global config (overridden by spawn_edproducers.py before each producer starts).
ITERATIONS = 1
PRODUCER_ID = "edproducer-1"
SHAPE = "1,16"
VALUES = ""
RANDOM_MIN = 0.0
RANDOM_MAX = 1.0
RANDOM_PRECISION = 6

SEED = None
SLEEP_SECONDS = 0.0
HOST = "[::1]"
PORT = 50051
MODEL = "model3"
CHUNK_BYTES = 64 * 1024
PYTHON_BIN = sys.executable
CLIENT_PATH = ""


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


def run_producer() -> int:
    if ITERATIONS <= 0:
        raise ValueError("ITERATIONS must be > 0")
    if RANDOM_MIN > RANDOM_MAX:
        raise ValueError("RANDOM_MIN cannot be greater than RANDOM_MAX")

    if SEED is not None:
        random.seed(SEED)

    shape = parse_shape(SHAPE)
    expected = math.prod(shape)

    here = Path(__file__).resolve()
    repo_root = here.parent.parent
    client_path = Path(CLIENT_PATH) if CLIENT_PATH else (repo_root / "python-client" / "client.py")
    client_path = client_path.resolve()
    if not client_path.exists():
        raise FileNotFoundError(f"client script not found at {client_path}")

    fixed_values: List[float] = []
    if VALUES.strip():
        fixed_values = parse_values(VALUES)
        if len(fixed_values) != expected:
            raise ValueError(
                f"VALUES count mismatch: shape {shape} expects {expected}, got {len(fixed_values)}"
            )

    for i in range(ITERATIONS):
        iteration = i + 1
        values = fixed_values or build_random_values(
            expected, RANDOM_MIN, RANDOM_MAX, RANDOM_PRECISION
        )

        cmd = [
            PYTHON_BIN,
            str(client_path),
            "--host",
            HOST,
            "--port",
            str(PORT),
            "--model",
            MODEL,
            "--shape",
            SHAPE,
            "--values",
            csv(values),
            "--chunk-bytes",
            str(CHUNK_BYTES),
        ]

        print(
            f"[{PRODUCER_ID}] iteration {iteration}/{ITERATIONS} "
            f"shape={shape} values={values}"
        )
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"[{PRODUCER_ID}] iteration {iteration} failed with exit code {result.returncode}"
            )
            return result.returncode

        if SLEEP_SECONDS > 0 and iteration < ITERATIONS:
            time.sleep(SLEEP_SECONDS)

    print(f"[{PRODUCER_ID}] complete ({ITERATIONS} iterations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_producer())
