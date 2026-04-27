#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spawn multiple dummy ED producers to simulate concurrent clients"
    )
    parser.add_argument("--producers", type=int, default=2, help="number of ED producers")
    parser.add_argument(
        "--iterations-per-producer",
        type=int,
        default=1,
        help="client runs per producer",
    )
    parser.add_argument("--shape", default="1,16", help="input tensor shape, e.g. 1,16")
    parser.add_argument("--random-min", type=float, default=0.0, help="minimum random value")
    parser.add_argument("--random-max", type=float, default=1.0, help="maximum random value")
    parser.add_argument(
        "--random-precision",
        type=int,
        default=6,
        help="decimal digits for random values; use negative to disable rounding",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="base seed; producer i gets seed-base + i",
    )
    parser.add_argument("--host", default="[::1]", help="gRPC host")
    parser.add_argument("--port", type=int, default=50051, help="gRPC port")
    parser.add_argument("--model", default="model3", help="model name")
    parser.add_argument("--chunk-bytes", type=int, default=64 * 1024, help="chunk bytes")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="delay between iterations inside each producer",
    )
    parser.add_argument(
        "--stagger-seconds",
        type=float,
        default=0.0,
        help="delay between spawning each producer process",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop all producers when any producer fails",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="python executable used to invoke producer/client scripts",
    )
    parser.add_argument(
        "--producer-script",
        default="",
        help="path to scripts/EDproducer.py (default: auto-detect from repo root)",
    )
    return parser


def terminate_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.producers <= 0:
        raise ValueError("--producers must be > 0")
    if args.iterations_per_producer <= 0:
        raise ValueError("--iterations-per-producer must be > 0")

    here = Path(__file__).resolve()
    repo_root = here.parent.parent
    producer_script = (
        Path(args.producer_script).resolve()
        if args.producer_script
        else (repo_root / "scripts" / "EDproducer.py")
    )
    if not producer_script.exists():
        raise FileNotFoundError(f"ED producer script not found at {producer_script}")

    procs: List[Tuple[str, subprocess.Popen]] = []

    def shutdown_all() -> None:
        for _, proc in procs:
            terminate_proc(proc)

    try:
        for idx in range(args.producers):
            producer_name = f"edproducer-{idx + 1}"
            cmd = [
                args.python_bin,
                str(producer_script),
                "--iterations",
                str(args.iterations_per_producer),
                "--producer-id",
                producer_name,
                "--shape",
                args.shape,
                "--random-min",
                str(args.random_min),
                "--random-max",
                str(args.random_max),
                "--random-precision",
                str(args.random_precision),
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--model",
                args.model,
                "--chunk-bytes",
                str(args.chunk_bytes),
                "--sleep-seconds",
                str(args.sleep_seconds),
                "--python-bin",
                args.python_bin,
            ]
            if args.seed_base is not None:
                cmd.extend(["--seed", str(args.seed_base + idx)])

            proc = subprocess.Popen(cmd)
            procs.append((producer_name, proc))
            print(f"[spawner] started {producer_name} pid={proc.pid}")

            if args.stagger_seconds > 0 and idx < args.producers - 1:
                time.sleep(args.stagger_seconds)

        failures: List[Tuple[str, int]] = []

        while procs:
            active: List[Tuple[str, subprocess.Popen]] = []
            for name, proc in procs:
                rc = proc.poll()
                if rc is None:
                    active.append((name, proc))
                    continue

                print(f"[spawner] {name} exited rc={rc}")
                if rc != 0:
                    failures.append((name, rc))
                    if args.fail_fast:
                        print("[spawner] fail-fast enabled; terminating remaining producers")
                        for _, other in active:
                            terminate_proc(other)
                        active = []
                        break

            procs = active
            if procs:
                time.sleep(0.2)

        if failures:
            first_name, first_rc = failures[0]
            print(f"[spawner] completed with failures (first: {first_name} rc={first_rc})")
            return first_rc

        print("[spawner] all producers completed successfully")
        return 0

    except KeyboardInterrupt:
        print("[spawner] interrupted, shutting down producers")
        shutdown_all()
        return 130
if __name__ == "__main__":
    raise SystemExit(main())
