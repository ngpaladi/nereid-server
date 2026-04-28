#!/usr/bin/env python3
# This script is the single entrypoint for launching multiple dummy ED producers.
# Configure values below, then run this script; it directly calls EDproducer functions.
import multiprocessing as mp
import sys
import time
from typing import List, Tuple

import EDproducer

# Global config for spawner behavior.
PRODUCERS = 2
ITERATIONS_PER_PRODUCER = 1
SHAPE = "1,16"
RANDOM_MIN = 0.0
RANDOM_MAX = 1.0
RANDOM_PRECISION = 6
SEED_BASE = None
HOST = "[::1]"
PORT = 50051
MODEL = "model3"
CHUNK_BYTES = 64 * 1024
SLEEP_SECONDS = 0.0
STAGGER_SECONDS = 0.0
FAIL_FAST = False
PYTHON_BIN = sys.executable
CLIENT_PATH = ""


def terminate_proc(proc: mp.Process) -> None:
    if not proc.is_alive():
        return
    proc.terminate()
    proc.join(timeout=3)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=3)


def producer_entry(producer_name: str, seed: int | None) -> int:
    EDproducer.ITERATIONS = ITERATIONS_PER_PRODUCER
    EDproducer.PRODUCER_ID = producer_name
    EDproducer.SHAPE = SHAPE
    EDproducer.VALUES = ""
    EDproducer.RANDOM_MIN = RANDOM_MIN
    EDproducer.RANDOM_MAX = RANDOM_MAX
    EDproducer.RANDOM_PRECISION = RANDOM_PRECISION
    EDproducer.SEED = seed
    EDproducer.SLEEP_SECONDS = SLEEP_SECONDS
    EDproducer.HOST = HOST
    EDproducer.PORT = PORT
    EDproducer.MODEL = MODEL
    EDproducer.CHUNK_BYTES = CHUNK_BYTES
    EDproducer.PYTHON_BIN = PYTHON_BIN
    EDproducer.CLIENT_PATH = CLIENT_PATH
    return EDproducer.run_producer()


def main() -> int:
    if PRODUCERS <= 0:
        raise ValueError("PRODUCERS must be > 0")
    if ITERATIONS_PER_PRODUCER <= 0:
        raise ValueError("ITERATIONS_PER_PRODUCER must be > 0")

    procs: List[Tuple[str, mp.Process]] = []

    def shutdown_all() -> None:
        for _, proc in procs:
            terminate_proc(proc)

    try:
        for idx in range(PRODUCERS):
            producer_name = f"edproducer-{idx + 1}"
            seed = None if SEED_BASE is None else (SEED_BASE + idx)
            proc = mp.Process(target=producer_entry, args=(producer_name, seed), name=producer_name)
            proc.start()
            procs.append((producer_name, proc))
            print(f"[spawner] started {producer_name} pid={proc.pid}")

            if STAGGER_SECONDS > 0 and idx < PRODUCERS - 1:
                time.sleep(STAGGER_SECONDS)

        failures: List[Tuple[str, int]] = []

        while procs:
            active: List[Tuple[str, mp.Process]] = []
            for name, proc in procs:
                if proc.exitcode is None:
                    active.append((name, proc))
                    continue

                rc = proc.exitcode
                print(f"[spawner] {name} exited rc={rc}")
                if rc != 0:
                    failures.append((name, rc))
                    if FAIL_FAST:
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
