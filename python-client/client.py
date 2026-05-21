#!/usr/bin/env python3
import argparse
import importlib
import math
import multiprocessing as mp
import random
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


ShapeDimSpec = int | tuple[int, int]


@dataclass(frozen=True)
class ClientConfig:
    host: str
    port: int
    model: str
    shape: list[int] | None
    shapes: list[list[int]] | None
    random_shape: list[ShapeDimSpec] | None
    producers: int
    inputs_per_producer: int
    chunk_bytes: int
    sleep_seconds: float


def load_config(path: Path) -> ClientConfig:
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("config must be a YAML mapping")

    allowed_keys = {
        "host",
        "port",
        "model",
        "shape",
        "shapes",
        "random_shape",
        "producers",
        "inputs_per_producer",
        "chunk_bytes",
        "sleep_seconds",
    }
    unknown_keys = sorted(set(raw) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"unknown config fields: {', '.join(unknown_keys)}")

    config = ClientConfig(
        host=require_str(raw, "host"),
        port=require_int(raw, "port"),
        model=require_str(raw, "model"),
        shape=optional_shape(raw, "shape"),
        shapes=optional_shapes(raw, "shapes"),
        random_shape=optional_random_shape(raw, "random_shape"),
        producers=require_int(raw, "producers"),
        inputs_per_producer=require_int(raw, "inputs_per_producer"),
        chunk_bytes=require_int(raw, "chunk_bytes"),
        sleep_seconds=optional_number(raw, "sleep_seconds", 0),
    )

    shape_modes = [
        config.shape is not None,
        config.shapes is not None,
        config.random_shape is not None,
    ]
    if sum(shape_modes) != 1:
        raise ValueError("exactly one of shape, shapes, or random_shape must be configured")

    if config.port < 1 or config.port > 65535:
        raise ValueError("port must be between 1 and 65535")
    if config.producers < 1:
        raise ValueError("producers must be >= 1")
    if config.inputs_per_producer < 1:
        raise ValueError("inputs_per_producer must be >= 1")
    if config.chunk_bytes < 1:
        raise ValueError("chunk_bytes must be >= 1")
    if config.sleep_seconds < 0:
        raise ValueError("sleep_seconds must be >= 0")

    return config


def require_str(raw: dict, key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def require_int(raw: dict, key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def validate_shape(value: object, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list of positive integers")
    if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in value):
        raise ValueError(f"{field} must contain only positive integers")
    return list(value)


def optional_shape(raw: dict, key: str) -> list[int] | None:
    if key not in raw:
        return None
    return validate_shape(raw.get(key), key)


def optional_shapes(raw: dict, key: str) -> list[list[int]] | None:
    if key not in raw:
        return None
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list of shapes")
    return [validate_shape(shape, f"{key}[{index}]") for index, shape in enumerate(value)]


def optional_random_shape(raw: dict, key: str) -> list[ShapeDimSpec] | None:
    if key not in raw:
        return None
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"{key} must be a non-empty list of fixed dimensions or [min, max] ranges"
        )

    dims: list[ShapeDimSpec] = []
    for index, dim in enumerate(value):
        field = f"{key}[{index}]"
        if isinstance(dim, bool):
            raise ValueError(f"{field} must be a positive integer or [min, max] range")
        if isinstance(dim, int):
            if dim <= 0:
                raise ValueError(f"{field} must be positive")
            dims.append(dim)
            continue

        if not isinstance(dim, list) or len(dim) != 2:
            raise ValueError(f"{field} must be a positive integer or [min, max] range")
        lower, upper = dim
        if (
            isinstance(lower, bool)
            or isinstance(upper, bool)
            or not isinstance(lower, int)
            or not isinstance(upper, int)
        ):
            raise ValueError(f"{field} range bounds must be integers")
        if lower <= 0 or upper <= 0 or lower > upper:
            raise ValueError(f"{field} range must satisfy 0 < min <= max")
        dims.append((lower, upper))

    return dims


def optional_number(raw: dict, key: str, default: float) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(value)


def describe_shape_source(config: ClientConfig) -> str:
    if config.shape is not None:
        return f"shape={config.shape}"
    if config.shapes is not None:
        return f"shapes={config.shapes}"
    return f"random_shape={config.random_shape}"


def choose_shape(config: ClientConfig) -> list[int]:
    if config.shape is not None:
        return list(config.shape)
    if config.shapes is not None:
        return list(random.choice(config.shapes))
    if config.random_shape is None:
        raise RuntimeError("no shape source configured")
    return [
        random.randint(dim[0], dim[1]) if isinstance(dim, tuple) else dim
        for dim in config.random_shape
    ]


def generate_proto_modules(repo_root: Path, out_dir: Path) -> None:
    from grpc_tools import protoc

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
    sys.path.insert(0, str(out_dir))
    pb2 = importlib.import_module("inference_pb2")
    pb2_grpc = importlib.import_module("inference_pb2_grpc")
    return pb2, pb2_grpc


def random_tensor_values(shape: Sequence[int]) -> list[float]:
    return [random.random() for _ in range(math.prod(shape))]


def encode_f32(values: Sequence[float]) -> bytes:
    return b"".join(struct.pack("<f", value) for value in values)


def request_stream(
    pb2,
    model: str,
    shape: Sequence[int],
    data: bytes,
    chunk_bytes: int,
) -> Iterator:
    yield pb2.CheckpointRequest(
        meta=pb2.CheckpointMeta(model_name=model, output_file="")
    )

    total_chunks = (len(data) + chunk_bytes - 1) // chunk_bytes
    for index, start in enumerate(range(0, len(data), chunk_bytes)):
        yield pb2.CheckpointRequest(
            chunk=pb2.TensorChunk(
                tensor_name="input",
                shape=shape,
                data=data[start : start + chunk_bytes],
                chunk_index=index,
                end_of_tensor=(index + 1 == total_chunks),
            )
        )


def run_producer(config: ClientConfig, producer_id: int) -> int:
    import grpc

    repo_root = Path(__file__).resolve().parent.parent
    pb2, pb2_grpc = load_proto_modules(repo_root)
    target = f"{config.host}:{config.port}"
    label = f"edproducer-{producer_id}"

    print(
        f"[{label}] start target={target} model={config.model} "
        f"inputs={config.inputs_per_producer} {describe_shape_source(config)}"
    )

    completed = 0
    with grpc.insecure_channel(target) as channel:
        stub = pb2_grpc.NereidStub(channel)

        for index in range(config.inputs_per_producer):
            input_number = index + 1
            shape = choose_shape(config)
            values = random_tensor_values(shape)
            data = encode_f32(values)

            try:
                responses = stub.Checkpoint(
                    request_stream(
                        pb2,
                        config.model,
                        shape,
                        data,
                        config.chunk_bytes,
                    )
                )
                saw_done = False
                for response in responses:
                    if response.done:
                        saw_done = True

                if not saw_done:
                    print(f"[{label}] failed input={input_number} reason=missing_done")
                    return 1
            except grpc.RpcError as err:
                print(
                    f"[{label}] failed input={input_number} "
                    f"shape={shape} code={err.code()} details={err.details()}"
                )
                return 1

            completed += 1
            if completed == config.inputs_per_producer or completed % 100 == 0:
                print(
                    f"[{label}] ok inputs={completed}/{config.inputs_per_producer} "
                    f"last_shape={shape}"
                )

            if config.sleep_seconds > 0 and input_number < config.inputs_per_producer:
                time.sleep(config.sleep_seconds)

    print(f"[{label}] complete inputs={completed}")
    return 0


def producer_entry(config: ClientConfig, producer_id: int) -> None:
    raise SystemExit(run_producer(config, producer_id))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run mock ED producers against Nereid")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "client.yaml"),
        help="path to YAML config file",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    repo_root = Path(__file__).resolve().parent.parent
    generate_proto_modules(repo_root, repo_root / "python-client" / "_generated")

    procs: list[tuple[int, mp.Process]] = []
    for producer_id in range(1, config.producers + 1):
        proc = mp.Process(
            target=producer_entry,
            args=(config, producer_id),
            name=f"edproducer-{producer_id}",
        )
        proc.start()
        procs.append((producer_id, proc))
        print(f"[client] started edproducer-{producer_id} pid={proc.pid}")

    failures = 0
    for producer_id, proc in procs:
        proc.join()
        if proc.exitcode == 0:
            continue
        failures += 1
        print(f"[client] edproducer-{producer_id} exited rc={proc.exitcode}")

    succeeded = config.producers - failures
    total_inputs = config.producers * config.inputs_per_producer
    print(
        f"[client] summary producers_ok={succeeded}/{config.producers} "
        f"producer_failures={failures} requested_inputs={total_inputs}"
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
