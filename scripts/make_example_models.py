#!/usr/bin/env python3
"""Generate the example ONNX and TensorFlow models used by the native backends.

These are tiny "add one" models used both as runnable examples and as test
fixtures for the ONNX/TensorFlow backends:

  onnxadd  — ONNX,       output = input + 1   (float32, shape [-1, 4])
  tfadd    — TF SavedModel, output = input + 1 (float32, shape [-1, 4])

ONNX generation needs only `onnx` (`pip install onnx`). TensorFlow generation
needs `tensorflow` and is skipped with a note when it isn't installed.

Usage:
    python scripts/make_example_models.py [--backends-dir ml-backends]
"""

import argparse
import os
import sys

INPUT_DIM = 4
TEXTPROTO = "input_shape: [%d]\noutput_shape: [%d]\nmax_batch_size: 8\n" % (
    INPUT_DIM,
    INPUT_DIM,
)


def make_onnx(model_dir: str) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import numpy as np

    os.makedirs(model_dir, exist_ok=True)
    one = numpy_helper.from_array(np.ones(INPUT_DIM, dtype=np.float32), name="one")
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, INPUT_DIM])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, INPUT_DIM])
    node = helper.make_node("Add", inputs=["input", "one"], outputs=["output"])
    graph = helper.make_graph([node], "add_one", [inp], [out], initializer=[one])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9  # compatible with ONNX Runtime 1.24 (ort 2.0)
    onnx.checker.check_model(model)
    onnx.save(model, os.path.join(model_dir, "model.onnx"))
    with open(os.path.join(model_dir, "model_inference.textproto"), "w") as f:
        f.write(TEXTPROTO)
    print("wrote", model_dir)


def make_tf(model_dir: str) -> None:
    try:
        import tensorflow as tf
    except ImportError:
        print("tensorflow not installed; skipping tfadd (pip install tensorflow)")
        return

    class AddOne(tf.Module):
        @tf.function(
            input_signature=[tf.TensorSpec([None, INPUT_DIM], tf.float32, name="input")]
        )
        def __call__(self, x):
            return {"output": x + 1.0}

    os.makedirs(model_dir, exist_ok=True)
    tf.saved_model.save(
        AddOne(), model_dir, signatures={"serving_default": AddOne().__call__}
    )
    with open(os.path.join(model_dir, "model_inference.textproto"), "w") as f:
        f.write(TEXTPROTO)
    print("wrote", model_dir)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends-dir", default="ml-backends")
    args = ap.parse_args()
    make_onnx(os.path.join(args.backends_dir, "onnxadd"))
    make_tf(os.path.join(args.backends_dir, "tfadd"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
