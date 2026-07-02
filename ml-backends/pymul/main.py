"""Tensor-output Python model fixture: output = input * 2 + 1.

Demonstrates the nereid Python tensor contract (the inverse of the stdin input
contract):
  * the validated input tensor arrives on stdin as little-endian float32,
    with its shape in NEREID_INPUT_SHAPE;
  * the output tensor is written to the file named by NEREID_OUTPUT_PATH in a
    self-describing framed format: a UTF-8 header line "float32 d0,d1,...\n"
    followed by the raw little-endian float32 bytes.

No third-party dependencies, so the venv build stays cheap.
"""

import os
import struct
import sys

shape = [int(d) for d in os.environ["NEREID_INPUT_SHAPE"].split(",")]
raw = sys.stdin.buffer.read()
count = len(raw) // 4
values = struct.unpack("<%df" % count, raw) if count else ()

result = [v * 2.0 + 1.0 for v in values]

with open(os.environ["NEREID_OUTPUT_PATH"], "wb") as f:
    header = "float32 " + ",".join(str(d) for d in shape) + "\n"
    f.write(header.encode("utf-8"))
    f.write(struct.pack("<%df" % len(result), *result))

print("pymul processed", count, "values")
