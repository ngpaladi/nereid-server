"""Non-float tensor-output Python model fixture: output = input + 1 (int32).

Exercises the Python byte-passthrough dtype path: the request declares INT32, so
the raw stdin bytes are little-endian int32, and the framed output header
declares `int32`. Uses struct's standard `<i` (guaranteed 4-byte) so element
sizes match nereid's INT32 mapping. No third-party dependencies.
"""

import os
import struct
import sys

raw = sys.stdin.buffer.read()
count = len(raw) // 4
values = struct.unpack("<%di" % count, raw) if count else ()

result = [v + 1 for v in values]

shape = os.environ["NEREID_INPUT_SHAPE"]
with open(os.environ["NEREID_OUTPUT_PATH"], "wb") as f:
    f.write(("int32 " + shape + "\n").encode("utf-8"))
    f.write(struct.pack("<%di" % len(result), *result))

print("pyaddint added 1 to", count, "int32 values")
