from time import sleep
import os
import struct

import numpy as np


def dot(a, b):
    return np.dot(a, b)


print("Starting model2 processing...")
results = []
for i in range(1, 10):
    value = float(dot(np.array([i, i]), np.array([i, i])))
    print("Dot product:", value)
    results.append(value)
    sleep(0.3)
print("Finished processing model2")

# Every Python reply is a typed tensor: write the 9 dot products as a
# little-endian float32 tensor of shape [9] to NEREID_OUTPUT_PATH.
with open(os.environ["NEREID_OUTPUT_PATH"], "wb") as f:
    f.write(("float32 %d\n" % len(results)).encode("utf-8"))
    f.write(struct.pack("<%df" % len(results), *results))
