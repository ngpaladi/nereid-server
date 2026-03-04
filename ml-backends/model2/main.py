from time import sleep
import numpy as np

def dot(a, b):
    return np.dot(a,b)

print("Starting model2 processing...")
for i in range(1, 10):
    print("Dot product:", dot(np.array([i, i]), np.array([i, i])))
    sleep(0.3)
print("Finished processing model2")