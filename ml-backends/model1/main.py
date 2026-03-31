from time import sleep
import numpy as np

def dot(a, b):
    return np.dot(a,b)

print("Starting model1 processing...")
for i in range(1, 10):
    print("Dot product:", dot(np.array([i, i+1]), np.array([i+2, i+3])))
    sleep(0.3)
print("Finished processing model1")