from numba import cuda
import numpy as np


@cuda.jit
def caller(a, f):
    f(a)


@cuda.jit
def callback(a):
    a[0] = 1


@cuda.jit
def kernel(a):
    caller(a, callback)


a = np.zeros(1)
kernel[1, 1](a)
print(a)
