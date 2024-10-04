import numpy as np
import cupy as cp
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray
import time
import os


# Function to clear the cache (Linux only)
def clear_cache():
    os.system("sync")  # Ensure filesystem buffers are flushed
    os.system("echo 3 | tee /proc/sys/vm/drop_caches")  # Clear cache
    print("Cache cleared.")

# Number of iterations and size of the array
iterations = 1000
array_size = 100000000  # 100 million elements

# Benchmark with NumPy (CPU)
def benchmark_numpy():
    arr = np.zeros(array_size, dtype=np.float32)  # Create an array with 1 million elements

    start_time = time.time()
    for _ in range(iterations):
        arr += 1  # Increment the array
    end_time = time.time()

    return end_time - start_time

# Benchmark with PyCUDA (GPU)
def benchmark_pycuda():
    arr = np.zeros(array_size, dtype=np.float32)
    arr_gpu = gpuarray.to_gpu(arr)  # Send array to the GPU

    # Kernel to increment the array on the GPU
    increment_kernel = SourceModule("""
    __global__ void increment(float *a, int size) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < size) {
            a[idx] += 1;
        }
    }
    """).get_function("increment")

    start_time = time.time()
    threads_per_block = 256
    blocks_per_grid = (array_size + (threads_per_block - 1)) // threads_per_block

    for _ in range(iterations):
        increment_kernel(arr_gpu, np.int32(array_size), block=(threads_per_block, 1, 1), grid=(blocks_per_grid, 1))

    cuda.Context.synchronize()  # Synchronize before measuring end time
    end_time = time.time()

    return end_time - start_time

# Benchmark with CuPy (GPU)
def benchmark_cupy():
    arr = cp.zeros(array_size, dtype=cp.float32)  # Create a CuPy array on the GPU

    start_time = time.time()
    for _ in range(iterations):
        arr += 1  # Increment the array
    end_time = time.time()

    return end_time - start_time

# Run benchmarks
if __name__ == "__main__":
    
    # Get the number of CUDA devices available
    num_devices = cuda.Device.count()
    cuda.init()
    print(f"Number of available CUDA devices: {num_devices}")
    if cuda:
        print("Cuda init success!")

    for i in range(num_devices):
        device = cuda.Device(i)
        print(f"Device {i}: {device.name()}, Compute Capability: {device.compute_capability()}")
    
    
    # Benchmark NumPy (CPU)
    clear_cache()  # Clear the cache before starting
    numpy_time = benchmark_numpy()
    print(f"NumPy (CPU) time: {numpy_time:.6f} seconds")

    # Benchmark PyCUDA (GPU)
    clear_cache()  # Clear the cache before starting
    pycuda_time = benchmark_pycuda()
    print(f"PyCUDA (GPU) time: {pycuda_time:.6f} seconds")

    # Benchmark CuPy (GPU)
    clear_cache()  # Clear the cache before starting
    cupy_time = benchmark_cupy()
    print(f"CuPy (GPU) time: {cupy_time:.6f} seconds")

