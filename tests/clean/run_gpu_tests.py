import datetime
import json
import logging
import platform
import time
import subprocess
from utilities import gpu_handling
from utilities.ResourceMonitor import ResourceMonitor
import cupy as cp
import os
from cupyx.scipy.signal import convolve2d

# Matrix Multiplication Test
def matrix_multiplication_test(batch_size, dtype=None):
    if dtype in [cp.float32, cp.float64]:
        # Use random float generation for float32 and float64
        A = cp.random.rand(batch_size, batch_size, dtype=dtype)
        B = cp.random.rand(batch_size, batch_size, dtype=dtype)
        return cp.dot(A, B)
    elif dtype in [cp.int32, cp.int16]:
        # Use random integer generation for int32 and int16
        A = cp.random.randint(0, 100, size=(batch_size, batch_size), dtype=dtype)
        B = cp.random.randint(0, 100, size=(batch_size, batch_size), dtype=dtype)
        return cp.dot(A, B)
    elif dtype == None:
        A = cp.random.rand(batch_size, batch_size)
        B = cp.random.rand(batch_size, batch_size)
        return cp.dot(A, B)
    else:
        raise TypeError(f"Unsupported data type: {dtype}")

# PCA Test
def pca_test(batch_size, n_components=2, dtype=None):
    """Optimized PCA on GPU using CuPy with enhanced memory efficiency."""
    if dtype in [cp.float32, cp.float64]:
        data = cp.random.rand(batch_size, batch_size, dtype=dtype)

    elif dtype in [cp.int32, cp.int16]:
        data = cp.random.randint(0, 100, size=(batch_size, batch_size), dtype=dtype)

    elif dtype is None:
        data = cp.random.rand(batch_size, batch_size)

    else:
        raise TypeError(f"Unsupported data type: {dtype}")

    data_centered = data - cp.mean(data, axis=0)
    cov_matrix = cp.dot(data_centered.T, data_centered) / (batch_size - 1)
    eigen_values, eigen_vectors = cp.linalg.eigh(cov_matrix)
    sorted_idx = cp.argsort(eigen_values)[::-1]
    eigen_vectors = eigen_vectors[:, sorted_idx]
    principal_components = cp.dot(data_centered, eigen_vectors[:, :n_components])

    return principal_components

# SVD Test
def svd_test(batch_size, dtype=None):
    if dtype in [cp.float32, cp.float64]:
        data = cp.random.rand(batch_size, batch_size, dtype=dtype)
        return cp.linalg.svd(data, full_matrices=False)
    elif dtype in [cp.int32, cp.int16]:
        data = cp.random.randint(0, 100, size=(batch_size, batch_size), dtype=dtype)
        return cp.linalg.svd(data, full_matrices=False)
    elif dtype == None:
        data = cp.random.rand(batch_size, batch_size)
        return cp.linalg.svd(data, full_matrices=False)
    else:
        raise TypeError(f"Unsupported data type: {dtype}")

# FFT Test (CPU using NumPy, GPU using CuPy)
def fft_test(batch_size, dtype=None):
    if dtype in [cp.float32, cp.float64]:
        data = cp.random.rand(batch_size, dtype=dtype)
        return cp.fft.fft(data)
    elif dtype in [cp.int32, cp.int16]:
        data = cp.random.randint(0, 100, size=(batch_size,), dtype=dtype)
        return cp.fft.fft(data)
    elif dtype == None:
        data = cp.random.rand(batch_size)
        return cp.fft.fft(data)
    else:
        raise TypeError(f"Unsupported data type: {dtype}")
    
def convolution_test(filter_size, sample_rate, dtype=None):
    """GPU-based 2D convolution using CuPy."""
    t = cp.linspace(0, 1, sample_rate, dtype=dtype or cp.float32)
    frequency = 440  # Hz
    sound_filter = cp.sin(2 * cp.pi * frequency * t)
    filter_2d = cp.reshape(sound_filter[:9], (3, 3)).astype(dtype or cp.float32)

    data = cp.random.rand(filter_size, filter_size).astype(dtype or cp.float32)

    # Optimized 2D convolution using CuPy's built-in function
    result = convolve2d(data, filter_2d, mode='same', boundary='fill', fillvalue=0)

    return result  # Keep result on GPU

# Function to run each test and collect results
def run_test(test_name, test_function, batch_size=0, filter_size=0, sample_rate=0, dtype=None):
    monitor = ResourceMonitor()
    system_info = monitor.get_system_info()
    
    # On Linux we can clear the cache before running the test
    if platform.system() == "Linux":
        monitor.clear_cache()

    monitor.start_monitoring()
    start_time = time.time()

    # Run the test based on the test type
    if test_name in ["matrix_multiplication_test", "pca_test", "svd_test", "fft_test"]:
        test_function(batch_size=batch_size, dtype=dtype)
    elif test_name == "convolution_test":
        test_function(filter_size=filter_size, sample_rate=sample_rate, dtype=dtype)
    else:
        exit(f"Test {test_name} is not supported")
    
    end_time = time.time()
    monitor.stop_monitoring()
    avg_usage = monitor.get_average_usage()
    elapsed_time = end_time - start_time    

    data_dict = {cp.int16 : "int16", cp.int32 : "int32", cp.float32 : "float32", cp.float64 : "float64", cp.double : "double", None:"Default"}
    # Collect system stats, including system info and epochs
    system_stats = {
        **system_info,  # CPU and GPU info
        "Run Type": "GPU",  # Indicate CPU or GPU run
        "Batch Size": batch_size,  # Batch size for tests
        "Filter Size": filter_size,
        "Sample Rate": sample_rate,
        "Epochs": 0,
        "Learning rate": 0,
        "Model Size": None,
        "Data Type": data_dict[dtype],
        "Elapsed Time": elapsed_time,
        "Result": "not relevant",
        **avg_usage
    }
    
    # Save results for this test
    monitor.save_results(system_stats, f"{test_name}", f"./results/{monitor.get_gpu_name()}/gpu/clean.json")

# Function to run all tests
def run_all_tests():
    data_type_list = [None, cp.int16, cp.int32, cp.float32, cp.double]

    for data_type in data_type_list:
        matrix_sizes = [64, 512, 1024, 2048, 4096, 8192]
        for matrix in matrix_sizes:
            print(f'Running Matrix Multiplication test, on GPU, matrix size (saved as batch size) : {matrix}')
            run_test("matrix_multiplication_test", matrix_multiplication_test, batch_size=matrix, filter_size=0, sample_rate=0, dtype=data_type)
            print(f'Running PCA test, on GPU, matrix size (saved as batch size) : {matrix}')
            run_test("pca_test", pca_test, batch_size=matrix, filter_size=0, sample_rate=0, dtype=data_type)
            print(f'Running SVD test, on GPU, matrix size (saved as batch size) : {matrix}')
            run_test("svd_test", svd_test, batch_size=matrix, filter_size=0, sample_rate=0, dtype=data_type)

        filter_sizes = [256, 1024, 2048, 4096, 8192]
        sample_rates = [16000, 44100]
        for filter in filter_sizes:
            for sample in sample_rates:
                print(f'Running Convolution test, on GPU, Filter size : {filter}, Sample rate : {sample}')
                run_test("convolution_test", convolution_test, batch_size=0, filter_size=filter, sample_rate=sample, dtype=data_type)

        batch_sizes = [256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
        for batch in batch_sizes:
            print(f'Running FFT test, on GPU, batch size : {batch}')
            run_test("fft_test", fft_test, batch_size=batch, filter_size=0, sample_rate=0, dtype=data_type)

if __name__ == "__main__":
    run_all_tests()