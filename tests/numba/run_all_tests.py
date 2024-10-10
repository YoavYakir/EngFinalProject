import time
import numpy as np
import cupy as cp
import os
import sys
import platform
import random
from numba import jit
from ...scripts.utility_functions import ResourceMonitor

# Iterative Test (CPU only with Numba)
@jit(nopython=True)
def iterative_test():
    result = 0
    for i in range(1000000):
        result += random.randint(0,10)
    return result

# Matrix Multiplication Test (CPU using Numba, GPU using CuPy)
def matrix_multiplication_test(on_gpu, batch_size=100):
    if on_gpu:
        # GPU run using CuPy
        with cp.cuda.Device(0):
            A = cp.random.rand(batch_size, batch_size)
            B = cp.random.rand(batch_size, batch_size)
            return cp.dot(A, B)
    else:
        # CPU run using NumPy and Numba
        return matrix_multiplication_cpu(batch_size)

@jit(nopython=True)
def matrix_multiplication_cpu(batch_size):
    A = np.random.rand(batch_size, batch_size)
    B = np.random.rand(batch_size, batch_size)
    return np.dot(A, B)

# PCA Test (CPU using Numba, GPU using CuPy)
def pca_test(on_gpu, batch_size=100, n_components=2):
    if on_gpu:
        with cp.cuda.Device(0):
            data = cp.random.rand(batch_size, batch_size)
            data_centered = data - cp.mean(data, axis=0)
            cov_matrix = cp.cov(data_centered.T)
            eigen_values, eigen_vectors = cp.linalg.eigh(cov_matrix)
            sorted_idx = cp.argsort(eigen_values)[::-1]
            eigen_vectors = eigen_vectors[:, sorted_idx]
            return cp.dot(data_centered, eigen_vectors[:, :n_components])
    else:
        # CPU run using Numba
        return pca_cpu(batch_size, n_components)

@jit(nopython=True)
def mean_axis_0(data):
    # Manually calculate the mean along axis 0
    return np.sum(data, axis=0) / data.shape[0]

@jit(nopython=True)
def pca_cpu(batch_size, n_components):
    data = np.random.rand(batch_size, batch_size)
    data_centered = data - mean_axis_0(data)  # Use manual mean along axis 0
    cov_matrix = np.cov(data_centered.T)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
    sorted_idx = np.argsort(eigen_values)[::-1]
    eigen_vectors = eigen_vectors[:, sorted_idx]
    return np.dot(data_centered, eigen_vectors[:, :n_components])


# SVD Test (CPU using Numba, GPU using CuPy)
def svd_test(on_gpu, batch_size=100):
    if on_gpu:
        with cp.cuda.Device(0):
            data = cp.random.rand(batch_size, batch_size)
            return cp.linalg.svd(data, full_matrices=False)
    else:
        # CPU run using Numba
        return svd_cpu(batch_size)

@jit(nopython=True)
def svd_cpu(batch_size):
    data = np.random.rand(batch_size, batch_size)
    return np.linalg.svd(data, full_matrices=False)

# Convolution Test (CPU using Numba, GPU using CuPy) with 1-second sound filter
def convolution_test(on_gpu, filter_size=256, sample_rate=44100):
    # Generate a sound filter (1-second sine wave)
    t = np.linspace(0, 1, sample_rate)  # 1 second duration at 44.1kHz sampling rate
    frequency = 440  # A typical tone at 440 Hz (A4 note)
    sound_filter = np.sin(2 * np.pi * frequency * t)  # Generate sine wave

    # Reshape the sound filter to 3x3 to match the sliding window
    filter_2d = np.reshape(sound_filter[:9], (3, 3))

    if on_gpu:
        # GPU-based convolution using CuPy
        with cp.cuda.Device(0):
            data = cp.random.rand(filter_size, filter_size)
            result = cp.empty_like(data)
            for i in range(1, data.shape[0] - 1):
                for j in range(1, data.shape[1] - 1):
                    result[i, j] = cp.sum(data[i - 1:i + 2, j - 1:j + 2] * cp.asarray(filter_2d))
            return result
    else:
        # CPU run using NumPy and Numba
        return convolution_cpu(filter_size, filter_2d)

@jit(nopython=True)
def convolution_cpu(filter_size, filter_2d):
    data = np.random.rand(filter_size, filter_size)
    result = np.empty_like(data)
    for i in range(1, data.shape[0] - 1):
        for j in range(1, data.shape[1] - 1):
            result[i, j] = np.sum(data[i - 1:i + 2, j - 1:j + 2] * filter_2d)
    return result

# FFT Test (CPU using NumPy, GPU using CuPy)
def fft_test(on_gpu, length=256, batch_size=256):
    if on_gpu:
        with cp.cuda.Device(0):
            data = cp.random.rand(length)
            return cp.fft.fft(data)
    else:
        # CPU-based FFT without Numba, because Numba doesn't support np.fft.fft
        return fft_cpu(length)

# Remove Numba here since it's not supported for FFT
def fft_cpu(length):
    data = np.random.rand(length)
    return np.fft.fft(data)

# MNIST test remains unchanged, no Numba here (as per your request)

# Function to run each test and collect results
def run_test(test_name, test_function, on_gpu=True, workers=1, batch_size=100, filter_size=256, length=256, epochs=10, learning_rate=0.01):
    monitor = ResourceMonitor()
    system_info = monitor.get_system_info()

    # On Linux we can clear the cache before running the test
    if platform.system() == "Linux":
        monitor.clear_cache()

    monitor.start_monitoring()
    start_time = time.time()

    # Run the test based on the test type
    if test_name in ["matrix_multiplication_test", "pca_test", "svd_test"]:
        result = test_function(on_gpu, batch_size)
    elif test_name == "convolution_test":
        result = test_function(on_gpu, filter_size)
    elif test_name == "fft_test":
        result = test_function(on_gpu, length, batch_size)
    elif test_name == "train_test_mnist":
        result = test_function(on_gpu, workers, epochs, batch_size, learning_rate) 
    else:
        result = test_function()
    
    end_time = time.time()
    monitor.stop_monitoring()

    average_usage = monitor.get_average_usage()
    elapsed_time = end_time - start_time

    # Collect system stats, including system info and epochs
    system_stats = {
        **system_info,  # CPU and GPU info
        "Run Type": "GPU" if on_gpu else "CPU",
        "Workers": workers,
        "Batch Size": batch_size,
        "Filter Size": filter_size,
        "Learning rate": learning_rate,
        "FFT Length": length,
        "Epochs": epochs,
        "Elapsed Time": elapsed_time,
        **average_usage
    }

    # Save results for this test
    monitor.save_results(system_stats, f"numba_{test_name}", "./EngFinalProject/results/results.json", workers=workers, batch_size=batch_size, epochs=epochs)

# Function to run all tests
def run_all_tests(batch_size):
    tests = {
        "iterative_test": iterative_test,
        "matrix_multiplication_test": matrix_multiplication_test,
        "pca_test": pca_test,
        "svd_test": svd_test,
        "convolution_test": convolution_test,
        "fft_test": fft_test,
    }

    # Run all tests on CPU
    for test_name, test_function in tests.items():
        print(f'Running {test_name}, on CPU')
        run_test(test_name, test_function, on_gpu=False, batch_size=batch_size)

    # Run all tests on GPU
    for test_name, test_function in tests.items():
        print(f'Running {test_name}, on GPU')
        run_test(test_name, test_function, on_gpu=True, batch_size=batch_size)

if __name__ == "__main__":
    batch_sizes = [256, 1024, 4096]

    for batch_size in batch_sizes:
        print(f"Running tests with batch size: {batch_size}")
        run_all_tests(batch_size=batch_size)
