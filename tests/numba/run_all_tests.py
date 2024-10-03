import time
import numpy as np
from numba import jit, cuda
import os
from scripts.utility_functions import ResourceMonitor

# Iterative Test using Numba (CPU and GPU versions)
@jit(nopython=True)
def iterative_test_cpu():
    result = 0
    for i in range(1000000):
        result += i
    return result

@cuda.jit
def iterative_test_gpu(result):
    i = cuda.grid(1)
    if i < result.size:
        result[i] += i

# Matrix Multiplication Test (CPU using Numba, GPU using cuda.jit)
@cuda.jit
def matrix_multiplication_gpu(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

@jit(nopython=True)
def matrix_multiplication_cpu(A, B):
    return np.dot(A, B)

# PCA Test (CPU and GPU)
@cuda.jit
def pca_gpu(data, eigen_vectors):
    i = cuda.grid(1)
    if i < data.shape[0]:
        for j in range(eigen_vectors.shape[1]):
            eigen_vectors[i, j] = data[i, j]  # Just an example, you should apply PCA logic here

@jit(nopython=True)
def pca_cpu(data):
    data_centered = data - np.mean(data, axis=0)
    cov_matrix = np.cov(data_centered.T)
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    sorted_idx = np.argsort(eigen_values)[::-1]  # extracting most popular attributes
    return np.dot(data_centered, eigen_vectors[:, :2])

# SVD Test (CPU and GPU)
@cuda.jit
def svd_gpu(data, U, S, Vt):
    i = cuda.grid(1)
    if i < data.shape[0]:
        U[i, i] = data[i, i]  # Simplified example, implement full SVD logic here

@jit(nopython=True)
def svd_cpu(data):
    return np.linalg.svd(data, full_matrices=False)

# Function to run each test and collect results
def run_test(test_name, test_function, on_gpu=False, workers=1, batch_size=100, learning_rate=0.01):
    monitor = ResourceMonitor()
    system_info = monitor.get_system_info()

    monitor.start_monitoring()
    start_time = time.time()

    # Run the test based on the test type
    if test_name == "iterative_test":
        if on_gpu:
            result = cuda.device_array(1000000, dtype=np.int32)
            threads_per_block = 1024
            blocks_per_grid = (result.size + threads_per_block - 1) // threads_per_block
            iterative_test_gpu[blocks_per_grid, threads_per_block](result)
        else:
            result = iterative_test_cpu()

    elif test_name == "matrix_multiplication_test":
        if on_gpu:
            A = np.random.rand(batch_size, batch_size).astype(np.float32)
            B = np.random.rand(batch_size, batch_size).astype(np.float32)
            C = np.zeros_like(A)
            threads_per_block = (16, 16)
            blocks_per_grid = (A.shape[0] // threads_per_block[0] + 1, B.shape[1] // threads_per_block[1] + 1)
            matrix_multiplication_gpu[blocks_per_grid, threads_per_block](cuda.to_device(A), cuda.to_device(B), cuda.to_device(C))
        else:
            A = np.random.rand(batch_size, batch_size).astype(np.float32)
            B = np.random.rand(batch_size, batch_size).astype(np.float32)
            matrix_multiplication_cpu(A, B)

    elif test_name == "pca_test":
        if on_gpu:
            data = cuda.to_device(np.random.rand(batch_size, batch_size).astype(np.float32))
            eigen_vectors = cuda.device_array((batch_size, 2), dtype=np.float32)
            threads_per_block = 128
            blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block
            pca_gpu[blocks_per_grid, threads_per_block](data, eigen_vectors)
        else:
            data = np.random.rand(batch_size, batch_size)
            pca_cpu(data)

    elif test_name == "svd_test":
        if on_gpu:
            data = cuda.to_device(np.random.rand(batch_size, batch_size).astype(np.float32))
            U = cuda.device_array_like(data)
            S = cuda.device_array((batch_size,), dtype=np.float32)
            Vt = cuda.device_array_like(data)
            threads_per_block = 128
            blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block
            svd_gpu[blocks_per_grid, threads_per_block](data, U, S, Vt)
        else:
            data = np.random.rand(batch_size, batch_size)
            svd_cpu(data)

    end_time = time.time()
    monitor.stop_monitoring()

    average_usage = monitor.get_average_usage()
    elapsed_time = end_time - start_time

    # Collect system stats, including system info
    system_stats = {
        **system_info,  # CPU and GPU info
        "Run Type": "GPU" if on_gpu else "CPU",  # Indicate CPU or GPU run
        "Workers": workers,  # Number of workers used for GPU tests
        "Batch Size": batch_size,  # Include batch size for analysis
        "Elapsed Time": elapsed_time,
        "Learning rate": learning_rate,  # Add learning rate to system stats
        **average_usage
    }

    # Save results for this test
    monitor.save_results(system_stats, f"numba_{test_name}", "results/results.json", workers=workers, batch_size=batch_size)

# Function to run all tests with Nsight profiling for GPU
def run_all_tests_with_nsight(workers_list, batch_size=100, learning_rates=[0.01]):
    for workers in workers_list:
        for test_name in ["iterative_test", "matrix_multiplication_test", "fibonacci_test", "pca_test", "svd_test", "gpu_stress_test"]:
            for lr in learning_rates:  # Add learning rates to the test loop
                nsight_command = f"nsys profile --output={test_name}_profile_{workers}_workers_lr_{lr} python3 -c 'run_test(\"{test_name}\", None, True, {workers}, {batch_size}, {lr})'"
                os.system(nsight_command)
                print(f"Results for {test_name} with {workers} workers and learning rate {lr} saved to results.json")

# Function to run all tests (CPU and GPU)
def run_all_tests(batch_size=100, learning_rates=[0.01]):
    tests = {
        "iterative_test": iterative_test_cpu,
        "matrix_multiplication_test": lambda: matrix_multiplication_cpu(np.random.rand(batch_size, batch_size), np.random.rand(batch_size, batch_size)),
        "pca_test": lambda: pca_cpu(np.random.rand(batch_size, batch_size)),
        "svd_test": lambda: svd_cpu(np.random.rand(batch_size, batch_size)),
        "gpu_stress_test": lambda: gpu_stress_test(np.random.rand(batch_size, batch_size), np.zeros((batch_size, batch_size))),
        "fibonacci_test": lambda: fibonacci_test_cpu(20),
        "quicksort_test": lambda: quicksort_test_cpu([5, 3, 8, 6, 7, 2, 1])
    }

    # Run all tests on CPU
    for test_name, test_function in tests.items():
        run_test(test_name, test_function, on_gpu=False, batch_size=batch_size)

    # Run all tests on GPU with Nsight profiling
    workers_list = [1, 2, 4, 8]
    run_all_tests_with_nsight(workers_list, batch_size=batch_size, learning_rates=learning_rates)

if __name__ == "__main__":
    # Example batch sizes and learning rates for testing
    batch_sizes = [100, 500, 1000]
    learning_rates = [0.001, 0.01, 0.1]

    for batch_size in batch_sizes:
        print(f"Running tests with batch size: {batch_size}")
        run_all_tests(batch_size=batch_size, learning_rates=learning_rates)
