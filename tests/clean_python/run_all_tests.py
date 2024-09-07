import time
import numpy as np
import cupy as cp
import os
from ...scripts.utility_functions import ResourceMonitor

# Iterative Test (CPU only)
def iterative_test():
    result = 0
    for i in range(1000000):
        result += i
    return result

# Matrix Multiplication Test (CPU using numpy, GPU using cupy)
def matrix_multiplication_test(on_gpu, workers=1, batch_size=100):
    if on_gpu:
        # GPU run using CuPy
        A = cp.random.rand(batch_size, batch_size)
        B = cp.random.rand(batch_size, batch_size)
        return cp.dot(A, B)
    else:
        # CPU run using NumPy
        A = np.random.rand(batch_size, batch_size)
        B = np.random.rand(batch_size, batch_size)
        return np.dot(A, B)

# PCA Test (CPU using NumPy, GPU using CuPy)
def pca_test(on_gpu, batch_size=100, n_components=2):
    if on_gpu:
        data = cp.random.rand(batch_size, batch_size)
        data_centered = data - cp.mean(data, axis=0)
        cov_matrix = cp.cov(data_centered.T)
        eigen_values, eigen_vectors = cp.linalg.eig(cov_matrix)
        sorted_idx = cp.argsort(eigen_values)[::-1]
        eigen_vectors = eigen_vectors[:, sorted_idx]
        return cp.dot(data_centered, eigen_vectors[:, :n_components])
    else:
        data = np.random.rand(batch_size, batch_size)
        data_centered = data - np.mean(data, axis=0)
        cov_matrix = np.cov(data_centered.T)
        eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
        sorted_idx = np.argsort(eigen_values)[::-1]
        return np.dot(data_centered, eigen_vectors[:, :n_components])

# SVD Test (CPU using NumPy, GPU using CuPy)
def svd_test(on_gpu, batch_size=100):
    if on_gpu:
        data = cp.random.rand(batch_size, batch_size)
        return cp.linalg.svd(data, full_matrices=False)
    else:
        data = np.random.rand(batch_size, batch_size)
        return np.linalg.svd(data, full_matrices=False)

# GPU Stress Test (Intensive matrix multiplication to stress GPU)
def gpu_stress_test(on_gpu, batch_size=100):
    if on_gpu:
        data = cp.random.rand(batch_size, batch_size)
        result = cp.dot(data, data.T)
        return result
    else:
        data = np.random.rand(batch_size, batch_size)
        result = np.dot(data, data.T)
        return result

# Function to run each test and collect results
def run_test(test_name, test_function, on_gpu=False, workers=1, batch_size=100):
    monitor = ResourceMonitor()
    system_info = monitor.get_system_info()

    monitor.start_monitoring()
    start_time = time.time()

    # Run the test
    result = test_function(on_gpu, batch_size) if test_name in ["matrix_multiplication_test", "pca_test", "svd_test", "gpu_stress_test"] else test_function()

    end_time = time.time()
    monitor.stop_monitoring()

    average_usage = monitor.get_average_usage()
    elapsed_time = end_time - start_time

    # Collect system stats, including system info
    system_stats = {
        **system_info,  # CPU and GPU info
        "Run Type": "GPU" if on_gpu else "CPU",  # Indicate CPU or GPU run
        "Workers": workers,  # Number of workers used for GPU tests
        "Batch Size": batch_size,  # Batch size for tests
        "Elapsed Time": elapsed_time,
        **average_usage
    }

    # Save results for this test
    monitor.save_results(system_stats, f"clean_python_{test_name}", "../../results/results.json", workers=workers, batch_size=batch_size)

# Function to run the matrix multiplication test with Nsight profiling
def run_matrix_test_with_nsight(workers_list, on_gpu=True, batch_size=100):
    for workers in workers_list:
        # Run the matrix multiplication with different numbers of workers
        nsight_command = f"nsys profile --output=matrix_test_profile_{workers}_workers python3 -c 'run_test(\"matrix_multiplication_test\", matrix_multiplication_test, {on_gpu}, {workers}, {batch_size})'"
        
        # Execute the Nsight profiling
        os.system(nsight_command)

        # After Nsight profiling, the results for each run will still be saved to results.json
        print(f"Results for {workers} workers with batch size {batch_size} saved to results.json")

# Function to run all tests
def run_all_tests(batch_size=100):
    tests = {
        "iterative_test": iterative_test,
        "matrix_multiplication_test": matrix_multiplication_test,
        "pca_test": pca_test,
        "svd_test": svd_test,
        "gpu_stress_test": gpu_stress_test,
        "fibonacci_test": lambda: fibonacci_test(20),  # Fibonacci with n=20
        "quicksort_test": lambda: quicksort_test([5, 3, 8, 6, 7, 2, 1])
    }

    # Run all tests on CPU
    for test_name, test_function in tests.items():
        run_test(test_name, test_function, on_gpu=False, batch_size=batch_size)

    # Run matrix multiplication test on GPU with Nsight profiling
    workers_list = [1, 2, 4, 8]  # Different worker counts for GPU runs
    run_matrix_test_with_nsight(workers_list, on_gpu=True, batch_size=batch_size)

if __name__ == "__main__":
    # Example batch sizes for testing
    batch_sizes = [100, 500, 1000]

    for batch_size in batch_sizes:
        print(f"Running tests with batch size: {batch_size}")
        run_all_tests(batch_size=batch_size)
