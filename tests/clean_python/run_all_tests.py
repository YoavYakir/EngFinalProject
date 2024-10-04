import time
import numpy as np
import cupy as cp
import os
import sys
import platform
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
        eigen_values, eigen_vectors = cp.linalg.eigh(cov_matrix)
        sorted_idx = cp.argsort(eigen_values)[::-1]
        eigen_vectors = eigen_vectors[:, sorted_idx]
        return cp.dot(data_centered, eigen_vectors[:, :n_components])
    else:
        data = np.random.rand(batch_size, batch_size)
        data_centered = data - np.mean(data, axis=0)
        cov_matrix = np.cov(data_centered.T)
        eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
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


# Convolution Test (CPU using NumPy, GPU using CuPy) with 1-second sound filter
def convolution_test(on_gpu, filter_size=256, sample_rate=44100):
    # Generate a sound filter (1-second sine wave)
    t = np.linspace(0, 1, sample_rate)  # 1 second duration at 44.1kHz sampling rate
    frequency = 440  # A typical tone at 440 Hz (A4 note)
    sound_filter = np.sin(2 * np.pi * frequency * t)  # Generate sine wave

    # Reshape the sound filter to 3x3 to match the sliding window
    filter_2d = np.reshape(sound_filter[:9], (3, 3))  # Reshape to 3x3

    if on_gpu:
        # GPU-based convolution using CuPy
        data = cp.random.rand(filter_size, filter_size)
        result = cp.empty_like(data)
        for i in range(1, data.shape[0] - 1):
            for j in range(1, data.shape[1] - 1):
                result[i, j] = cp.sum(data[i - 1:i + 2, j - 1:j + 2] * cp.asarray(filter_2d))
        return result
    else:
        # CPU-based convolution using NumPy
        data = np.random.rand(filter_size, filter_size)
        result = np.empty_like(data)
        for i in range(1, data.shape[0] - 1):
            for j in range(1, data.shape[1] - 1):
                result[i, j] = np.sum(data[i - 1:i + 2, j - 1:j + 2] * filter_2d)
        return result
        
# FFT Test (CPU using NumPy, GPU using CuPy)
def fft_test(on_gpu, length=256):
    if on_gpu:
        data = cp.random.rand(length)
        return cp.fft.fft(data)
    else:
        data = np.random.rand(length)
        return np.fft.fft(data)

# Train Test over MList (simulating training with workers)
def train_test_mlist(on_gpu, workers=1, epochs=10, batch_size=100, learning_rate=0.01):
    data_size = 10000  # Simulated dataset size
    input_size = 100   # Input feature size
    output_size = 10   # Output feature size (for a 10-class problem)
    iterations = data_size // batch_size

    if on_gpu:
        data = cp.random.rand(data_size, input_size)  # Simulated input data on GPU
        targets = cp.random.randint(0, output_size, size=(data_size,))  # Random integer targets (0 to 9)
        
        # Initialize random weights
        weights = cp.random.rand(input_size, output_size)
        
        for epoch in range(epochs):
            for i in range(iterations // workers):
                batch = data[i * batch_size:(i + 1) * batch_size]
                target_batch = targets[i * batch_size:(i + 1) * batch_size]
                
                # Simulate forward pass (linear layer + softmax)
                logits = cp.dot(batch, weights)  # Linear transformation
                softmax = cp.exp(logits) / cp.sum(cp.exp(logits), axis=1, keepdims=True)  # Softmax activation

                # Simulate backward pass (Gradient descent update on weights)
                error = softmax
                error[cp.arange(batch_size), target_batch] -= 1  # Cross-entropy gradient
                gradient = cp.dot(batch.T, error) / batch_size  # Weight gradient
                weights -= learning_rate * gradient  # Update weights
                
                cp.cuda.Device(0).synchronize()  # Ensure GPU completes work before moving on to the next batch
    else:
        data = np.random.rand(data_size, input_size)  # Simulated input data on CPU
        targets = np.random.randint(0, output_size, size=(data_size,))  # Random integer targets
        
        # Initialize random weights
        weights = np.random.rand(input_size, output_size)
        
        for epoch in range(epochs):
            for i in range(iterations // workers):
                batch = data[i * batch_size:(i + 1) * batch_size]
                target_batch = targets[i * batch_size:(i + 1) * batch_size]
                
                # Simulate forward pass (linear layer + softmax)
                logits = np.dot(batch, weights)  # Linear transformation
                softmax = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)  # Softmax activation

                # Simulate backward pass (Gradient descent update on weights)
                error = softmax
                error[np.arange(batch_size), target_batch] -= 1  # Cross-entropy gradient
                gradient = np.dot(batch.T, error) / batch_size  # Weight gradient
                weights -= learning_rate * gradient  # Update weights

    return weights  # Returning the weights after training as an output of the test


# Function to run each test and collect results
def run_test(test_name, test_function, on_gpu=True, workers=1, batch_size=100, filter_size=256, length=256, epochs=110, learning_rate=0.01):
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
        result = test_function(on_gpu, length)
    elif test_name == "train_test_mlist":
        result = test_function(on_gpu, workers, epochs, batch_size, learning_rate) 
    else:
        result = test_function()

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
        "Filter Size": filter_size,
        "Learning rate": learning_rate,
        "FFT Length": length,
        "Elapsed Time": elapsed_time,
        **average_usage
    }

    # Save results for this test
    monitor.save_results(system_stats, f"clean_python_{test_name}", "./EngFinalProject/results/results.json", workers=workers, batch_size=batch_size)

# Function to run all tests
def run_all_tests(batch_size=100):
    tests = {
        "iterative_test": iterative_test,
        "matrix_multiplication_test": matrix_multiplication_test,
        "pca_test": pca_test,
        "svd_test": svd_test,
        "convolution_test": convolution_test,
        "fft_test": fft_test,
        "train_test_mlist": train_test_mlist,
    }

    # Run all tests on CPU
    for test_name, test_function in tests.items():
        print(f'Running {test_name}, on function {test_function}')
        run_test(test_name, test_function, on_gpu=True, batch_size=batch_size)

    # Run tests on GPU with Nsight profiling
    workers_list = [1, 2, 4, 8]  # Different worker counts for GPU runs
    batch_sizes = [256, 512, 1024]  # Different batch sizes for the MList train test
    learning_rates = [0.001, 0.01, 0.1]  # Different learning rates for the MList train test

    for workers in workers_list:
        # Run the train_test_mlist with varying batch sizes and learning rates
        print(f'Running mlist train with {workers} workers')
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                run_test("train_test_mlist", train_test_mlist, on_gpu=True, workers=workers, batch_size=batch_size, learning_rate=learning_rate)

if __name__ == "__main__":
    # Example batch sizes for testing
    batch_sizes = [256, 1024, 4096]

    for batch_size in batch_sizes:
        print(f"Running tests with batch size: {batch_size}")
        run_all_tests(batch_size=batch_size)
