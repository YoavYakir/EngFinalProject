import datetime
import json
import platform
import subprocess
import time
import numpy as np
import random

from utilities.ResourceMonitor import ResourceMonitor    

# Iterative Test (CPU only)
def iterative_test(iterations=1000000, num=100):
    result = 0
    for i in range(iterations):
        result += random.randint(0,num)
    return result

# Matrix Multiplication Test
def matrix_multiplication_test(batch_size):
        A = np.random.rand(batch_size, batch_size)
        B = np.random.rand(batch_size, batch_size)
        return np.dot(A, B)
    
# PCA Test
def pca_test(batch_size, n_components=2):
        data = np.random.rand(batch_size, batch_size)
        data_centered = data - np.mean(data, axis=0)
        cov_matrix = np.cov(data_centered.T)
        eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
        sorted_idx = np.argsort(eigen_values)[::-1]
        return np.dot(data_centered, eigen_vectors[:, :n_components])

# SVD Test
def svd_test(batch_size):
    data = np.random.rand(batch_size, batch_size)
    return np.linalg.svd(data, full_matrices=False)


def convolution_test(filter_size, sample_rate):
    # Generate a sound filter (1-second sine wave)
    t = np.linspace(0, 1, sample_rate)  # 1 second duration
    frequency = 440  # A typical tone at 440 Hz (A4 note)
    sound_filter = np.sin(2 * np.pi * frequency * t)  # Generate sine wave

    # Reshape the sound filter to 3x3 to match the sliding window
    filter_2d = np.reshape(sound_filter[:9], (3, 3)) # Reshape to 3x3

    # CPU-based convolution using NumPy only (no explicit loops)
    data = np.random.rand(filter_size, filter_size)

    # Pad the input array to handle edges
    pad_width = filter_2d.shape[0] // 2  # Assumes filter is square
    padded_data = np.pad(data, pad_width, mode='constant', constant_values=0)

    # Prepare the output array
    result = np.empty_like(data)

    # Apply convolution using NumPy broadcasting
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Extract the region of interest and apply the filter
            region = padded_data[i:i + filter_2d.shape[0], j:j + filter_2d.shape[1]]
            result[i, j] = np.sum(region * filter_2d)
    
    return result

# FFT Test
def fft_test(batch_size):
    data = np.random.rand(batch_size)
    return np.fft.fft(data)

# Function to run each test and collect results
def run_test(test_name, test_function, workers=0, batch_size=0, filter_size=0, sample_rate=0, epochs=0, learning_rate=0, model_size=""):
    monitor = ResourceMonitor()
    system_info = monitor.get_system_info()

    # On Linux we can clear the cache before running the test
    # if platform.system() == "Linux":
    #     monitor.clear_cache()
    

    monitor.start_monitoring()
    start_time = time.time()

    # Run the test based on the test type
    if test_name in ["matrix_multiplication_test", "pca_test", "svd_test", "fft_test"]:
        test_function(batch_size)
    elif test_name == "convolution_test":
        test_function(filter_size, sample_rate)
    else:
        test_function()

    end_time = time.time()
    monitor.stop_monitoring()

    average_usage = monitor.get_average_usage()
    elapsed_time = end_time - start_time

    # Collect system stats, including system info and epochs
    system_stats = {
        **system_info,  # CPU and GPU info
        "Run Type": "CPU",  # Indicate CPU or GPU run
        "Batch Size": batch_size,  # Batch size for tests
        "Filter Size": filter_size,
        "Sample Rate": sample_rate,
        "Epochs": epochs,
        "Learning rate": learning_rate,
        "Model Size": model_size,
        "Data Type": "Default",
        "Elapsed Time": elapsed_time,
        "Result": "Not relevant",
        **average_usage
    }
    
    # Save results for this test
    monitor.save_results(system_stats, f"{test_name}", f"./results/{monitor.get_gpu_name()}/cpu/clean.json")


# Function to run all tests
def run_all_tests():
    print(f'Running iterative addition test')
    run_test("iterative_test", iterative_test)

    matrix_sizes = [64, 512, 1024, 2048, 4096]
    for matrix in matrix_sizes:
       print(f'Running Matrix Multiplication test, on CPU, matrix size (saved as batch size) : {matrix}')
       run_test("matrix_multiplication_test", matrix_multiplication_test, batch_size=matrix)
       print(f'Running PCA test, on CPU, matrix size (saved as batch size) : {matrix}')
       run_test("pca_test", pca_test, batch_size=matrix)
       print(f'Running SVD test, on CPU, matrix size (saved as batch size) : {matrix}')
       run_test("svd_test", svd_test, batch_size=matrix)

    filter_sizes = [256, 1024, 2048, 4096]
    sample_rates = [16000, 44100]
    for filter in filter_sizes:
       for sample in sample_rates:
           print(f'Running Convolution test, on CPU, Filter size : {filter}, Sample rate : {sample}')
           run_test("convolution_test", convolution_test, filter_size=filter, sample_rate=sample)

    batch_sizes = [256, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    for batch in batch_sizes:
        print(f'Running FFT test, on CPU, batch size : {batch}')
        run_test("fft_test", fft_test, batch_size=batch)

    # ResourceMonitor.fix_json_file(f"./results/cpu/clean_{date_string}.json")

if __name__ == "__main__":
    run_all_tests()