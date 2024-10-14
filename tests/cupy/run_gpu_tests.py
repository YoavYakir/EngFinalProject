import datetime
import json
import logging
import platform
import time
import subprocess
import numpy as np
from utilities import nsight_manager
from utilities import gpu_handling
from utilities.ResourceMonitor import ResourceMonitor
import cupy as cp

import os
import subprocess


def run_test_with_nsys(test_name, batch_size, filter_size, sample_rate, data_type):
    """
    Run the test under Nsight profiling and save results to a single JSON file.
    """
    monitor = ResourceMonitor()
    system_info = monitor.get_system_info()
    
    # Profiling setup
    output_dir = "./nsight_outputs2"
    os.makedirs(output_dir, exist_ok=True)
    nsys_file = os.path.join(output_dir, f"{test_name}_{batch_size}_{filter_size}_{sample_rate}_{data_type}.nsys-rep")

    data_type_str = "None" if data_type is None else f"'{data_type}'"

    cmd = [
    "nsys", "profile",
    "--output", nsys_file.replace(".nsys-rep", ""),
    "--trace=cuda,cublas",
    "--force-overwrite=true",
    "--show-output=true",
    "python3", "-u", "-c",
    f"\"from tests.clean_python.run_gpu_tests import *; print(run_single_test('{test_name}', {batch_size}, {filter_size}, {sample_rate}, {data_type_str}))\""
    ]

    print(f"Running Nsight profiling: {' '.join(cmd)}")
    process = subprocess.run(" ".join(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, timeout=1200)

    if process.returncode != 0:
        print("@"*20 + "FAILURE_START" + "@"*20)
        print(process.stderr.decode('utf-8'))
        print("@"*20 + "FAILURE_END" + "@"*20)
        print(f"Nsight Systems profiling failed:\n{process.stderr}")
        return

    # Extract JSON result from the child process output
    test_output = process.stdout.decode('utf-8').strip().split("\n")
    test_result = None
    for line in test_output:
        try:
            # Attempt to parse each line as JSON
            test_result = json.loads(line)
            break
        except json.JSONDecodeError:
            continue  # Ignore lines that are not JSON

    if not test_result:
        print(f"Failed to parse test output: {process.stdout}")
        raise ValueError("No valid JSON result found in child process output")

    # Parse profiling output
    profiling_data = nsight_manager.parse_nsys_output(nsys_file)

    # Collect and merge results
    system_stats = {
        **system_info,
        "Run Type": "GPU",
        "Batch Size": batch_size,
        "Filter Size": filter_size,
        "Sample Rate": sample_rate,
        "Workers": 0,
        "Epochs": 0,
        "Learning rate": 0,
        "Model Size": 0,
        "Data Type": data_type or "Default",
        "Elapsed Time": test_result["elapsed_time"],
        "Test Name": test_name,
        "Loss": "Not Relevant",
        "Accuracy": "Not Relevant",
        **test_result["avg_usage"],
        **profiling_data,
    }

    # Save results to a single JSON file
    results_file = f"./results/gpu/clean.json"
    monitor.save_results(system_stats, test_name, results_file)


# Matrix Multiplication Test
def matrix_multiplication_test(gpu_index, batch_size, dtype=None):
    with np.cuda.Device(gpu_index):
        if dtype in [np.float32, np.float64]:
            # Use random float generation for float32 and float64
            A = np.random.rand(batch_size, batch_size, dtype=dtype)
            B = np.random.rand(batch_size, batch_size, dtype=dtype)
            return np.dot(A, B)
        elif dtype in [np.int32, np.int16]:
            # Use random integer generation for int32 and int16
            A = np.random.randint(0, 100, size=(batch_size, batch_size), dtype=dtype)
            B = np.random.randint(0, 100, size=(batch_size, batch_size), dtype=dtype)
            return np.dot(A, B)
        elif dtype == None:
            A = np.random.rand(batch_size, batch_size)
            B = np.random.rand(batch_size, batch_size)
            return np.dot(A, B)
        else:
            raise TypeError(f"Unsupported data type: {dtype}")

# PCA Test
def pca_test(gpu_index, batch_size, n_components=2, dtype=None):
    with np.cuda.Device(gpu_index):
        if dtype in [np.float32, np.float64]:
            # Use random float generation for float32 and float64
            data = np.random.rand(batch_size, batch_size, dtype=dtype)
            data_centered = data - np.mean(data, axis=0)
            cov_matrix = np.cov(data_centered.T)
            eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
            sorted_idx = np.argsort(eigen_values)[::-1]
            eigen_vectors = eigen_vectors[:, sorted_idx]
            return np.dot(data_centered, eigen_vectors[:, :n_components])

        elif dtype in [np.int32, np.int16]:
            # Use random integer generation for int32 and int16
            data = np.random.randint(0, 100, size=(batch_size, batch_size), dtype=dtype)
            data_centered = data - np.mean(data, axis=0)
            cov_matrix = np.cov(data_centered.T)
            eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
            sorted_idx = np.argsort(eigen_values)[::-1]
            eigen_vectors = eigen_vectors[:, sorted_idx]
            return np.dot(data_centered, eigen_vectors[:, :n_components])
        elif dtype == None:
            data = np.random.rand(batch_size, batch_size)
            data_centered = data - np.mean(data, axis=0)
            cov_matrix = np.cov(data_centered.T)
            eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
            sorted_idx = np.argsort(eigen_values)[::-1]
            eigen_vectors = eigen_vectors[:, sorted_idx]
            return np.dot(data_centered, eigen_vectors[:, :n_components])
        else:
            raise TypeError(f"Unsupported data type: {dtype}")

# SVD Test
def svd_test(gpu_index, batch_size, dtype=None):
    with np.cuda.Device(gpu_index):
        if dtype in [np.float32, np.float64]:
            data = np.random.rand(batch_size, batch_size, dtype=dtype)
            return np.linalg.svd(data, full_matrices=False)
        elif dtype in [np.int32, np.int16]:
            data = np.random.randint(0, 100, size=(batch_size, batch_size), dtype=dtype)
            return np.linalg.svd(data, full_matrices=False)
        elif dtype == None:
            data = np.random.rand(batch_size, batch_size)
            return np.linalg.svd(data, full_matrices=False)
        else:
            raise TypeError(f"Unsupported data type: {dtype}")

# FFT Test (CPU using NumPy, GPU using CuPy)
def fft_test(gpu_index, batch_size, dtype=None):
    with np.cuda.Device(gpu_index):
        if dtype in [np.float32, np.float64]:
            data = np.random.rand(batch_size, dtype=dtype)
            return np.fft.fft(data)
        elif dtype in [np.int32, np.int16]:
            data = np.random.randint(0, 100, size=(batch_size,), dtype=dtype)
            return np.fft.fft(data)
        elif dtype == None:
            data = np.random.rand(batch_size)
            return np.fft.fft(data)
        else:
            raise TypeError(f"Unsupported data type: {dtype}")

def convolution_test(gpu_index, filter_size, sample_rate, dtype=None):
    # Generate a sound filter (1-second sine wave)
    t = np.linspace(0, 1, sample_rate)  # 1 second duration
    frequency = 440  # A typical tone at 440 Hz (A4 note)
    sound_filter = np.sin(2 * np.pi * frequency * t)  # Generate sine wave

    # Reshape the sound filter to 3x3 to match the sliding window
    filter_2d = np.reshape(sound_filter[:9], (3, 3)).astype(dtype or np.float32)  # Reshape to 3x3

    with np.cuda.Device(gpu_index):
        # GPU-based convolution using CuPy
        data = np.random.rand(filter_size, filter_size).astype(dtype or np.float32)

        # Pad the input array to handle edges
        pad_width = filter_2d.shape[0] // 2  # Assumes filter is square
        padded_data = np.pad(data, pad_width, mode='constant', constant_values=0)

        # Prepare the output array
        result = np.empty_like(data)

        # Apply convolution using CuPy
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                # Extract the region of interest and apply the filter
                region = padded_data[i:i + filter_2d.shape[0], j:j + filter_2d.shape[1]]
                result[i, j] = np.sum(region * np.asarray(filter_2d))
        
        return result.get()  # Return the result as a NumPy array

# Function to run each test and collect results
def run_test(test_name, test_function, batch_size=0, filter_size=0, sample_rate=0, dtype=None):
    monitor = ResourceMonitor()
    system_info = monitor.get_system_info()

    # Run the test based on the test type
    if test_name in ["matrix_multiplication_test", "pca_test", "svd_test", "fft_test"]:
        tests_result, profiling_results = nsight_manager.run_test_with_nsys(test_name, "clean_python", "run_gpu_tests", batch_size=batch_size, dtype=dtype)
    elif test_name == "convolution_test":
        tests_result, profiling_results = nsight_manager.run_test_with_nsys(test_name, "clean_python", "run_gpu_tests", filter_size=filter_size, sample_rate=sample_rate, dtype=dtype)
    else:
        exit(f"Test {test_name} is not supported")
    

    data_dict = {np.int16 : "int16", np.int32 : "int32", np.float32 : "float32", np.float64 : "float64", np.double : "double", None:"Default"}
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
        "Elapsed Time": tests_result["elapsed_time"],
        "Result": "not relevant",
        **tests_result["avg_usage"],
        **profiling_results
    }

    # Save results for this test
    monitor.save_results(system_stats, f"{test_name}", f"./results/gpu/clean.json")

def run_single_test(test_name, batch_size, filter_size, sample_rate, dtype=None):
    """
    Run a single test and collect accuracy and loss.
    Ensure GPU setup is handled in the current process.
    """
    # GPU setup in the current process
    gpu_index = gpu_handling.setup_gpu()
    gpu_handling.clear_gpu_memory(gpu_index)
    
    data_type_dict = {"None":None, "np.int16":np.int16, "np.int32":np.int32, "np.float32":np.float32, "np.double":np.double}

    sub_monitor = ResourceMonitor()
    sub_monitor.start_monitoring()
    start_time = time.time()

    # Run the test based on the test type
    if test_name == "matrix_multiplication_test":
        matrix_multiplication_test(gpu_index, batch_size, dtype=data_type_dict[dtype])
    elif test_name == "pca_test":
        pca_test(gpu_index, batch_size, dtype=data_type_dict[dtype])
    elif test_name == "svd_test":
        svd_test(gpu_index, batch_size, dtype=data_type_dict[dtype])
    elif test_name == "fft_test":
        fft_test(gpu_index, batch_size, dtype=data_type_dict[dtype])
    elif test_name == "convolution_test":
        convolution_test(gpu_index, filter_size, sample_rate, dtype=data_type_dict[dtype])
    else:
        exit(f"Test {test_name} is not supported")

    end_time = time.time()
    sub_monitor.stop_monitoring()
    avg_usage = sub_monitor.get_average_usage()
    elapsed_time = end_time - start_time    

    print(json.dumps({"elapsed_time": elapsed_time, "avg_usage" : avg_usage}))


# Function to run all tests
def run_all_tests():
    ResourceMonitor.init_results_file(f"./results/gpu/clean.json")

    data_type_list = ["None", "np.int16", "np.int32", "np.float32", "np.double"]

    for data_type in data_type_list:
        matrix_sizes = [64, 512, 1024, 2048, 4096]
        for matrix in matrix_sizes:
            print(f'Running Matrix Multiplication test, on CPU, matrix size (saved as batch size) : {matrix}')
            run_test_with_nsys("matrix_multiplication_test", batch_size=matrix, filter_size=0, sample_rate=0, data_type=data_type)
            print(f'Running PCA test, on CPU, matrix size (saved as batch size) : {matrix}')
            run_test_with_nsys("pca_test", batch_size=matrix, filter_size=0, sample_rate=0, data_type=data_type)
            print(f'Running SVD test, on CPU, matrix size (saved as batch size) : {matrix}')
            run_test_with_nsys("svd_test", batch_size=matrix, filter_size=0, sample_rate=0, data_type=data_type)

        filter_sizes = [256, 1024, 2048, 4096]
        sample_rates = [16000, 44100]
        for filter in filter_sizes:
            for sample in sample_rates:
                print(f'Running Convolution test, on CPU, Filter size : {filter}, Sample rate : {sample}')
                run_test_with_nsys("convolution_test", batch_size=0, filter_size=filter, sample_rate=sample, data_type=data_type)

        batch_sizes = [256, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
        for batch in batch_sizes:
            print(f'Running FFT test, on CPU, batch size : {batch}')
            run_test_with_nsys("fft_test", batch_size=batch, filter_size=0, sample_rate=0, data_type=data_type)

        ResourceMonitor.fix_json_file(f"./results/gpu/clean.json")

if __name__ == "__main__":
    run_all_tests()
