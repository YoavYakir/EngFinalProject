import time
import cupy as cp
import numpy as np
from numba import jit, cuda
import platform
import os
from ...scripts.utility_functions import ResourceMonitor

# Iterative Test
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

# Matrix Multiplication Test
@jit(nopython=True)
def matrix_multiplication_cpu(A, B):
    return np.dot(A, B)

@cuda.jit
def matrix_multiplication_gpu(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        temp = 0
        for k in range(A.shape[1]):
            temp += A[i, k] * B[k, j]
        C[i, j] = temp

# PCA Test
@jit(nopython=True)
def pca_cpu(data):
    # Manually compute the mean along axis=0 (column-wise mean)
    n_samples, n_features = data.shape
    mean = np.zeros(n_features)
    
    # Calculate the mean for each feature (column)
    for j in range(n_features):
        sum_column = 0.0
        for i in range(n_samples):
            sum_column += data[i, j]
        mean[j] = sum_column / n_samples

    # Subtract the mean from the data (center the data)
    data_centered = np.empty_like(data)
    for i in range(n_samples):
        for j in range(n_features):
            data_centered[i, j] = data[i, j] - mean[j]

    # Compute the covariance matrix
    cov_matrix = np.cov(data_centered.T)

    # Compute eigenvalues and eigenvectors
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)

    # Sort the eigenvectors by descending eigenvalues
    sorted_idx = np.argsort(eigen_values)[::-1]

    # Return the data projected onto the first two principal components
    return np.dot(data_centered, eigen_vectors[:, sorted_idx[:2]])

# Numba Kernel to center data on the GPU for better performance
@cuda.jit
def center_data_kernel(data, mean):
    i = cuda.grid(1)  # Get the thread index
    if i < data.shape[0]:
        for j in range(data.shape[1]):
            data[i, j] -= mean[j]  # Subtract the mean from each element

# PCA on GPU using both Numba and CuPy
def pca_gpu(data, n_components=2):
    """
    Perform PCA on the GPU using both Numba for custom kernels and CuPy for matrix operations.
    :param data: Input data as a CuPy array (data points x features)
    :param n_components: Number of principal components to keep
    :return: Projected data onto the top principal components
    """

    # Ensure the data is a CuPy array (for GPU computation)
    data = cp.asarray(data)

    # Step 1: Compute the mean of each feature using CuPy
    mean = cp.mean(data, axis=0)

    # Step 2: Center the data using Numba's custom kernel
    threads_per_block = 128
    blocks_per_grid = (data.shape[0] + threads_per_block - 1) // threads_per_block
    center_data_kernel[blocks_per_grid, threads_per_block](data, mean)
    
    # Ensure the GPU is done with the kernel execution before proceeding
    cuda.synchronize()

    # Step 3: Compute the covariance matrix using CuPy
    n_samples = data.shape[0]
    cov_matrix = cp.dot(data.T, data) / (n_samples - 1)

    # Step 4: Compute the eigenvalues and eigenvectors using CuPy
    eigen_values, eigen_vectors = cp.linalg.eigh(cov_matrix)

    # Step 5: Sort the eigenvalues in descending order and select the top n_components
    sorted_idx = cp.argsort(eigen_values)[::-1]
    top_eigen_vectors = eigen_vectors[:, sorted_idx[:n_components]]

    # Step 6: Project the data onto the top eigenvectors
    projected_data = cp.dot(data, top_eigen_vectors)

    return projected_data

# SVD Test
@jit(nopython=True)
def svd_cpu(data):
    return np.linalg.svd(data, full_matrices=False)

# SVD on GPU using CuPy for matrix operations
def svd_gpu(data):
    """
    Perform SVD on the GPU using CuPy for linear algebra and Numba for preprocessing.
    :param data: Input data as a CuPy array (data points x features)
    :return: U, S, Vt matrices from SVD decomposition
    """

    # Ensure the data is a CuPy array (for GPU computation)
    data = cp.asarray(data)

    # (Optional) Step 1: Center the data using Numba if needed
    mean = cp.mean(data, axis=0)
    threads_per_block = 128
    blocks_per_grid = (data.shape[0] + threads_per_block - 1) // threads_per_block
    center_data_kernel[blocks_per_grid, threads_per_block](data, mean)
    
    # Synchronize after kernel execution
    cuda.synchronize()

    # Step 2: Compute the SVD using CuPy's GPU-accelerated function
    # U: Left singular vectors, S: Singular values, Vt: Right singular vectors
    U, S, Vt = cp.linalg.svd(data, full_matrices=False)

    return U, S, Vt


# Convolution Test
@jit(nopython=True)
def convolution_cpu(data, filter_2d):
    result = np.empty_like(data)
    for i in range(1, data.shape[0] - 1):
        for j in range(1, data.shape[1] - 1):
            result[i, j] = np.sum(data[i - 1:i + 2, j - 1:j + 2] * filter_2d)
    return result

@cuda.jit
def convolution_gpu(data, result, filter_2d):
    i, j = cuda.grid(2)
    if i >= 1 and j >= 1 and i < data.shape[0] - 1 and j < data.shape[1] - 1:
        result[i, j] = 0
        for m in range(3):
            for n in range(3):
                result[i, j] += data[i - 1 + m, j - 1 + n] * filter_2d[m, n]

# FFT Test
def fft_cpu(data):
    return np.fft.fft(data)

# Numba Kernel to center data (subtract the mean) on the GPU
@cuda.jit
def center_data_kernel_fft(data, mean):
    i = cuda.grid(1)  # 1D thread index for parallel execution
    if i < data.size:
        data[i] -= mean  # Subtract the mean from each element (centering)

# FFT on GPU using CuPy for the actual FFT operation, with centering using Numba
def fft_gpu(data):
    """
    Perform FFT on the GPU using CuPy for FFT and Numba for centering the data.
    :param data: Input data as a CuPy array
    :return: FFT result
    """

    # Ensure the input data is a CuPy array (to run on the GPU)
    data = cp.asarray(data)

    # Step 1: Compute the mean of the data using CuPy
    mean = cp.mean(data)

    # Step 2: Center the data using Numba's custom kernel
    threads_per_block = 128
    blocks_per_grid = (data.size + threads_per_block - 1) // threads_per_block
    center_data_kernel_fft[blocks_per_grid, threads_per_block](data, mean)

    # Synchronize to ensure the kernel execution completes before continuing
    cuda.synchronize()

    # Step 3: Perform FFT using CuPy's GPU-accelerated FFT function
    result = cp.fft.fft(data)

    return result

# Train Test over MList
@jit(nopython=True)
def train_test_cpu(data, targets, weights, learning_rate, batch_size):
    for i in range(batch_size):
        logits = np.dot(data[i], weights)
        softmax = np.exp(logits) / np.sum(np.exp(logits))
        error = softmax
        error[targets[i]] -= 1
        gradient = np.dot(data[i].reshape(-1, 1), error.reshape(1, -1))
        weights -= learning_rate * gradient

# GPU kernel for forward and backward pass (training)
@cuda.jit
def train_test_gpu(data, targets, weights, batch_size, learning_rate):
    i = cuda.grid(1)  # Get thread index
    if i < batch_size:
        input_size = data.shape[1]
        output_size = weights.shape[1]
        
        # Allocate local memory for logits and softmax
        logits = cuda.local.array(10, dtype=cuda.float32)  # Assuming 10 output classes
        softmax = cuda.local.array(10, dtype=cuda.float32)

        # Forward pass: Compute logits (linear transformation)
        for j in range(output_size):
            logits[j] = 0.0
            for k in range(input_size):
                logits[j] += data[i, k] * weights[k, j]
        
        # Compute softmax
        sum_exp = 0.0
        for j in range(output_size):
            softmax[j] = cuda.math.exp(logits[j])
            sum_exp += softmax[j]

        for j in range(output_size):
            softmax[j] /= sum_exp

        # Backward pass: Cross-entropy gradient
        for j in range(output_size):
            gradient = softmax[j]
            if j == targets[i]:
                gradient -= 1  # Cross-entropy gradient adjustment

            # Weight update (gradient descent)
            for k in range(input_size):
                weights[k, j] -= learning_rate * gradient * data[i, k]

import numpy as np
import time
import platform
from numba import cuda

# Extend run_test to handle all test cases and differentiate CPU/GPU runs
def run_test(test_name, test_function_cpu, test_function_gpu, on_gpu=True, workers=1, batch_size=100, filter_size=256, length=256, epochs=110, learning_rate=0.01):
    monitor = ResourceMonitor()
    system_info = monitor.get_system_info()

    if platform.system() == "Linux":
        monitor.clear_cache()

    monitor.start_monitoring()
    start_time = time.time()

    # Run the test based on the test type
    if test_name == "matrix_multiplication_test":
        # Matrix multiplication test
        A = np.random.rand(batch_size, batch_size).astype(np.float32)
        B = np.random.rand(batch_size, batch_size).astype(np.float32)
        if on_gpu:
            # GPU execution
            A_gpu = cuda.to_device(A)
            B_gpu = cuda.to_device(B)
            C_gpu = cuda.device_array_like(A_gpu)
            threads_per_block = (16, 16)
            blocks_per_grid = (A.shape[0] // threads_per_block[0] + 1, B.shape[1] // threads_per_block[1] + 1)
            test_function_gpu[blocks_per_grid, threads_per_block](A_gpu, B_gpu, C_gpu)
            C_result = C_gpu.copy_to_host()
        else:
            # CPU execution
            C_result = test_function_cpu(A, B)

    elif test_name == "pca_test":
        # PCA test
        data = np.random.rand(batch_size, batch_size).astype(np.float32)
        if on_gpu:
            # GPU execution
            data_gpu = cuda.to_device(data)
            eigen_vectors_gpu = cuda.device_array((batch_size, 2), dtype=np.float32)
            threads_per_block = 128
            blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block
            test_function_gpu[blocks_per_grid, threads_per_block](data_gpu, eigen_vectors_gpu)
            result = eigen_vectors_gpu.copy_to_host()
        else:
            # CPU execution
            result = test_function_cpu(data)

    elif test_name == "fft_test":
        # FFT test
        data = np.random.rand(length).astype(np.float32)
        if on_gpu:
            # GPU execution
            data_gpu = cuda.to_device(data)
            result_gpu = cuda.device_array_like(data_gpu)
            threads_per_block = 128
            blocks_per_grid = (length + threads_per_block - 1) // threads_per_block
            test_function_gpu[blocks_per_grid, threads_per_block](data_gpu, result_gpu)
            result = result_gpu.copy_to_host()
        else:
            # CPU execution
            result = test_function_cpu(data)

    elif test_name == "train_test_mlist":
        # Train Test over MList
        input_size = 100  # Assuming fixed input size
        output_size = 10  # Assuming fixed number of output classes
        data = np.random.rand(batch_size, input_size).astype(np.float32)
        targets = np.random.randint(0, output_size, size=batch_size).astype(np.int32)
        weights = np.random.rand(input_size, output_size).astype(np.float32)

        if on_gpu:
            # GPU execution
            data_gpu = cuda.to_device(data)
            targets_gpu = cuda.to_device(targets)
            weights_gpu = cuda.to_device(weights)
            threads_per_block = 128
            blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block
            test_function_gpu[blocks_per_grid, threads_per_block](data_gpu, targets_gpu, weights_gpu, batch_size, learning_rate)
            weights_result = weights_gpu.copy_to_host()
        else:
            # CPU execution
            test_function_cpu(data, targets, weights, learning_rate, batch_size)
            weights_result = weights

    elif test_name == "convolution_test":
        # Convolution test
        data = np.random.rand(filter_size, filter_size).astype(np.float32)
        filter_2d = np.random.rand(3, 3).astype(np.float32)  # 3x3 filter
        if on_gpu:
            # GPU execution
            data_gpu = cuda.to_device(data)
            result_gpu = cuda.device_array_like(data_gpu)
            threads_per_block = (16, 16)
            blocks_per_grid = (filter_size // threads_per_block[0] + 1, filter_size // threads_per_block[1] + 1)
            test_function_gpu[blocks_per_grid, threads_per_block](data_gpu, result_gpu, filter_2d)
            result = result_gpu.copy_to_host()
        else:
            # CPU execution
            result = test_function_cpu(data, filter_2d)

    elif test_name == "svd_test":
        # SVD test
        data = np.random.rand(batch_size, batch_size).astype(np.float32)
        if on_gpu:
            # GPU execution
            data_gpu = cuda.to_device(data)
            U_gpu = cuda.device_array_like(data_gpu)
            S_gpu = cuda.device_array((batch_size,), dtype=np.float32)
            Vt_gpu = cuda.device_array_like(data_gpu)
            threads_per_block = 128
            blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block
            test_function_gpu[blocks_per_grid, threads_per_block](data_gpu, U_gpu, S_gpu, Vt_gpu)
            U_result = U_gpu.copy_to_host()
            S_result = S_gpu.copy_to_host()
            Vt_result = Vt_gpu.copy_to_host()
        else:
            # CPU execution
            U_result, S_result, Vt_result = test_function_cpu(data)

    # Collect monitoring data and calculate the elapsed time
    end_time = time.time()
    monitor.stop_monitoring()

    elapsed_time = end_time - start_time
    average_usage = monitor.get_average_usage()

    # Prepare system statistics for saving
    system_stats = {
        **system_info,
        "Run Type": "GPU" if on_gpu else "CPU",
        "Workers": workers,
        "Batch Size": batch_size,
        "Filter Size": filter_size,
        "FFT Length": length,
        "Learning rate": learning_rate,
        "Elapsed Time": elapsed_time,
        **average_usage
    }

    # Save the results to a JSON file
    monitor.save_results(system_stats, f"numba_{test_name}", "./EngFinalProject/results/results.json")


# Function to run all tests
def run_all_tests():
    tests = {
        "iterative_test": (iterative_test_cpu, iterative_test_gpu),
        "matrix_multiplication_test": (matrix_multiplication_cpu, matrix_multiplication_gpu),
        "pca_test": (pca_cpu, pca_gpu),
        "svd_test": (svd_cpu, svd_gpu),
        "convolution_test": (convolution_cpu, convolution_gpu),
        "fft_test": (fft_cpu, fft_gpu),
        "train_test_mlist": (train_test_cpu, train_test_gpu),
    }

    # Run all tests on CPU
    for test_name, (test_function_cpu, test_function_gpu) in tests.items():
        if test_name == "train_test_mlist":
            # For train_test_mlist, run with different batch sizes and workers
            for batch_size in [256, 512, 1024]:  # Define different batch sizes
                for workers in [1, 2, 4, 8]:  # Define different worker counts
                    print(f'Running {test_name} on CPU with batch size {batch_size} and {workers} workers...')
                    run_test(test_name, test_function_cpu, test_function_gpu, on_gpu=False, workers=workers, batch_size=batch_size)
        else:
            print(f'Running {test_name} on CPU...')
            run_test(test_name, test_function_cpu, test_function_gpu, on_gpu=False, batch_size=1024)

    # Run all tests on GPU
    for test_name, (test_function_cpu, test_function_gpu) in tests.items():
        if test_name == "train_test_mlist":
            # For train_test_mlist, run with different batch sizes and workers
            for batch_size in [256, 512, 1024]:  # Define different batch sizes
                for workers in [1, 2, 4, 8]:  # Define different worker counts
                    print(f'Running {test_name} on GPU with batch size {batch_size} and {workers} workers...')
                    run_test(test_name, test_function_cpu, test_function_gpu, on_gpu=True, workers=workers, batch_size=batch_size)
        else:
            print(f'Running {test_name} on GPU...')
            run_test(test_name, test_function_cpu, test_function_gpu, on_gpu=True, batch_size=1024)


if __name__ == "__main__":
    # Call run_all_tests to run all tests with varying batch sizes for train_test_mlist
    run_all_tests()