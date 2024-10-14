import time
import cupy as cp
import platform
from ...scripts.utility_functions import ResourceMonitor

# Matrix Multiplication Test (GPU using CuPy)
def matrix_multiplication_test(matrix_size=64, dtype=cp.float32):
    with cp.cuda.Device(0):
        if dtype in [cp.float32, cp.float64]:
            # Use random float generation for float32 and float64
            A = cp.random.rand(matrix_size, matrix_size, dtype=dtype)
            B = cp.random.rand(matrix_size, matrix_size, dtype=dtype)
        elif dtype in [cp.int32, cp.int16]:
            # Use random integer generation for int32 and int16
            A = cp.random.randint(0, 100, size=(matrix_size, matrix_size), dtype=dtype)
            B = cp.random.randint(0, 100, size=(matrix_size, matrix_size), dtype=dtype)
        else:
            raise TypeError(f"Unsupported data type: {dtype}")
        
        return cp.dot(A, B)

# PCA Test (GPU using CuPy)
def pca_test(matrix_size=64, dtype=cp.float32, n_components=2):
    with cp.cuda.Device(0):
        if dtype in [cp.float32, cp.float64]:
            # Use random float generation for float32 and float64
            data = cp.random.rand(matrix_size, matrix_size, dtype=dtype)
        elif dtype in [cp.int32, cp.int16]:
            # Use random integer generation for int32 and int16
            data = cp.random.randint(0, 100, size=(matrix_size, matrix_size), dtype=dtype)
        else:
            raise TypeError(f"Unsupported data type: {dtype}")
        
        # PCA computation remains the same for any data type
        data_centered = data - cp.mean(data, axis=0)
        cov_matrix = cp.cov(data_centered.T)
        eigen_values, eigen_vectors = cp.linalg.eigh(cov_matrix)
        sorted_idx = cp.argsort(eigen_values)[::-1]
        eigen_vectors = eigen_vectors[:, sorted_idx]
        return cp.dot(data_centered, eigen_vectors[:, :n_components])

# SVD Test (GPU using CuPy)
def svd_test(matrix_size=64, dtype=cp.float32):
    with cp.cuda.Device(0):
        if dtype in [cp.float32, cp.float64]:
            data = cp.random.rand(matrix_size, matrix_size, dtype=dtype)
        elif dtype in [cp.int32, cp.int16]:
            data = cp.random.randint(0, 100, size=(matrix_size, matrix_size), dtype=dtype)
        else:
            raise TypeError(f"Unsupported data type: {dtype}")
        
        return cp.linalg.svd(data, full_matrices=False)

# Convolution Test (GPU using CuPy)
def convolution_test(filter_size=256, sample_rate=44100, dtype=cp.float32):
    t = cp.linspace(0, 1, sample_rate, dtype=dtype)
    frequency = 440
    sound_filter = cp.sin(2 * cp.pi * frequency * t)
    filter_2d = cp.reshape(sound_filter[:9], (3, 3))

    with cp.cuda.Device(0):
        if dtype in [cp.float32, cp.float64]:
            data = cp.random.rand(filter_size, filter_size, dtype=dtype)
        elif dtype in [cp.int32, cp.int16]:
            data = cp.random.randint(0, 100, size=(filter_size, filter_size), dtype=dtype)
        else:
            raise TypeError(f"Unsupported data type: {dtype}")
        
        result = cp.empty_like(data)
        for i in range(1, data.shape[0] - 1):
            for j in range(1, data.shape[1] - 1):
                result[i, j] = cp.sum(data[i - 1:i + 2, j - 1:j + 2] * cp.asarray(filter_2d))
        return result

# FFT Test (GPU using CuPy)
def fft_test(length=256, dtype=cp.float32):
    with cp.cuda.Device(0):
        if dtype in [cp.float32, cp.float64]:
            data = cp.random.rand(length, dtype=dtype)
        elif dtype in [cp.int32, cp.int16]:
            data = cp.random.randint(0, 100, size=(length,), dtype=dtype)
        else:
            raise TypeError(f"Unsupported data type: {dtype}")
        
        return cp.fft.fft(data)

# MNIST Test (GPU using CuPy)
def train_test_mnist(workers=1, epochs=10, batch_size=100, learning_rate=0.01, model_size='small'):
    data_size = 10000
    input_size = 100
    output_size = 10
    iterations = data_size // batch_size

    with cp.cuda.Device(0):
        data = cp.random.rand(data_size, input_size, dtype=cp.float32)
        targets = cp.random.randint(0, output_size, size=(data_size,))
        weights = cp.random.rand(input_size, output_size, dtype=cp.float32)

        for epoch in range(epochs):
            for i in range(iterations // workers):
                batch = data[i * batch_size:(i + 1) * batch_size]
                target_batch = targets[i * batch_size:(i + 1) * batch_size]
                logits = cp.dot(batch, weights)
                softmax = cp.exp(logits) / cp.sum(cp.exp(logits), axis=1, keepdims=True)
                error = softmax
                error[cp.arange(batch_size), target_batch] -= 1
                gradient = cp.dot(batch.T, error) / batch_size
                weights -= learning_rate * gradient
                cp.cuda.Device(0).synchronize()

    return weights

# Function to run each test
def run_test(test_name, test_function, matrix_size=64, filter_size=256, sample_rate=44100, batch_size=256, epochs=10, learning_rate=0.01, dtype=cp.float32):
    monitor = ResourceMonitor()
    system_info = monitor.get_system_info()

    if platform.system() == "Linux":
        monitor.clear_cache()

    monitor.start_monitoring()
    start_time = time.time()

    # Run the test
    if test_name in ["matrix_multiplication_test", "pca_test", "svd_test"]:
        result = test_function(matrix_size=matrix_size, dtype=dtype)
    elif test_name == "convolution_test":
        result = test_function(filter_size=filter_size, sample_rate=sample_rate, dtype=dtype)
    elif test_name == "fft_test":
        result = test_function(batch_size=batch_size, dtype=dtype)
    elif test_name == "train_test_mnist":
        result = test_function(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    
    end_time = time.time()
    monitor.stop_monitoring()

    average_usage = monitor.get_average_usage()
    elapsed_time = end_time - start_time

    # Convert dtype to a string to make it JSON serializable
    dtype_str = dtype.__name__

    # Collect system stats, including system info and epochs
    system_stats = {
        **system_info,
        "Run Type": "GPU",
        "Matrix Size": matrix_size,
        "Filter Size": filter_size,
        "Sample Rate": sample_rate,
        "Batch Size": batch_size,
        "Learning rate": learning_rate,
        "FFT Length": batch_size,
        "Epochs": epochs,
        "Elapsed Time": elapsed_time,
        "Data Type": dtype_str,  # Save dtype as string, not class
        **average_usage
    }

    monitor.save_results(system_stats, f"cupy_{test_name}", "./EngFinalProject/results/results.json")

# Function to run all tests with different sizes, data types, and parameters
def run_all_tests():
    tests = {
        "matrix_multiplication_test": matrix_multiplication_test,
        "pca_test": pca_test,
        "svd_test": svd_test,
        "convolution_test": convolution_test,
        "fft_test": fft_test,
        "train_test_mnist": train_test_mnist,
    }

    # Different matrix sizes, filter sizes, FFT lengths, and data types
    matrix_sizes = [64, 512, 1024]
    filter_sizes = [256, 1024, 4096]
    fft_lengths = [256, 1024, 4096]
    sample_rates = [44100, 16000]
    batch_sizes = [256, 512, 1024]
    data_types = [cp.float32, cp.float64, cp.int32, cp.int16]

    # Run matrix, PCA, and SVD tests for different matrix sizes and data types
    for matrix_size in matrix_sizes:
        for dtype in data_types:
            print(f"Running matrix tests with matrix size: {matrix_size}, data type: {dtype.__name__}")
            for test_name, test_function in tests.items():
                if test_name in ["matrix_multiplication_test", "pca_test", "svd_test"]:
                    run_test(test_name, test_function, matrix_size=matrix_size, dtype=dtype)

    # Run convolution tests for different filter sizes, sample rates, and data types
    for filter_size in filter_sizes:
        for sample_rate in sample_rates:
            for dtype in data_types:
                print(f"Running convolution test with filter size: {filter_size}, sample rate: {sample_rate}, data type: {dtype.__name__}")
                run_test("convolution_test", convolution_test, filter_size=filter_size, sample_rate=sample_rate, dtype=dtype)

    # Run FFT tests for different batch sizes and data types
    for fft_length in fft_lengths:
        for dtype in data_types:
            print(f"Running FFT test with length: {fft_length}, data type: {dtype.__name__}")
            run_test("fft_test", fft_test, batch_size=fft_length, dtype=dtype)

    # Run MNIST training test with different workers, batch sizes, learning rates, and epochs
    workers_list = [1, 2, 4, 8]
    learning_rates = [0.001, 0.01, 0.1]
    epochs_list = list(range(1, 11))

    for workers in workers_list:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                for epochs in epochs_list:
                    print(f"Running MNIST test with workers: {workers}, batch size: {batch_size}, learning rate: {learning_rate}, epochs: {epochs}")
                    run_test("train_test_mnist", train_test_mnist, batch_size=batch_size, learning_rate=learning_rate, epochs=epochs)

if __name__ == "__main__":
    run_all_tests()

