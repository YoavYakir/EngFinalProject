import time
import dask
import dask.array as da
import cupy as cp
import cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask import delayed
from tensorflow.keras.datasets import mnist
from ...scripts.utility_functions import ResourceMonitor  # Adjust the path accordingly

# Initialize the Dask client with CUDA support (if available)
def initialize_dask_client(use_gpu=True):
    if use_gpu:
        cluster = LocalCUDACluster()
    else:
        cluster = None  # Default CPU-based Dask cluster
    client = Client(cluster)
    return client

# Iterative Test (using Dask for parallelization)
@delayed
def iterative_test():
    result = 0
    for i in range(1000000):
        result += cp.random.randint(0, 10)
    return result

# Matrix Multiplication Test (using Dask and Dask-CuPy)
@delayed
def matrix_multiplication_test(on_gpu, batch_size=100):
    if on_gpu:
        A = da.from_array(cp.random.rand(batch_size, batch_size), chunks=(batch_size // 10, batch_size // 10))
        B = da.from_array(cp.random.rand(batch_size, batch_size), chunks=(batch_size // 10, batch_size // 10))
        return da.dot(A, B).compute()
    else:
        A = da.from_array(cp.random.rand(batch_size, batch_size), chunks=(batch_size // 10, batch_size // 10))
        B = da.from_array(cp.random.rand(batch_size, batch_size), chunks=(batch_size // 10, batch_size // 10))
        return da.dot(A, B).compute()

# PCA Test (using Dask for parallel processing)
@delayed
def pca_test(on_gpu, batch_size=100, n_components=2):
    if on_gpu:
        data = da.from_array(cp.random.rand(batch_size, batch_size), chunks=(batch_size // 10, batch_size // 10))
        data_centered = data - data.mean(axis=0)
        cov_matrix = da.cov(data_centered.T)
        eigen_values, eigen_vectors = da.linalg.eigh(cov_matrix)
        sorted_idx = da.argsort(eigen_values)[::-1]
        eigen_vectors = eigen_vectors[:, sorted_idx]
        return da.dot(data_centered, eigen_vectors[:, :n_components]).compute()
    else:
        data = da.from_array(cp.random.rand(batch_size, batch_size), chunks=(batch_size // 10, batch_size // 10))
        data_centered = data - data.mean(axis=0)
        cov_matrix = da.cov(data_centered.T)
        eigen_values, eigen_vectors = da.linalg.eigh(cov_matrix)
        sorted_idx = da.argsort(eigen_values)[::-1]
        return da.dot(data_centered, eigen_vectors[:, :n_components]).compute()

# SVD Test (using Dask for parallel SVD computation)
@delayed
def svd_test(on_gpu, batch_size=100):
    if on_gpu:
        data = da.from_array(cp.random.rand(batch_size, batch_size), chunks=(batch_size // 10, batch_size // 10))
        return da.linalg.svd(data, full_matrices=False).compute()
    else:
        data = da.from_array(cp.random.rand(batch_size, batch_size), chunks=(batch_size // 10, batch_size // 10))
        return da.linalg.svd(data, full_matrices=False).compute()

# Convolution Test (using Dask for parallel convolution)
@delayed
def convolution_test(on_gpu, filter_size=256, sample_rate=44100):
    t = da.linspace(0, 1, sample_rate, chunks=sample_rate // 10)
    frequency = 440
    sound_filter = da.sin(2 * cp.pi * frequency * t)
    filter_2d = sound_filter[:9].reshape((3, 3))

    if on_gpu:
        data = da.from_array(cp.random.rand(filter_size, filter_size), chunks=(filter_size // 10, filter_size // 10))
        result = da.empty_like(data)
        for i in range(1, data.shape[0] - 1):
            for j in range(1, data.shape[1] - 1):
                result[i, j] = da.sum(data[i - 1:i + 2, j - 1:j + 2] * filter_2d).compute()
        return result
    else:
        data = da.from_array(cp.random.rand(filter_size, filter_size), chunks=(filter_size // 10, filter_size // 10))
        result = da.empty_like(data)
        for i in range(1, data.shape[0] - 1):
            for j in range(1, data.shape[1] - 1):
                result[i, j] = da.sum(data[i - 1:i + 2, j - 1:j + 2] * filter_2d).compute()
        return result

# FFT Test (using Dask for parallel FFT computation)
@delayed
def fft_test(on_gpu, length=256):
    if on_gpu:
        data = da.from_array(cp.random.rand(length), chunks=(length // 10))
        return da.fft.fft(data).compute()
    else:
        data = da.from_array(cp.random.rand(length), chunks=(length // 10))
        return da.fft.fft(data).compute()

# MNIST Test using Dask (CPU and GPU support)
@delayed
def mnist_test(on_gpu, workers=1, epochs=10, batch_size=256, learning_rate=0.01, model_size='small'):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if on_gpu:
        x_train = da.from_array(cp.array(x_train / 255.0), chunks=(batch_size, 28, 28))
        x_test = da.from_array(cp.array(x_test / 255.0), chunks=(batch_size, 28, 28))
    else:
        x_train = da.from_array(x_train / 255.0, chunks=(batch_size, 28, 28))
        x_test = da.from_array(x_test / 255.0, chunks=(batch_size, 28, 28))

    # Train a simple neural network (Dask-compatible)
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, workers=workers, use_multiprocessing=True)
    
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, accuracy

# Function to run each test and collect results
def run_test(test_name, test_function, on_gpu=True, **kwargs):
    monitor = ResourceMonitor()
    system_info = monitor.get_system_info()

    # Start Dask monitoring and testing
    monitor.start_monitoring()
    start_time = time.time()

    # Run the Dask-delayed test
    result = test_function(on_gpu, **kwargs).compute()

    end_time = time.time()
    monitor.stop_monitoring()

    elapsed_time = end_time - start_time
    average_usage = monitor.get_average_usage()

    # Collect and save the results with method name "dask"
    system_stats = {
        **system_info,
        "Run Type": "GPU" if on_gpu else "CPU",
        "Method": "dask",  # Add the method name as "dask"
        "Elapsed Time": elapsed_time,
        **average_usage
    }

    monitor.save_results(system_stats, f"dask_{test_name}", "./EngFinalProject/results/results.json")

# Function to run all tests
def run_all_tests(on_gpu=True):
    tests = {
        "iterative_test": iterative_test,
        "matrix_multiplication_test": matrix_multiplication_test,
        "pca_test": pca_test,
        "svd_test": svd_test,
        "convolution_test": convolution_test,
        "fft_test": fft_test,
        "mnist_test": mnist_test
    }

    for test_name, test_function in tests.items():
        print(f'Running {test_name}')
        run_test(test_name, test_function, on_gpu=on_gpu, batch_size=256)

if __name__ == "__main__":
    client = initialize_dask_client(use_gpu=True)  # Set `use_gpu=False` for CPU
    run_all_tests(on_gpu=True)
