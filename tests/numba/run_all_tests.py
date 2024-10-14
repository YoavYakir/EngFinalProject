import time
import numpy as np
import numba as nb
import os
import sys
import platform
import random
from ...scripts.utility_functions import ResourceMonitor
#import tensorflow as tf
#from tensorflow.keras.datasets import mnist
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Flatten
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.utils import to_categorical, Sequence
from numba import njit, cuda
#import cupy as cp
from scipy.fftpack import fft as scipy_fft


# Custom data generator inheriting from keras.utils.Sequence
#class NumpyDataGenerator(Sequence):
#    def __init__(self, x, y, batch_size, dtype=None):
#        self.x = x
#        self.y = y
#        self.batch_size = batch_size
#        self.dtype = dtype
#        self.num_samples = x.shape[0]#

#    def __len__(self):
#        return int(np.ceil(self.num_samples / self.batch_size))

#    def __getitem__(self, index):
#        # Get batch
#        x_batch = self.x[index * self.batch_size:(index + 1) * self.batch_size]
#        y_batch = self.y[index * self.batch_size:(index + 1) * self.batch_size]

#        # Apply dtype conversion if specified
#        if self.dtype is not None:
#            x_batch = x_batch.astype(self.dtype)
#            y_batch = y_batch.astype(self.dtype)

 #       return x_batch, y_batch

#    def on_epoch_end(self):
#        pass


# Iterative Test (CPU only) - Optimized with Numba
@nb.jit(nopython=True)
def iterative_test(iterations=1000000, num=100):
    result = 0
    for i in range(iterations):
        result += random.randint(0, num)
    return result


# Matrix Multiplication Test
@nb.jit(nopython=True)
def matrix_multiplication_test_cpu(batch_size, dtype=np.float32):
    A = np.random.rand(batch_size, batch_size)
    B = np.random.rand(batch_size, batch_size)
    return np.dot(A, B)

@cuda.jit
def matrix_multiplication_test_gpu(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        temp = 0
        for k in range(A.shape[1]):
            temp += A[row, k] * B[k, col]
        C[row, col] = temp

def matrix_multiplication_test(batch_size, dtype=None, on_gpu=False):
    if on_gpu:
        A = cp.random.rand(batch_size, batch_size).astype(dtype or cp.float32)
        B = cp.random.rand(batch_size, batch_size).astype(dtype or cp.float32)
        C = cp.zeros((batch_size, batch_size), dtype=dtype or cp.float32)
        
        threads_per_block = (16, 16)
        blocks_per_grid = ((batch_size + 15) // 16, (batch_size + 15) // 16)
        matrix_multiplication_test_gpu[blocks_per_grid, threads_per_block](A, B, C)
        
        return C
    else:
        return matrix_multiplication_test_cpu(batch_size, dtype)

@cuda.jit
def compute_cov_matrix(data, mean, cov_matrix):
    row, col = cuda.grid(2)
    if row < data.shape[1] and col < data.shape[1]:
        cov = 0
        for i in range(data.shape[0]):
            cov += (data[i, row] - mean[row]) * (data[i, col] - mean[col])
        cov_matrix[row, col] = cov / (data.shape[0] - 1)

def pca_test(batch_size, n_components=2, dtype=None, on_gpu=False):
    if on_gpu:
        # Generate data on GPU
        data = cp.random.rand(batch_size, batch_size).astype(dtype or cp.float32)
        mean_values = cp.mean(data, axis=0)
        
        # Define the covariance matrix on GPU
        cov_matrix = cp.zeros((batch_size, batch_size), dtype=dtype or cp.float32)
        
        # Define grid and block sizes for the kernel
        threads_per_block = (16, 16)
        blocks_per_grid = ((data.shape[1] + 15) // 16, (data.shape[1] + 15) // 16)
        
        # Run the custom CUDA kernel
        compute_cov_matrix[blocks_per_grid, threads_per_block](data, mean_values, cov_matrix)
        
        # Perform eigen decomposition on GPU
        eigen_values, eigen_vectors = cp.linalg.eigh(cov_matrix)
        idx = cp.argsort(eigen_values)[::-1]
        eigen_vectors = eigen_vectors[:, idx[:n_components]]
        
        # Return the transformed data
        return data @ eigen_vectors
    else:
        # CPU version of PCA here
        data = np.random.rand(batch_size, batch_size).astype(dtype or np.float32)
        mean_values = np.mean(data, axis=0)
        data_centered = data - mean_values
        cov_matrix = np.cov(data_centered, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigen_values)[::-1]
        eigen_vectors = eigen_vectors[:, idx[:n_components]]
        return data_centered @ eigen_vectors

# SVD Test - Optimized with Numba
@nb.jit(nopython=True)
def svd_test(batch_size, dtype=None, on_gpu=False):
    if dtype is None:
        data = np.random.rand(batch_size, batch_size)
    else:
        data = np.random.rand(batch_size, batch_size).astype(dtype)
    return np.linalg.svd(data, full_matrices=False)


# Convolution Test (1-second sound filter) - Optimized with Numba
@cuda.jit
def convolution_gpu(data, filter_2d, result):
    # Calculate the row and column index of the element
    row, col = cuda.grid(2)

    # Apply the convolution only within bounds
    if 1 <= row < data.shape[0] - 1 and 1 <= col < data.shape[1] - 1:
        temp = 0
        for i in range(3):
            for j in range(3):
                temp += data[row + i - 1, col + j - 1] * filter_2d[i, j]
        result[row, col] = temp

def convolution_test(filter_size, sample_rate, dtype=np.float32, on_gpu=False):
    # Define a 3x3 filter for convolution
    t = np.linspace(0, 1, sample_rate)
    frequency = 440
    sound_filter = np.sin(2 * np.pi * frequency * t)
    filter_2d = np.reshape(sound_filter[:9], (3, 3)).astype(dtype)

    if on_gpu:
        # Generate data and transfer to GPU
        data = np.random.rand(filter_size, filter_size).astype(dtype)
        data_gpu = cuda.to_device(data)
        filter_gpu = cuda.to_device(filter_2d)
        result_gpu = cuda.device_array((filter_size, filter_size), dtype=dtype)

        # Define grid and block sizes
        threads_per_block = (16, 16)
        blocks_per_grid = ((filter_size + 15) // 16, (filter_size + 15) // 16)

        # Launch convolution kernel on GPU
        convolution_gpu[blocks_per_grid, threads_per_block](data_gpu, filter_gpu, result_gpu)

        # Copy result back to host
        result = result_gpu.copy_to_host()
        return result
    else:
        # CPU implementation
        data = np.random.rand(filter_size, filter_size)
        result = np.empty_like(data)
        for i in range(1, data.shape[0] - 1):
            for j in range(1, data.shape[1] - 1):
                result[i, j] = np.sum(data[i - 1:i + 2, j - 1:j + 2] * filter_2d)
        return result


from scipy.fftpack import fft

@cuda.jit
def elementwise_multiply(frequencies, multiplier, output):
    idx = cuda.grid(1)
    if idx < frequencies.size:
        output[idx] = frequencies[idx] * multiplier[idx]

def fft_test(batch_size, dtype=np.float32, on_gpu=False):
    if on_gpu:
        # Generate random data and a multiplier on GPU
        data = cp.random.rand(batch_size).astype(dtype)
        multiplier = cp.random.rand(batch_size).astype(dtype)
        
        # Perform FFT with CuPy
        frequencies = cp.fft.fft(data)
        
        # Allocate memory for output
        output = cp.empty_like(frequencies)
        
        # Launch elementwise multiplication kernel
        threads_per_block = 256
        blocks_per_grid = (frequencies.size + threads_per_block - 1) // threads_per_block
        elementwise_multiply[blocks_per_grid, threads_per_block](frequencies, multiplier, output)
        
        return output
    else:
        # CPU implementation
        data = np.random.rand(batch_size)
        frequencies = scipy_fft(data)
        multiplier = np.random.rand(batch_size)
        output = frequencies * multiplier  # Element-wise multiplication on CPU
        return output

# Numba-enhanced function for data normalization
@njit
def normalize_data(x):
    return x / 255.0

# Ensure data consistency by converting input data to float32 or desired dtype
#class NumbaDataGenerator(Sequence):
#    def __init__(self, x_data, y_data, batch_size, dtype=np.float32):  # Default to float32
#        self.x_data = normalize_data(x_data).astype(dtype)  # Normalize and cast to consistent dtype
#        self.y_data = y_data.astype(dtype)
#        self.batch_size = batch_size
#        self.dtype = dtype
#        self.indices = np.arange(len(x_data))

#    def __len__(self):
#        return len(self.x_data) // self.batch_size

#    def __getitem__(self, idx):
#        start = idx * self.batch_size
#        x_batch, y_batch = extract_batch(self.x_data, self.y_data, start, self.batch_size, dtype=self.dtype)
#        return x_batch, y_batch

@njit
def extract_batch(x, y, start, batch_size, dtype=None):
    x_batch = x[start:start + batch_size]
    y_batch = y[start:start + batch_size]
    if dtype is not None:
        x_batch = x_batch.astype(dtype)
        y_batch = y_batch.astype(dtype)
    return x_batch, y_batch

def train_test_mnist_tensor_flow(epochs, batch_size, learning_rate, model_size, dtype=None, gpu_index=0):
    # GPU selection (via TensorFlow)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
            print(f"Using GPU: {gpus[gpu_index]}")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found, running on CPU.")

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Convert labels to categorical
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Initialize the generator with Numba-enhanced preprocessing
    train_gen = NumbaDataGenerator(x_train, y_train, batch_size, dtype)

    # Model definitions for small, medium, and large networks
    if model_size == 'small':
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')  # Output layer should have 10 units
        ])
    elif model_size == 'medium':
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(10, activation='softmax')  # Output layer should have 10 units
        ])
    else:  # 'huge'
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(1024, activation='relu'),
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(10, activation='softmax')  # Ensure output layer has 10 units
        ])

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_gen),
        epochs=epochs,
    )

    # Evaluation (apply dtype conversion on test data if specified)
    x_test = normalize_data(x_test)  # Normalize test data with Numba
    x_test = x_test.astype(dtype) if dtype is not None else x_test
    y_test = y_test.astype(dtype) if dtype is not None else y_test
    loss, accuracy = model.evaluate(x_test, y_test)
    
    return loss, accuracy

# Function to run each test and collect results
def run_test(test_name, test_function, workers=0, batch_size=0, filter_size=0, sample_rate=0, epochs=0, learning_rate=0, model_size="", data_type=None):
    monitor = ResourceMonitor()
    system_info = monitor.get_system_info()

    if platform.system() == "Linux":
        monitor.clear_cache()

    monitor.start_monitoring()
    start_time = time.time()

    loss = None
    if test_name in ["matrix_multiplication_test", "pca_test", "svd_test", "fft_test"]:
        test_function(batch_size, dtype=data_type, on_gpu=False)
    elif test_name == "convolution_test":
        test_function(filter_size, sample_rate, dtype=data_type, on_gpu=False)
    elif test_name == "train_test_mnist_tensor_flow":
        loss, accuracy = test_function(epochs, batch_size, learning_rate, model_size, dtype=data_type)
    else:
        test_function()
    
    end_time = time.time()
    monitor.stop_monitoring()

    average_usage = monitor.get_average_usage()
    elapsed_time = end_time - start_time
    result = f'loss: {loss}, accuracy: {accuracy}' if loss else "Not relevant"

    data_dict = {np.int16 : "int16", np.int32 : "int32", np.float32 : "float32", np.float64 : "float64", np.double : "double", None:"Default"}

    system_stats = {
        **system_info,
        "Run Type": "GPU",
        "Batch Size": batch_size,
        "Filter Size": filter_size,
        "Sample Rate": sample_rate,
        "Epochs": epochs,
        "Learning rate": learning_rate,
        "Model Size": model_size,
        "Data Type": data_dict[data_type],
        "Elapsed Time": elapsed_time,
        "Result": result,
        **average_usage
    }

    monitor.save_results(system_stats, f"numba_{test_name}", "./EngFinalProject/results/results_numba_cpu_2710.json")


# Function to run all tests
#def run_all_tests(learning_rate, data_type, batch_size):
def run_all_tests():
    print(f'*********** Running CPU Tests with Numba ***********')
    run_test("iterative_test", iterative_test)
    # data_types = [None, np.int16, np.int32, np.float32, np.double]
    # for data in data_types:
    matrix_sizes = [64, 512, 1024, 2048, 4096]
    for matrix in matrix_sizes:
        print(f'Running Matrix Multiplication test, matrix size : {matrix}')
        run_test("matrix_multiplication_test", matrix_multiplication_test, batch_size=matrix)
        print(f'Running PCA test, matrix size : {matrix}')
        run_test("pca_test", pca_test, batch_size=matrix)
        print(f'Running SVD test, matrix size : {matrix}')
        run_test("svd_test", svd_test, batch_size=matrix)

    filter_sizes = [256, 1024, 2048]
    sample_rates = [16000, 44100]
    for filter_size in filter_sizes:
        for sample_rate in sample_rates:
            print(f'Running Convolution test, Filter size : {filter_size}, Sample rate : {sample_rate}')
            run_test("convolution_test", convolution_test, filter_size=filter_size, sample_rate=sample_rate)

    batch_sizes = [256, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    for batch in batch_sizes:
        print(f'Running FFT test, batch size : {batch}')
        run_test("fft_test", fft_test, batch_size=batch)

    return 0
    if data_type == "1":
        dtype = None
    elif data_type == "2":
        dtype = np.int16
    elif data_type == "3":
        dtype = np.int32
    elif data_type == "4":
        dtype = np.float32
    elif data_type == "5":
        dtype = np.double
    else:
        raise "error data typeeee"


    if learning_rate == "1":
        rate = 0.001
    elif learning_rate == "2":
        rate = 0.01
    elif learning_rate == "3":
        rate = 0.1
    else:
        raise "error learning rateee"

    if batch_size == "1":
        batch = 256
    elif batch_size == "2":
        batch = 512
    elif batch_size == "3":
        batch = 1024
    else:
        raise "error batch sizeee"
    
    # mlist parameters
    epochs = [1, 2, 4, 6, 8, 10]  # Different learning rates for the MList train test
    model_sizes = ["small", "medium", "huge"]

    for epoch in epochs:
        for model in model_sizes:
                print(f'Running MNIST Tensorflow Training test with Numba, on GPU, Batch size : {batch}, Learning Rate : {rate}, Epochs : {epoch}, Model Size : {model}, Data Type : {dtype}')
                run_test("train_test_mnist_tensor_flow", train_test_mnist_tensor_flow, batch_size=batch, learning_rate=rate, epochs=epoch, model_size=model, data_type=dtype)




if __name__ == "__main__":
#    learning_rate = sys.argv[1]
#    data_type = sys.argv[2]
#    batch_size = sys.argv[3]
#    run_all_tests(learning_rate, data_type, batch_size)
    run_all_tests()
