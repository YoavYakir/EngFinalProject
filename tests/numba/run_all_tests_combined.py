import time
import numpy as np
import numba as nb
import os
import sys
import platform
import random
from ...scripts.utility_functions import ResourceMonitor
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, Sequence
from numba import njit, cuda
import cupy as cp
from scipy.fftpack import fft as scipy_fft

# CuPy-enhanced and Numba-augmented Data Generator
class CuPyNumbaDataGenerator(Sequence):
    def __init__(self, x, y, batch_size, dtype=cp.float32):
        self.x = cp.array(x, dtype=dtype) / 255.0  # Normalize with CuPy on GPU
        self.y = cp.array(y, dtype=dtype)
        self.batch_size = batch_size
        self.num_samples = x.shape[0]

    def __len__(self):
        return int(cp.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        # Get batch and apply augmentations
        x_batch = self.x[index * self.batch_size:(index + 1) * self.batch_size]
        y_batch = self.y[index * self.batch_size:(index + 1) * self.batch_size]
        x_batch = apply_random_rotation(x_batch)  # Apply Numba-based GPU augmentation
        return x_batch.get(), y_batch.get()  # Transfer to CPU for TensorFlow

    def on_epoch_end(self):
        pass

# Numba GPU kernel for random rotation augmentation
@cuda.jit
def random_rotate_kernel(x, angle):
    idx = cuda.grid(1)
    if idx < x.shape[0]:  # Process only valid indices
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                x[idx, i, j] *= angle

def apply_random_rotation(batch):
    """Apply random rotation to each image in the batch using Numba."""
    angles = cp.random.uniform(0.9, 1.1, size=batch.shape[0], dtype=cp.float32)
    angles_device = cuda.to_device(angles)
    random_rotate_kernel[batch.shape[0], 1](batch, angles_device)  # Apply in-place
    return batch

# Main MNIST test function with TensorFlow training on GPU
def train_test_mnist_tensor_flow(epochs, batch_size, learning_rate, model_size, dtype=cp.float32, gpu_index=0):
    # Set up GPU configuration in TensorFlow
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

    # Load and preprocess MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Use CuPy and Numba-augmented data generator
    train_gen = CuPyNumbaDataGenerator(x_train, y_train, batch_size, dtype=dtype)

    # Define the model based on chosen size
    if model_size == 'small':
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
    elif model_size == 'medium':
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(10, activation='softmax')
        ])
    else:
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(1024, activation='relu'),
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(10, activation='softmax')
        ])

    # Compile model with GPU-optimized Adam optimizer
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model using the data generator
    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_gen),
        epochs=epochs,
    )

    # Evaluate model with CuPy for test data
    x_test = cp.array(x_test, dtype=dtype) / 255.0
    loss, accuracy = model.evaluate(x_test.get(), y_test)
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
        test_function(batch_size, dtype=data_type, on_gpu=True)
    elif test_name == "convolution_test":
        test_function(filter_size, sample_rate, dtype=data_type, on_gpu=True)
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

    monitor.save_results(system_stats, f"numba_{test_name}", "./EngFinalProject/results/results_numba_MNIST.json")


# Function to run all tests
def run_all_tests(learning_rate, data_type, batch_size):
# def run_all_tests():
    print(f'*********** Running CPU Tests with Numba ***********')
    # data_types = [None, np.int16, np.int32, np.float32, np.double]
    # for data in data_types:
        # matrix_sizes = [64, 512, 1024]
        # for matrix in matrix_sizes:
        #     print(f'Running Matrix Multiplication test, matrix size : {matrix}, data type : {data}')
        #     run_test("matrix_multiplication_test", matrix_multiplication_test, batch_size=matrix, data_type=data)
        #     print(f'Running PCA test, matrix size : {matrix}, data type : {data}')
        #     run_test("pca_test", pca_test, batch_size=matrix, data_type=data)
        #     # print(f'Running SVD test, matrix size : {matrix}')
        #     # run_test("svd_test", svd_test, batch_size=matrix)

        # filter_sizes = [256, 1024, 4096]
        # sample_rates = [16000, 44100]
        # for filter_size in filter_sizes:
        #     for sample_rate in sample_rates:
        #         print(f'Running Convolution test, Filter size : {filter_size}, Sample rate : {sample_rate}, data type : {data}')
        #         run_test("convolution_test", convolution_test, filter_size=filter_size, sample_rate=sample_rate, data_type=data)

        # batch_sizes = [256, 1024, 4096]
        # for batch in batch_sizes:
        #     print(f'Running FFT test, batch size : {batch}, data type : {data}')
        #     run_test("fft_test", fft_test, batch_size=batch, data_type=data)

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
    learning_rate = sys.argv[1]
    data_type = sys.argv[2]
    batch_size = sys.argv[3]
    run_all_tests(learning_rate, data_type, batch_size)

    # run_all_tests()
