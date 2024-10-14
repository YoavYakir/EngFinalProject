import time
import numpy as np
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
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.utils import Sequence


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

        # Apply dtype conversion if specified
 #       if self.dtype is not None:
 #           x_batch = x_batch.astype(self.dtype)
 #           y_batch = y_batch.astype(self.dtype)

#        return x_batch, y_batch

#    def on_epoch_end(self):
#        pass


# Iterative Test (CPU only)
# pythran export iterative_test(int, int)
def iterative_test(iterations=1000000, num=100):
    result = 0
    for i in range(iterations):
        result += random.randint(0, num)
    return result


# Matrix Multiplication Test - Optimized with Pythran
# pythran export matrix_multiplication_test(int, str or None)
def matrix_multiplication_test(batch_size, dtype=None):
    A = np.random.rand(batch_size, batch_size).astype(dtype)
    B = np.random.rand(batch_size, batch_size).astype(dtype)
    return np.dot(A, B)


# PCA Test - Optimized with Pythran
# pythran export pca_test(int, int, str or None)
def pca_test(batch_size, n_components=2, dtype=None):
    data = np.random.rand(batch_size, batch_size).astype(dtype)
    data_centered = data - np.mean(data, axis=0)
    cov_matrix = np.cov(data_centered.T)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
    sorted_idx = np.argsort(eigen_values)[::-1]
    eigen_vectors = eigen_vectors[:, sorted_idx]
    return np.dot(data_centered, eigen_vectors[:, :n_components])


# SVD Test - Optimized with Pythran
# pythran export svd_test(int, str or None)
def svd_test(batch_size, dtype=None):
    data = np.random.rand(batch_size, batch_size).astype(dtype)
    return np.linalg.svd(data, full_matrices=False)


# Convolution Test (1-second sound filter) - Optimized with Pythran
# pythran export convolution_test(int, int, str or None)
def convolution_test(filter_size, sample_rate, dtype=None):
    t = np.linspace(0, 1, sample_rate)
    frequency = 440
    sound_filter = np.sin(2 * np.pi * frequency * t)
    filter_2d = np.reshape(sound_filter[:9], (3, 3))

    data = np.random.rand(filter_size, filter_size).astype(dtype)
    result = np.empty_like(data)
    for i in range(1, data.shape[0] - 1):
        for j in range(1, data.shape[1] - 1):
            result[i, j] = np.sum(data[i - 1:i + 2, j - 1:j + 2] * filter_2d)
    return result


# FFT Test - Optimized with Pythran
# pythran export fft_test(int, str or None)
def fft_test(batch_size, dtype=None):
    data = np.random.rand(batch_size).astype(dtype)
    return np.fft.fft(data)


# Train MNIST Model with TensorFlow (same as before, using Numpy generator)
def train_test_mnist(workers, epochs, batch_size, learning_rate, model_size, dtype=None, gpu_index=0):
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

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    train_gen = NumpyDataGenerator(x_train, y_train, batch_size, dtype)
    
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

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_gen),
        epochs=epochs,
        workers=workers,
        use_multiprocessing=True
    )

    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, accuracy


# Function to run each test and collect results
def run_test(test_name, test_function, batch_size=0, filter_size=0, sample_rate=0, epochs=0, learning_rate=0, model_size="", data_type=None):
    monitor = ResourceMonitor()
    system_info = monitor.get_system_info()

    if platform.system() == "Linux":
        monitor.clear_cache()

    monitor.start_monitoring()
    start_time = time.time()

    loss = None
    if test_name in ["matrix_multiplication_test", "pca_test", "svd_test", "fft_test"]:
        test_function(batch_size, dtype=data_type)
    elif test_name == "convolution_test":
        test_function(filter_size, sample_rate, dtype=data_type)
    elif test_name == "train_test_mnist":
        loss, accuracy = test_function(1, epochs, batch_size, learning_rate, model_size, dtype=data_type)
    else:
        test_function()
    
    end_time = time.time()
    monitor.stop_monitoring()

    average_usage = monitor.get_average_usage()
    elapsed_time = end_time - start_time
    result = f'loss: {loss}, accuracy: {accuracy}' if loss else "Not relevant"

    system_stats = {
        **system_info,
        "Run Type": "CPU",
        "Batch Size": batch_size,
        "Filter Size": filter_size,
        "Sample Rate": sample_rate,
        "Epochs": epochs,
        "Learning rate": learning_rate,
        "Model Size": model_size,
        "Elapsed Time": elapsed_time,
        "Result": result,
        **average_usage
    }

    monitor.save_results(system_stats, f"pythran_{test_name}", "./EngFinalProject/results/results_pythran_cpu2710.json")


# Function to run all tests
def run_all_tests():
    print(f'*********** Running CPU Tests with Pythran ***********')

    run_test("iterative_test", iterative_test)

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

    # Run TensorFlow MNIST training test
    # print(f'Running MNIST Training test')
    # epochs = [1, 2, 5]
    # learning_rates = [0.001, 0.01]
    # model_sizes = ["small", "medium", "huge"]

    # for batch_size in batch_sizes:
    #     for learning_rate in learning_rates:
    #         for epoch in epochs:
    #             for model_size in model_sizes:
    #                 print(f'Running MNIST Training test, Batch size: {batch_size}, Learning Rate: {learning_rate}, Epochs: {epoch}, Model Size: {model_size}')
    #                 run_test("train_test_mnist", train_test_mnist, batch_size=batch_size, learning_rate=learning_rate, epochs=epoch, model_size=model_size)

if __name__ == "__main__":
    run_all_tests()
