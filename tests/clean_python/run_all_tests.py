import time
import numpy as np
import cupy as cp
import os
import sys
import platform
import random
from ...scripts.utility_functions import ResourceMonitor

# Iterative Test (CPU only)
def iterative_test():
    result = 0
    for i in range(1000000):
        result += random.randint(0,10)
    return result

# Matrix Multiplication Test (CPU using numpy, GPU using cupy)
def matrix_multiplication_test(on_gpu, batch_size=100):
    if on_gpu:
        # GPU run using CuPy
        with cp.cuda.Device(0):
            A = cp.random.rand(batch_size, batch_size)
            B = cp.random.rand(batch_size, batch_size)
            return cp.dot(A, B)
    else:
        # CPU run using NumPy
        A = np.random.rand(batch_size, batch_size)
        B = np.random.rand(batch_size, batch_size)
        return np.dot(A, B)
    
# PCA Test (CPU using NumPy, GPU using CuPy)
def pca_test(on_gpu, batch_size={64, 512, 1024}, n_components=2):
    if on_gpu:
        with cp.cuda.Device(0):
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
def svd_test(on_gpu, batch_size={64, 512, 1024}):
    if on_gpu:
        with cp.cuda.Device(0):
            data = cp.random.rand(batch_size, batch_size)
            return cp.linalg.svd(data, full_matrices=False)
    else:
        data = np.random.rand(batch_size, batch_size)
        return np.linalg.svd(data, full_matrices=False)


# Convolution Test (CPU using NumPy, GPU using CuPy) with 1-second sound filter
def convolution_test(on_gpu, filter_size={256, 1024, 4096}, sample_rate=44100):
    # Generate a sound filter (1-second sine wave)
    t = np.linspace(0, 1, sample_rate)  # 1 second duration at 44.1kHz sampling rate
    frequency = 440  # A typical tone at 440 Hz (A4 note)
    sound_filter = np.sin(2 * np.pi * frequency * t)  # Generate sine wave

    # Reshape the sound filter to 3x3 to match the sliding window
    filter_2d = np.reshape(sound_filter[:9], (3, 3))  # Reshape to 3x3

    if on_gpu:
        # GPU-based convolution using CuPy
        with cp.cuda.Device(0):
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
def fft_test(on_gpu, length=256, batch_size=256):
    if on_gpu:
        with cp.cuda.Device(0):
            data = cp.random.rand(length)
            return cp.fft.fft(data)
    else:
        data = np.random.rand(length)
        return np.fft.fft(data)


# MNIST Test using varying workers, epochs, batch_size, and learning rates
def train_test_mnist(on_gpu, workers=1, epochs=10, batch_size={64, 512, 1024}, learning_rate=0.01, model_size='small'):
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Model definitions for small, medium, and large networks
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
    else:  # 'huge'
        model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(1024, activation='relu'),
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(10, activation='softmax')
        ])

    # Compile the model with Adam optimizer and learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, workers=workers, use_multiprocessing=True)

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    
    return loss, accuracy

def train_test_mnist(on_gpu, workers=1, epochs=10, batch_size={64, 512, 1024}, learning_rate=0.01):
    data_size = 10000  # Simulated dataset size
    input_size = 100   # Input feature size
    output_size = 10   # Output feature size (for a 10-class problem)
    iterations = data_size // batch_size

    if on_gpu:
        with cp.cuda.Device(0):
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
def run_test(test_name, test_function, on_gpu=True, workers=1, batch_size=100, filter_size=256, length=256, epochs=10, learning_rate=0.01):
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
        result = test_function(on_gpu, length, batch_size)
    elif test_name == "train_test_mnist":
        result = test_function(on_gpu, workers, epochs, batch_size, learning_rate) 
    else:
        result = test_function()
    
    end_time = time.time()
    monitor.stop_monitoring()

    average_usage = monitor.get_average_usage()
    elapsed_time = end_time - start_time

    # Collect system stats, including system info and epochs
    system_stats = {
        **system_info,  # CPU and GPU info
        "Run Type": "GPU" if on_gpu else "CPU",  # Indicate CPU or GPU run
        "Workers": workers,  # Number of workers used for GPU tests
        "Batch Size": batch_size,  # Batch size for tests
        "Filter Size": filter_size,
        "Learning rate": learning_rate,
        "FFT Length": length,
        "Epochs": epochs,  # Include epochs in the system stats
        "Elapsed Time": elapsed_time,
        **average_usage
    }

    # Save results for this test
    monitor.save_results(system_stats, f"clean_{test_name}", "./EngFinalProject/results/results.json", workers=workers, batch_size=batch_size, epochs=epochs)

# Function to run all tests
def run_all_tests(batch_size):
    tests = {
        "iterative_test": iterative_test,
        "matrix_multiplication_test": matrix_multiplication_test,
        "pca_test": pca_test,
        "svd_test": svd_test,
        "convolution_test": convolution_test,
        "fft_test": fft_test,
    }

    # Run all tests on CPU
    for test_name, test_function in tests.items():
        print(f'Running {test_name}, on function {test_function}')
        run_test(test_name, test_function, on_gpu=False, batch_size=batch_size)

    # Run all tests on GPU
    for test_name, test_function in tests.items():
        print(f'Running {test_name}, on function {test_function}')
        run_test(test_name, test_function, on_gpu=True, batch_size=batch_size)

    # mlist parameters
    workers_list = [1, 2, 4, 8]  # Different worker counts for GPU runs
    batch_sizes = [256, 512, 1024]  # Different batch sizes for the MList train test
    learning_rates = [0.001, 0.01, 0.1]  # Different learning rates for the MList train test
    epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Different learning rates for the MList train test

    for workers in workers_list:
        # Run the train_test_mnist with varying batch sizes and learning rates
        print(f'Running mlist train with {workers} workers')
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                for epoch in epochs:
                    run_test("train_test_mnist", train_test_mnist, on_gpu=True, workers=workers, batch_size=batch_size, learning_rate=learning_rate, epochs=epoch)
                    run_test("train_test_mnist", train_test_mnist, on_gpu=False, workers=workers, batch_size=batch_size, learning_rate=learning_rate, epochs=epoch)

if __name__ == "__main__":
    # Example batch sizes for testing
    batch_sizes = [256, 1024, 4096]

    for batch_size in batch_sizes:
        print(f"Running tests with batch size: {batch_size}")
        run_all_tests(batch_size=batch_size)