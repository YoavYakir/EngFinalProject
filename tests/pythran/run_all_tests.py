from ...scripts.utility_functions import ResourceMonitor
import time
import platform
import numpy as np
import random

#pythran export iterative_test()
def iterative_test():
    result = 0
    for i in range(1000000):
        result += random.randint(0, 10)
    return result

#pythran export matrix_multiplication_test(int)
def matrix_multiplication_test(batch_size=100):
    A = np.random.rand(batch_size, batch_size)
    B = np.random.rand(batch_size, batch_size)
    return np.dot(A, B)

#pythran export pca_test(int, int)
def pca_test(batch_size=100, n_components=2):
    data = np.random.rand(batch_size, batch_size)
    data_centered = data - np.mean(data, axis=0)
    cov_matrix = np.cov(data_centered.T)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
    sorted_idx = np.argsort(eigen_values)[::-1]
    eigen_vectors = eigen_vectors[:, sorted_idx]
    return np.dot(data_centered, eigen_vectors[:, :n_components])

#pythran export svd_test(int)
def svd_test(batch_size=100):
    data = np.random.rand(batch_size, batch_size)
    return np.linalg.svd(data, full_matrices=False)

#pythran export convolution_test(int, float[:,:])
def convolution_test(filter_size=256, filter_2d=None):
    if filter_2d is None:
        sample_rate = 44100
        t = np.linspace(0, 1, sample_rate)
        frequency = 440
        sound_filter = np.sin(2 * np.pi * frequency * t)
        filter_2d = np.reshape(sound_filter[:9], (3, 3))

    data = np.random.rand(filter_size, filter_size)
    result = np.empty_like(data)
    for i in range(1, data.shape[0] - 1):
        for j in range(1, data.shape[1] - 1):
            result[i, j] = np.sum(data[i - 1:i + 2, j - 1:j + 2] * filter_2d)
    return result

#pythran export fft_test(int)
def fft_test(length=256):
    data = np.random.rand(length)
    return np.fft.fft(data)

# GPU and TensorFlow-based functions (don't compile with Pythran)
def mnist_train(workers=1, epochs=10, batch_size=256, learning_rate=0.01):
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

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, workers=workers, use_multiprocessing=True)

    loss, accuracy = model.evaluate(x_test, y_test)
    
    return loss, accuracy


# Function to run and collect results for each test
def run_test(test_name, test_function, batch_size=100, filter_size=256, length=256, workers=1, epochs=10, learning_rate=0.01):
    monitor = ResourceMonitor()
    system_info = monitor.get_system_info()

    # On Linux we can clear the cache before running the test
    if platform.system() == "Linux":
        monitor.clear_cache()

    monitor.start_monitoring()
    start_time = time.time()

    # Run the test based on the test type
    if test_name == "matrix_multiplication_test":
        result = test_function(batch_size)
    elif test_name == "pca_test":
        result = test_function(batch_size, 2)
    elif test_name == "svd_test":
        result = test_function(batch_size)
    elif test_name == "convolution_test":
        result = test_function(filter_size)
    elif test_name == "fft_test":
        result = test_function(length)
    elif test_name == "mnist_train":
        result = test_function(workers, epochs, batch_size, learning_rate)
    else:
        result = test_function()
    
    end_time = time.time()
    monitor.stop_monitoring()

    average_usage = monitor.get_average_usage()
    elapsed_time = end_time - start_time

    # Collect system stats, including system info
    system_stats = {
        **system_info,  # CPU info
        "Run Type": "CPU",  # Add the missing "Run Type" key for CPU tests
        "Workers": workers,
        "Batch Size": batch_size,
        "Filter Size": filter_size,
        "Learning rate": learning_rate,
        "FFT Length": length,
        "Epochs": epochs,
        "Elapsed Time": elapsed_time,
        **average_usage
    }

    # Save results for this test
    monitor.save_results(system_stats, f"pythran_{test_name}", "./EngFinalProject/results/results.json")

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

    for test_name, test_function in tests.items():
        print(f'Running {test_name}')
        run_test(test_name, test_function, batch_size=batch_size)

if __name__ == "__main__":
    # Example batch sizes for testing
    batch_sizes = [256, 1024, 4096]

    for batch_size in batch_sizes:
        print(f"Running tests with batch size: {batch_size}")
        run_all_tests(batch_size=batch_size)
