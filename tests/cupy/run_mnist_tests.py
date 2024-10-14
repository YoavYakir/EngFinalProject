from concurrent.futures import ThreadPoolExecutor
import datetime
import json
import logging
import os
import platform
import time
import subprocess
import numpy as np
from utilities import nsight_manager
from utilities import gpu_handling
from utilities.ResourceMonitor import ResourceMonitor
import cupy as cp
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import os
import subprocess


def run_test_with_nsys(test_name, batch_size, epochs, learning_rate, model_size, data_type, workers):
    """
    Run the test under Nsight profiling and save results to a single JSON file.
    """
    monitor = ResourceMonitor()
    system_info = monitor.get_system_info()
    
    # Profiling setup
    output_dir = "./nsight_outputs3"
    os.makedirs(output_dir, exist_ok=True)
    nsys_file = os.path.join(output_dir, f"{test_name}_{workers}_{batch_size}_{epochs}_{learning_rate}_{model_size}.nsys-rep")

    data_type_str = "None" if data_type is None else f"'{data_type}'"

    cmd = [
    "nsys", "profile",
    "--output", nsys_file.replace(".nsys-rep", ""),
    "--trace=cuda,cublas",
    "--force-overwrite=true",
    "--show-output=true",
    "python3", "-u", "-c",
    f"\"from tests.clean_python.run_mnist_tests import *; print(run_single_test('{test_name}', {batch_size}, {epochs}, {learning_rate}, '{model_size}', {data_type_str}, {workers}))\""
    ]


    print(f"Running Nsight profiling: {' '.join(cmd)}")
    try:
        process = subprocess.run(" ".join(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, timeout=240)
    except:
        print("@"*20 + "FAILURE_START" + "@"*20)
        if process.stderr:
            print(process.stderr.decode('utf-8'))
        if process.stdout:
            print(process.stdout.decode('utf-8'))
        print("@"*20 + "FAILURE_END" + "@"*20)
        print(f"Nsight Systems profiling failed:\n{process.stderr}")
        return


    if process.returncode != 0:
        print("@"*20 + "FAILURE_START" + "@"*20)
        print(process.stderr.decode('utf-8'))
        if process.stdout:
            print(process.stdout.decode('utf-8'))
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
        "Filter Size": 0,
        "Sample Rate": 0,
        "Workers": workers,
        "Epochs": epochs,
        "Learning rate": learning_rate,
        "Model Size": model_size,
        "Data Type": data_type or "Default",
        "Elapsed Time": test_result["elapsed_time"],
        "Test Name": test_name,
        "Loss": test_result["loss"],
        "Accuracy": test_result["accuracy"],
        **test_result["avg_usage"],
        **profiling_data,
    }

    # Save results to a single JSON file
    results_file = f"./results/mnist/cupy.json"
    monitor.save_results(system_stats, test_name, results_file)

# Custom data generator for CuPy arrays
def cupy_data_generator(gpu_index, x, y, batch_size, dtype, workers=1):
    num_samples = x.shape[0]
    with cp.cuda.Device(gpu_index):
        def process_batch(i):
            # Transfer slices to the active device
            x_batch = cp.asarray(x[i:i + batch_size], dtype=dtype)
            y_batch = cp.asarray(y[i:i + batch_size], dtype=dtype if dtype in [cp.float32, cp.float64] else cp.float32)
            
            # Perform type conversion if necessary
            if dtype in [cp.int32, cp.int16]:
                x_batch = (x_batch * 255).astype(dtype)  # Rescale for int types
            
            return cp.asnumpy(x_batch), cp.asnumpy(y_batch)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            while True:
                futures = [executor.submit(process_batch, i) for i in range(0, num_samples, batch_size)]
                for future in futures:
                    yield future.result()


# Function to train and test MNIST using TensorFlow and GPU
def train_test_mnist_cupy(gpu_index, epochs, batch_size, learning_rate, model_size, dtype=None, workers=1):
    sub_monitor = ResourceMonitor()
    sub_monitor.start_monitoring()
    start_time = time.time()

    with tf.device(f'/GPU:{gpu_index}'):
        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Normalize the data and convert to CuPy arrays
        x_train = cp.asarray(x_train / 255.0)
        x_test = cp.asarray(x_test / 255.0)

        # Convert labels to categorical (remains on the CPU for TensorFlow)
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

        # Use the custom CuPy data generator
        train_gen = cupy_data_generator(gpu_index, x_train, y_train, batch_size, dtype, workers)

        # Train the model using multiple workers for data loading
        history = model.fit(
            train_gen,  # Use the data generator
            steps_per_epoch=len(x_train) // batch_size,
            epochs=epochs,
        )

        # Evaluate the model (convert CuPy test data back to NumPy)
        loss, accuracy = model.evaluate(cp.asnumpy(x_test), cp.asnumpy(y_test))

    end_time = time.time()
    sub_monitor.stop_monitoring()
    avg_usage = sub_monitor.get_average_usage()
    elapsed_time = end_time - start_time    
    return loss, accuracy, elapsed_time, avg_usage

def numpy_data_generator(x, y, batch_size, dtype=None, workers=1):
    num_samples = x.shape[0]

    def process_batch(i):
        x_batch = x[i:i + batch_size]
        y_batch = y[i:i + batch_size]
        if dtype is not None:
            x_batch = x_batch.astype(dtype)
            y_batch = y_batch.astype(dtype)
        return x_batch, y_batch

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while True:
            futures = [executor.submit(process_batch, i) for i in range(0, num_samples, batch_size)]
            for future in futures:
                yield future.result()


def train_test_mnist_numpy(gpu_index, epochs, batch_size, learning_rate, model_size, dtype=None, workers=1):
    sub_monitor = ResourceMonitor()
    sub_monitor.start_monitoring()
    start_time = time.time()

    with tf.device(f'/GPU:{gpu_index}'):
        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Normalize the data
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        # Convert labels to categorical (for TensorFlow compatibility)
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        # Create a custom NumPy data generator for the training data
        train_gen = numpy_data_generator(x_train, y_train, batch_size, dtype, workers)

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

        # Train the model using multiple workers for data loading
        history = model.fit(
            train_gen,  # Use the custom data generator
            steps_per_epoch=x_train.shape[0] // batch_size,
            epochs=epochs,
        )

        # Evaluate the model (convert test data and labels to specified dtype)
        x_test = x_test.astype(dtype) if dtype is not None else x_test
        y_test = y_test.astype(dtype) if dtype is not None else y_test
        loss, accuracy = model.evaluate(x_test, y_test)
    
    end_time = time.time()
    sub_monitor.stop_monitoring()
    avg_usage = sub_monitor.get_average_usage()
    elapsed_time = end_time - start_time    
    return loss, accuracy, elapsed_time, avg_usage


# Function to run GPU tests
def run_test(test_name, test_function, batch_size=0, epochs=0, learning_rate=0, model_size="", data_type=None, workers=1):
    monitor = ResourceMonitor()
    system_info = monitor.get_system_info()

    # On Linux we can clear the cache before running the test
    if platform.system() == "Linux":
        monitor.clear_cache()

    monitor.start_monitoring()
    start_time = time.time()


    loss, accuracy = test_function(epochs, batch_size, learning_rate, model_size, dtype=data_type) 

    end_time = time.time()
    monitor.stop_monitoring()

    average_usage = monitor.get_average_usage()
    elapsed_time = end_time - start_time

    result = f'loss: {loss}, accuracy: {accuracy}'

    data_dict = {cp.int16 : "int16", cp.int32 : "int32", cp.float32 : "float32", cp.float64 : "float64", cp.double : "double", None:"Default"}

    # Collect system stats, including system info and epochs
    system_stats = {
        **system_info,  # CPU and GPU info
        "Run Type": "GPU",  # Indicate CPU or GPU run
        "Batch Size": batch_size,  # Batch size for tests
        "Filter Size": 0,
        "Sample Rate": 0,
        "Epochs": epochs,
        "Learning rate": learning_rate,
        "Model Size": model_size,
        "Data Type": data_dict[data_type],
        "Elapsed Time": elapsed_time,
        "Result": result,
        **average_usage
    }

    # Save results for this test
    monitor.save_results(system_stats, f"{test_name}", f"./results/mnist/cupy.json")
    

def run_single_test(test_name, batch_size, epochs, learning_rate, model_size, data_type, workers):
    """
    Run a single test and collect accuracy and loss.
    Ensure GPU setup is handled in the current process.
    """
    # GPU setup in the current process
    # gpu_index = gpu_handling.setup_gpu()
    gpu_index = 3
    gpu_handling.clear_gpu_memory(gpu_index)

    data_type_dict = {"None":None, "cp.int16":cp.int16, "cp.int32":cp.int32, "cp.float32":cp.float32, "cp.double":cp.double}

    if test_name == "train_test_mnist_numpy":
        loss, accuracy, elapsed_time, avg_usage = train_test_mnist_numpy(gpu_index, epochs, batch_size, learning_rate, model_size, data_type_dict[data_type], workers)
    elif test_name == "train_test_mnist_cupy":
        loss, accuracy, elapsed_time, avg_usage = train_test_mnist_cupy(gpu_index, epochs, batch_size, learning_rate, model_size, data_type_dict[data_type], workers)
    else:
        raise ValueError(f"Unknown test name: {test_name}")

    gpu_handling.reset_tensorflow_gpu()

    print(json.dumps({"loss": loss, "accuracy": accuracy, "elapsed_time": elapsed_time, "avg_usage" : avg_usage}))

if __name__ == "__main__":
    ResourceMonitor.init_results_file(f"./results/mnist/cupy.json")

    worker_list = [1, 2, 4, 6, 8, 10]
    batch_size_list = [256, 512, 1024]
    epochs_list = [1, 2, 4, 6, 8, 10]
    learning_rate_list = [0.001, 0.01, 0.1]
    data_type_list = ["None", "cp.int16", "cp.int32", "cp.float32", "cp.double"]
    model_size_list = ["small", "medium", "large"]

    for worker in worker_list:
        for batch in batch_size_list:
            for epoch in epochs_list:
                for learning_rate in learning_rate_list:
                    for data_type in data_type_list:
                        for model_size in model_size_list:
                            # print(f'\nRunning MNIST train test with NumPy on GPU, workers : {worker}, batch size : {batch}, epoch : {epoch}, learning_rate : {learning_rate}, model_size : {model_size}, data_type : {data_type}')
                            # run_test_with_nsys("train_test_mnist_numpy", batch, epoch, learning_rate, model_size, data_type, worker)                        
                            print(f'\nRunning MNIST train test with CuPy on GPU, workers : {worker}, batch size : {batch}, epoch : {epoch}, learning_rate : {learning_rate}, model_size : {model_size}, data_type : {data_type}')
                            run_test_with_nsys("train_test_mnist_cupy", batch, epoch, learning_rate, model_size, data_type, worker)

    ResourceMonitor.fix_json_file(f"./results/mnist/cupy.json")



