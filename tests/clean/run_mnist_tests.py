from concurrent.futures import ThreadPoolExecutor
import datetime
import sys
import os
import platform
import time
from utilities import gpu_handling
from utilities.ResourceMonitor import ResourceMonitor
import cupy as cp
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import (Conv2D, BatchNormalization, Activation,
                                     MaxPooling2D, Dropout, Flatten, Dense)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
# Function to train and test MNIST using TensorFlow and GPU
def train_test_mnist(gpu_index, epochs, batch_size, learning_rate, model_size, dtype=tf.float32, workers=1):
    # List available GPUs and show the one we plan to use
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Available GPUs:", physical_devices)
    print("Using device:", tf.test.gpu_device_name())
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

    with tf.device(f'/GPU:{gpu_index}'):
        # Load MNIST dataset (this happens on the CPU)
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Expand dims to include the channel dimension: (28,28) -> (28,28,1)
        x_train = x_train[..., None]
        x_test = x_test[..., None]

        # Normalize the data and convert to TensorFlow constants on the GPU
        x_train = tf.constant(x_train / 255.0, dtype=dtype)
        x_test = tf.constant(x_test / 255.0, dtype=dtype)
        y_train = tf.constant(to_categorical(y_train, 10), dtype=dtype)
        y_test = tf.constant(to_categorical(y_test, 10), dtype=dtype)

        # Build a tf.data.Dataset pipeline:
        # 1. Cache the data in memory (after transfer to GPU via copy_to_device)
        # 2. Shuffle, batch, and prefetch the data
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.cache()  # Cache in memory to avoid reloading
        train_dataset = train_dataset.shuffle(buffer_size=60000).batch(batch_size)
        train_dataset = train_dataset.apply(tf.data.experimental.copy_to_device(f'/GPU:{gpu_index}'))
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # For evaluation we can use the test data directly.
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = test_dataset.batch(batch_size)
        test_dataset = test_dataset.apply(tf.data.experimental.copy_to_device(f'/GPU:{gpu_index}'))
        test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # Build the CNN model.
        # We provide three options based on model_size: small, medium, and huge.
        if model_size == 'small':
            # A basic network with one conv block and one Dense layer.
            model = Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(10, activation='softmax')
            ])
        elif model_size == 'medium':
            # A deeper network with two conv blocks.
            model = Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(512, activation='relu'),
                layers.Dense(256, activation='relu'),
                layers.Dense(10, activation='softmax')
            ])
        else:  # 'huge'
            # A deeper network with two conv blocks plus extra dense layers and dropout.
            model = Sequential([
                # First conv block
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Second conv block
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                layers.Flatten(),
                # Dense layers
                layers.Dense(1024, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.Dense(10, activation='softmax')
            ])

        # Compile the model.
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model.
        history = model.fit(
            train_dataset,
            epochs=epochs,
            verbose=1,
        )

        # Evaluate the model.
        loss, accuracy = model.evaluate(x_test, y_test)

    return loss, accuracy

# Function to run GPU tests
def run_test(test_name, test_function, batch_size=0, epochs=0, learning_rate=0, model_size="", data_type=None, workers=1):
    monitor = ResourceMonitor(tensorflow=True)
    system_info = monitor.get_system_info()

    monitor.start_monitoring()
    start_time = time.time()

    loss, accuracy = test_function(monitor.gpu_index, epochs, batch_size, learning_rate, model_size, data_type, workers) 

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
    monitor.save_results(system_stats, f"{test_name}", f"./results/{monitor.get_gpu_name()}/mnist/clean.json")
    

if __name__ == "__main__":
    # Read command-line arguments
    workers = int(sys.argv[1])       # First argument: workers
    batch_size = int(sys.argv[2])    # Second argument: batch size
    epochs = int(sys.argv[3])        # Third argument: epochs
    learning_rate = float(sys.argv[4])  # Fourth argument: learning rate
    model_size = sys.argv[5]         # Fifth argument: model size

    data_type_list = [None]
    for data_type in data_type_list:
        print(f'\nRunning MNIST train test on GPU, workers : {workers}, batch size : {batch_size}, epoch : {epochs}, learning_rate : {learning_rate}, model_size : {model_size}, data_type : {data_type}')
        run_test("mnist_clean", train_test_mnist, batch_size, epochs, learning_rate, model_size, data_type, workers)

