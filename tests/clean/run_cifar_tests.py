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
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import (Conv2D, BatchNormalization, Activation,
                                     MaxPooling2D, Dropout, Flatten, Dense)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_test_cifar(gpu_index, epochs, batch_size, learning_rate, model_size, dtype, workers):
    # List available GPUs and enable memory growth.
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Available GPUs:", physical_devices)
    print("Using device:", tf.test.gpu_device_name())
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    # Force operations to be placed on the selected GPU.
    with tf.device(f'/GPU:{gpu_index}'):
        # Load CIFAR-10 dataset.
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # Preprocess data: Normalize images and one-hot encode labels.
        x_train = x_train.astype('float32') / 255.0
        x_test  = x_test.astype('float32') / 255.0
        y_train = to_categorical(y_train, 10)
        y_test  = to_categorical(y_test, 10)

        # Convert data to TensorFlow constants on the GPU.
        x_train = tf.constant(x_train, dtype=dtype)
        x_test  = tf.constant(x_test, dtype=dtype)
        y_train = tf.constant(y_train, dtype=dtype)
        y_test  = tf.constant(y_test, dtype=dtype)

        # Build a tf.data.Dataset pipeline:
        # - Cache in memory to avoid repeated transfers.
        # - Shuffle, batch, and then copy to the GPU.
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(buffer_size=x_train.shape[0])
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.apply(tf.data.experimental.copy_to_device(f'/GPU:{gpu_index}'))
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = test_dataset.batch(batch_size)
        test_dataset = test_dataset.apply(tf.data.experimental.copy_to_device(f'/GPU:{gpu_index}'))
        test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # Build the model based on the selected size.
        if model_size == 'small':
            # A compact network architecture.
            model = Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(10, activation='softmax')
            ])
        elif model_size == 'medium':
            # A moderately deep network.
            model = Sequential([
                layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                layers.Flatten(),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(10, activation='softmax')
            ])
        else:  # 'huge'
            # A deep network similar to your provided architecture, adapted for CIFAR-10.
            model = Sequential([
                layers.Conv2D(128, (3, 3), activation='relu', input_shape=(32, 32, 3)),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                layers.Conv2D(256, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.Conv2D(256, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),

                layers.Conv2D(512, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.GlobalAveragePooling2D(),

                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(10, activation='softmax')
            ])

        # Compile the model.
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])

        # Train the model.
        history = model.fit(
            train_dataset,
            epochs=epochs,
            verbose=1
        )

        # Evaluate the model on test data.
        loss, test_accuracy = model.evaluate(test_dataset, verbose=1)

    return loss, test_accuracy
    
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
    monitor.save_results(system_stats, f"{test_name}", f"./results/{monitor.get_gpu_name()}/cifar/clean.json")
    

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
        run_test("cifar_clean", train_test_cifar, batch_size, epochs, learning_rate, model_size, data_type, workers)

