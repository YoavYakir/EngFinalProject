from concurrent.futures import ThreadPoolExecutor
import datetime
import sys
import os
import platform
import time
from utilities import gpu_handling
from utilities.ResourceMonitor import ResourceMonitor
from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cupy as cp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# -----------------------------------------------------------------------------
# Numba-accelerated augmentation function for MNIST.
# -----------------------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def numba_augment_batch(images):
    B, H, W, C = images.shape  # Expecting grayscale with channel dimension (B, 28, 28, 1)
    out = np.empty_like(images)
    for i in prange(B):
        temp = images[i, :, :, 0]  # Remove the single channel for processing
        if np.random.rand() > 0.5:  # Random horizontal flip
            temp = temp[:, ::-1]
        out[i, :, :, 0] = temp  # Restore channel dimension
    return out


def numba_augment_batch_wrapper(images, labels):
    images_np = np.ascontiguousarray(np.array(images))
    augmented = numba_augment_batch(images_np)
    return augmented, labels

def augmented_batch_generator(x, y, batch_size, do_augment, executor):
    x_norm = x.astype('float32') / 255.0
    N = x_norm.shape[0]
    indices = np.arange(N)
    while True:
        np.random.shuffle(indices)
        for i in range(0, N, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_x = x_norm[batch_idx][..., None]
            batch_y = y[batch_idx]
            if do_augment:
                future = executor.submit(numba_augment_batch, batch_x)
                augmented = future.result()
            else:
                augmented = batch_x
            yield augmented, batch_y

def train_test_mnist(gpu_index, epochs, batch_size, learning_rate, model_size, do_augment=True, workers=1):
    # Configure GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Available GPUs:", physical_devices)
    print("Using device:", tf.test.gpu_device_name())
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Load and preprocess MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[..., None].astype('float32') / 255.0
    x_test = x_test[..., None].astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Efficient Numba-based augmentation
    def augment_data(images):
        # Ensure shape is (B, 28, 28, 1) before passing to Numba
        images = images[..., None] if images.ndim == 3 else images
        images_aug = numba_augment_batch(images)
        return images_aug

    # Apply Numba augmentation if enabled
    if do_augment:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            x_train_batches = np.array_split(x_train, workers)
            results = list(executor.map(augment_data, x_train_batches))
            x_train = np.concatenate(results)

    # Create dataset pipelines
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Build the model based on size
    def build_model(size):
        if size == 'small':
            return Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(10, activation='softmax')
            ])
        elif size == 'medium':
            return Sequential([
                Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
                BatchNormalization(),
                Conv2D(64, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.3),
                Flatten(),
                Dense(512, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])
        else:  # 'huge'
            return Sequential([
                Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
                BatchNormalization(),
                Conv2D(128, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.3),
                Conv2D(256, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.4),
                Flatten(),
                Dense(1024, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(512, activation='relu'),
                Dropout(0.4),
                Dense(10, activation='softmax')
            ])

    # Compile model
    model = build_model(model_size)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    # Learning rate scheduler and early stopping
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=1e-6)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    start_time = time.time()
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        callbacks=[lr_scheduler, early_stopping],
        verbose=1
    )
    end_time = time.time()

    # Evaluate the model
    loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
    print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print(f"Training Time: {end_time - start_time:.2f} seconds")

    # Model summary
    return loss, test_accuracy


# Function to run GPU tests
def run_test(test_name, test_function, batch_size=0, epochs=0, learning_rate=0, model_size="", data_type=None, workers=1):
    monitor = ResourceMonitor(tensorflow=True)
    system_info = monitor.get_system_info()

    monitor.start_monitoring()
    start_time = time.time()

    loss, accuracy = test_function(monitor.gpu_index, epochs, batch_size, learning_rate, model_size, workers=workers)

    end_time = time.time()
    monitor.stop_monitoring()

    average_usage = monitor.get_average_usage()
    elapsed_time = end_time - start_time

    result = f'loss: {loss}, accuracy: {accuracy}'
    data_dict = {cp.int16 : "int16", cp.int32 : "int32", cp.float32 : "float32", cp.float64 : "float64", cp.double : "double", None:"Default"}

    # Collect system stats.
    system_stats = {
        **system_info,
        "Run Type": "GPU",
        "Batch Size": batch_size,
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

    monitor.save_results(system_stats, f"{test_name}", f"./results/{monitor.get_gpu_name()}/mnist/numba.json")
    

if __name__ == "__main__":
    # Command-line arguments: workers, batch_size, epochs, learning_rate, model_size.
    workers = int(sys.argv[1])       # e.g., 10
    batch_size = int(sys.argv[2])      # e.g., 512
    epochs = int(sys.argv[3])          # e.g., 15
    learning_rate = float(sys.argv[4]) # e.g., 0.001
    model_size = sys.argv[5]           # e.g., "huge"

    data_type_list = [None]
    for data_type in data_type_list:
        print(f'\nRunning MNIST train test on GPU, workers: {workers}, batch size: {batch_size}, epochs: {epochs}, learning_rate: {learning_rate}, model_size: {model_size}, data_type: {data_type}')
        run_test("mnist_numba", train_test_mnist, batch_size, epochs, learning_rate, model_size, data_type, workers)
