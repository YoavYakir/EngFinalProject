import sys
import time
import numpy as np
import cupy as cp
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from numba import njit, prange
from utilities import gpu_handling
from utilities.ResourceMonitor import ResourceMonitor
from concurrent.futures import ThreadPoolExecutor

@njit(parallel=True, fastmath=True)
def numba_augment_batch(images):
    B, H, W, C = images.shape
    crop_H, crop_W = 28, 28
    out = np.empty_like(images)

    for i in prange(B):
        img = images[i].copy()
        
        # Random horizontal flip
        if np.random.rand() > 0.5:
            img = img[:, ::-1, :]

        # Random crop
        top = np.random.randint(0, H - crop_H + 1)
        left = np.random.randint(0, W - crop_W + 1)
        cropped = img[top:top + crop_H, left:left + crop_W, :]

        # Resize using nearest neighbor (NumPy broadcasting)
        resized = np.zeros((H, W, C), dtype=img.dtype)
        for h in prange(H):
            for w in prange(W):
                resized[h, w, :] = cropped[int(h * crop_H / H), int(w * crop_W / W), :]

        out[i] = resized
    return out

def numba_augment_batch_wrapper(images, labels):
    """
    Wrapper function to call the Numba-accelerated augmentation function.
    Converts the input to a contiguous numpy array and returns augmented batch.
    """
    images_np = np.array(images)  # Ensure it's a regular numpy array.
    images_np = np.ascontiguousarray(images_np)
    augmented = numba_augment_batch(images_np)
    return augmented, labels

# -----------------------------------------------------------------------------
# Generator that uses a ThreadPoolExecutor to run augmentation in parallel.
# -----------------------------------------------------------------------------
def augmented_batch_generator(x, y, batch_size, do_augment, executor):
    """
    Generator that yields augmented batches using Numba augmentation in a separate thread.
    
    Parameters:
      x: NumPy array of images, shape (N, 32, 32, 3), dtype uint8.
      y: NumPy array of labels.
      batch_size: Batch size.
      do_augment: Boolean flag.
      executor: ThreadPoolExecutor.
    
    Yields:
      Tuple (augmented_images, labels) for each batch.
    """
    # Normalize images once with vectorized NumPy.
    x_norm = x.astype('float32') / 255.0
    N = x_norm.shape[0]
    indices = np.arange(N)
    while True:
        np.random.shuffle(indices)
        for i in range(0, N, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_x = x_norm[batch_idx]
            batch_y = y[batch_idx]
            if do_augment:
                future = executor.submit(numba_augment_batch, batch_x)
                augmented = future.result()
            else:
                augmented = batch_x
            yield augmented, batch_y

# -----------------------------------------------------------------------------
# Training function using the generator.
# -----------------------------------------------------------------------------
def train_test_cifar(gpu_index=0, epochs=15, batch_size=512, learning_rate=0.001,
                     model_size='huge', do_augment=True, workers=4):
    # Configure GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Efficient Numba-based augmentation
    def augment_data(images):
        images_aug = np.empty_like(images)
        for i in range(images.shape[0]):
            images_aug[i] = numba_augment_batch(images[i])
        return images_aug

    # Augment training data if enabled
    if do_augment:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            x_train_batches = np.array_split(x_train, workers)
            results = list(executor.map(augment_data, x_train_batches))
            x_train = np.concatenate(results)

    # Create dataset pipelines
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(50000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Build model
    def build_model(size):
        if size == 'small':
            return Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.2),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(10, activation='softmax')
            ])
        elif size == 'medium':
            return Sequential([
                layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.3),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.3),
                layers.Flatten(),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                layers.Dense(10, activation='softmax')
            ])
        else:  # 'huge'
            return Sequential([
                layers.Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.3),
                layers.Conv2D(256, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.Conv2D(256, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.4),
                layers.Conv2D(512, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                layers.Dense(10, activation='softmax')
            ])

    model = build_model(model_size)

    # Compile model with optimizer and scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=1e-6)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    start_time = time.time()
    history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, callbacks=[lr_scheduler], verbose=1)
    end_time = time.time()

    # Evaluate model
    loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
    print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print(f"Training Time: {end_time - start_time:.2f} seconds")

    # Model summary
    return loss, test_accuracy

# =============================================================================
# Main entry point
# =============================================================================

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

    monitor.save_results(system_stats, f"{test_name}", f"./results/{monitor.get_gpu_name()}/cifar/numba.json")
    

if __name__ == "__main__":
    workers = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    epochs = int(sys.argv[3])
    learning_rate = float(sys.argv[4])
    model_size = sys.argv[5]

    data_type_list = [None]
    for data_type in data_type_list:
        print(f'\nRunning CIFAR train test on GPU NUMBA, workers: {workers}, batch size: {batch_size}, epochs: {epochs}, learning_rate: {learning_rate}, model_size: {model_size}, data_type: {data_type}')
        run_test("cifer_numba", train_test_cifar, batch_size, epochs, learning_rate, model_size, data_type, workers)
