import os
import sys
import platform
from ...scripts.utility_functions import ResourceMonitor
import dask.array as da
import cupy as cp
import cudf
import dask_cudf
from cuml.decomposition import PCA as cuPCA
from dask.distributed import Client, LocalCluster
import time
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
os.environ["NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY"] = "1"

# Matrix Multiplication Test on GPU
def matrix_multiplication_test(batch_size, dtype=None, cupy=False):
    if cupy:
        # Generate matrices on the GPU using CuPy
        A = cp.random.rand(batch_size, batch_size).astype(dtype or cp.float32)
        B = cp.random.rand(batch_size, batch_size).astype(dtype or cp.float32)
        
        # Convert to cuDF DataFrames
        A_df = cudf.DataFrame.from_pandas(cp.asnumpy(A))
        B_df = cudf.DataFrame.from_pandas(cp.asnumpy(B))
        
        # Convert to Dask-cuDF for distributed computation
        A_dask = dask_cudf.from_cudf(A_df, npartitions=1)
        B_dask = dask_cudf.from_cudf(B_df, npartitions=1)
        
        # Perform matrix multiplication (example, customize as needed for your specific test)
        result = A_dask.dot(B_dask)
        return result
    else:
        # CPU version or other implementation as fallback
        # Customize this section if needed
        A = np.random.rand(batch_size, batch_size).astype(dtype or np.float32)
        B = np.random.rand(batch_size, batch_size).astype(dtype or np.float32)
        return np.dot(A, B)

# SVD Test on GPU
def svd_test(batch_size, dtype=cp.float32, cupy=True):
    if cupy:
        # Use Dask + CuPy for GPU operations
        data = da.random.random((batch_size, batch_size), chunks=(batch_size, batch_size)).map_blocks(cp.asarray).astype(dtype)
    else:
        # Use Dask + cuDF for GPU operations without CuPy
        data = dask_cudf.from_cudf(cudf.DataFrame.from_gpu_matrix(cp.random.rand(batch_size, batch_size).astype(dtype)), chunksize=batch_size)
    
    u, s, vt = da.linalg.svd(data)
    return u.compute(), s.compute(), vt.compute()

# PCA Test on GPU with cuML
def pca_test(batch_size, n_components=2, dtype=cp.float32, cupy=True):
    if cupy:
        # Use cuML PCA with CuPy array
        data = cp.random.rand(batch_size, batch_size).astype(dtype)
    else:
        # Use cuDF DataFrame for PCA without CuPy
        data = cudf.DataFrame(cp.random.rand(batch_size, batch_size).astype(dtype))

    pca = cuPCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    return transformed_data

# Convolution Test using CuPy with GPU
def convolution_test(filter_size, sample_rate, dtype=cp.float32, cupy=True):
    # Generate 3x3 sound filter
    t = cp.linspace(0, 1, sample_rate)
    frequency = 440
    sound_filter = cp.sin(2 * cp.pi * frequency * t)
    filter_2d = sound_filter[:9].reshape(3, 3).astype(dtype)

    if cupy:
        # Use CuPy array for GPU-based convolution
        data = cp.random.rand(filter_size, filter_size).astype(dtype)
        padded_data = cp.pad(data, pad_width=1, mode='constant', constant_values=0)
        result = cp.empty_like(data)

        # Perform convolution
        for i in range(1, padded_data.shape[0] - 1):
            for j in range(1, padded_data.shape[1] - 1):
                region = padded_data[i - 1:i + 2, j - 1:j + 2]
                result[i - 1, j - 1] = cp.sum(region * filter_2d)
    else:
        # Use cuDF DataFrame for GPU convolution without CuPy
        data = cudf.DataFrame.from_gpu_matrix(cp.random.rand(filter_size, filter_size).astype(dtype))
        result = data.copy()

        # Apply convolution
        for i in range(1, filter_size - 1):
            for j in range(1, filter_size - 1):
                region = data.iloc[i - 1:i + 2, j - 1:j + 2].to_pandas().values
                result.iloc[i, j] = (region * filter_2d.get()).sum()
    
    return result

# FFT Test on GPU
def fft_test(batch_size, dtype=cp.float32, cupy=True):
    if cupy:
        data = cp.random.rand(batch_size).astype(dtype)
        frequencies = cp.fft.fft(data)
        return frequencies
    else:
        data = cudf.Series(cp.random.rand(batch_size).astype(dtype))
        frequencies = data.to_pandas().pipe(cp.fft.fft)
        return frequencies

def train_test_mnist_tensor_flow(batch_size, epochs, learning_rate, model_size="small", dtype=cp.float32, cupy=True):
    # Load MNIST dataset using Dask for distributed GPU processing
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize and distribute MNIST dataset
    x_train, x_test = x_train / 255.0, x_test / 255.0

    if cupy:
        # Convert data to Dask arrays with CuPy
        x_train = da.from_array(cp.asarray(x_train), chunks=(batch_size, 28, 28))
        y_train = da.from_array(cp.asarray(to_categorical(y_train, 10)), chunks=(batch_size, 10))
        x_test = da.from_array(cp.asarray(x_test), chunks=(batch_size, 28, 28))
        y_test = da.from_array(cp.asarray(to_categorical(y_test, 10)), chunks=(batch_size, 10))
    else:
        # Use Dask-cuDF for data handling without CuPy
        x_train = dask_cudf.from_cudf(cudf.DataFrame.from_gpu_matrix(cp.asarray(x_train).reshape(-1, 28*28)), chunksize=batch_size)
        y_train = dask_cudf.from_cudf(cudf.DataFrame(to_categorical(y_train, 10)), chunksize=batch_size)
        x_test = dask_cudf.from_cudf(cudf.DataFrame.from_gpu_matrix(cp.asarray(x_test).reshape(-1, 28*28)), chunksize=batch_size)
        y_test = dask_cudf.from_cudf(cudf.DataFrame(to_categorical(y_test, 10)), chunksize=batch_size)
    
    # Define the model structure
    if model_size == "small":
        model = Sequential([Flatten(input_shape=(28, 28)), Dense(128, activation='relu'), Dense(10, activation='softmax')])
    elif model_size == "medium":
        model = Sequential([Flatten(input_shape=(28, 28)), Dense(512, activation='relu'), Dense(256, activation='relu'), Dense(10, activation='softmax')])
    else:  # "large"
        model = Sequential([Flatten(input_shape=(28, 28)), Dense(1024, activation='relu'), Dense(512, activation='relu'), Dense(256, activation='relu'), Dense(10, activation='softmax')])

    # Compile the model for GPU processing
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Convert Dask arrays to NumPy for model training with TensorFlow (limited support for Dask arrays in Keras)
    x_train_np = x_train.compute().get() if cupy else x_train.to_dask_array().compute()
    y_train_np = y_train.compute().get() if cupy else y_train.to_dask_array().compute()
    x_test_np = x_test.compute().get() if cupy else x_test.to_dask_array().compute()
    y_test_np = y_test.compute().get() if cupy else y_test.to_dask_array().compute()

    # Train the model on GPU
    model.fit(x_train_np, y_train_np, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Evaluate the model on GPU
    loss, accuracy = model.evaluate(x_test_np, y_test_np, batch_size=batch_size, verbose=0)
    return loss, accuracy

# Function to run each test and collect results
def run_test(test_name, test_function, workers=0, batch_size=0, filter_size=0, sample_rate=0, epochs=0, learning_rate=0, model_size="", data_type=None, cupy=False):
    monitor = ResourceMonitor()
    system_info = monitor.get_system_info()

    if platform.system() == "Linux":
        monitor.clear_cache()

    monitor.start_monitoring()
    start_time = time.time()

    loss = None
    if test_name in ["matrix_multiplication_test", "pca_test", "svd_test", "fft_test"]:
        test_function(batch_size, dtype=data_type, cupy=cupy)
    elif test_name == "convolution_test":
        test_function(filter_size, sample_rate, dtype=data_type, cupy=cupy)
    elif test_name == "train_test_mnist_tensor_flow":
        loss, accuracy = test_function(epochs, batch_size, learning_rate, model_size, dtype=data_type, cupy=cupy)
    else:
        test_function()
    
    end_time = time.time()
    monitor.stop_monitoring()

    average_usage = monitor.get_average_usage()
    elapsed_time = end_time - start_time
    result = f'loss: {loss}, accuracy: {accuracy}' if loss else "Not relevant"

    data_dict = {cp.int16 : "int16", cp.int32 : "int32", cp.float32 : "float32", cp.float64 : "float64", cp.double : "double", None:"Default"}

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

    monitor.save_results(system_stats, f"numba_{test_name}", "./EngFinalProject/results/results_dask_GPU.json")


# Function to run all tests
# def run_all_tests(learning_rate, data_type, batch_size):
def run_all_tests():
    print(f'*********** Running GPU Tests with dask ***********')
    data_types = [None, cp.int16, cp.int32, cp.float32, cp.double]
    for data in data_types:
        matrix_sizes = [64, 512, 1024]
        for matrix in matrix_sizes:
            print(f'Running Matrix Multiplication test, matrix size : {matrix}, data type : {data}')
            run_test("matrix_multiplication_test", matrix_multiplication_test, batch_size=matrix, data_type=data)
            print(f'Running PCA test, matrix size : {matrix}, data type : {data}')
            run_test("pca_test", pca_test, batch_size=matrix, data_type=data)
            print(f'Running SVD test, matrix size : {matrix}')
            run_test("svd_test", svd_test, batch_size=matrix)

        filter_sizes = [256, 1024, 4096]
        sample_rates = [16000, 44100]
        for filter_size in filter_sizes:
            for sample_rate in sample_rates:
                print(f'Running Convolution test, Filter size : {filter_size}, Sample rate : {sample_rate}, data type : {data}')
                run_test("convolution_test", convolution_test, filter_size=filter_size, sample_rate=sample_rate, data_type=data)

        batch_sizes = [256, 1024, 4096]
        for batch in batch_sizes:
            print(f'Running FFT test, batch size : {batch}, data type : {data}')
            run_test("fft_test", fft_test, batch_size=batch, data_type=data)

    # if data_type == "1":
    #     dtype = None
    # elif data_type == "2":
    #     dtype = cp.int16
    # elif data_type == "3":
    #     dtype = cp.int32
    # elif data_type == "4":
    #     dtype = cp.float32
    # elif data_type == "5":
    #     dtype = cp.double
    # else:
    #     raise "error data typeeee"


    # if learning_rate == "1":
    #     rate = 0.001
    # elif learning_rate == "2":
    #     rate = 0.01
    # elif learning_rate == "3":
    #     rate = 0.1
    # else:
    #     raise "error learning rateee"

    # if batch_size == "1":
    #     batch = 256
    # elif batch_size == "2":
    #     batch = 512
    # elif batch_size == "3":
    #     batch = 1024
    # else:
    #     raise "error batch sizeee"
    
    # mlist parameters
    # epochs = [1, 2, 4, 6, 8, 10]  # Different learning rates for the MList train test
    # model_sizes = ["small", "medium", "huge"]

    # for epoch in epochs:
    #     for model in model_sizes:
    #             print(f'Running MNIST Tensorflow Training test with Numba, on GPU, Batch size : {batch}, Learning Rate : {rate}, Epochs : {epoch}, Model Size : {model}, Data Type : {dtype}')
    #             run_test("train_test_mnist_tensor_flow", train_test_mnist_tensor_flow, batch_size=batch, learning_rate=rate, epochs=epoch, model_size=model, data_type=dtype)




if __name__ == "__main__":
    # learning_rate = sys.argv[1]
    # data_type = sys.argv[2]
    # batch_size = sys.argv[3]
    # run_all_tests(learning_rate, data_type, batch_size)

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    run_all_tests()
