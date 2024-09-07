import time
import dask
import dask.array as da
import cupy as cp
import cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from ...scripts.utility_functions import ResourceMonitor

# Initialize Dask client to manage CPU/GPU tasks
cluster = LocalCUDACluster()
client = Client(cluster)

# Function to simulate CPU-based data preparation using Dask
def prepare_data(batch_size):
    data = da.random.random((batch_size, batch_size), chunks=(batch_size // 2, batch_size // 2))
    return data.compute()

# Function to simulate GPU-based data processing using CuPy and CuDF
def process_data_on_gpu(data):
    gpu_data = cp.asarray(data)
    result = cp.dot(gpu_data, gpu_data.T)  # Example matrix multiplication on the GPU
    return cp.asnumpy(result)

# PCA Test (GPU using CuPy and CuDF)
def pca_test(batch_size=100, n_components=2):
    data = cp.random.rand(batch_size, batch_size)
    data_centered = data - cp.mean(data, axis=0)
    cov_matrix = cp.cov(data_centered.T)
    eigen_values, eigen_vectors = cp.linalg.eig(cov_matrix)
    sorted_idx = cp.argsort(eigen_values)[::-1]
    return cp.dot(data_centered, eigen_vectors[:, :n_components])

# SVD Test (GPU using CuPy)
def svd_test(batch_size=100):
    data = cp.random.rand(batch_size, batch_size)
    return cp.linalg.svd(data, full_matrices=False)

# GPU Stress Test (Intensive matrix multiplication to stress GPU)
def gpu_stress_test(batch_size=100):
    data = cp.random.rand(batch_size, batch_size)
    result = cp.dot(data, data.T)  # Stress the GPU with large matrix operations
    return result

# Function to run the full pipeline: data preparation (CPU) + processing (GPU)
def run_pipeline(batch_size, test_function):
    # Start monitoring resources
    monitor = ResourceMonitor()
    monitor.start_monitoring()

    start_time = time.time()
    
    # Step 1: Data Preparation (CPU)
    prepared_data = prepare_data(batch_size)
    cpu_time = time.time() - start_time

    # Step 2: Data Processing (GPU) with the provided test function
    gpu_start_time = time.time()
    processed_data = test_function(batch_size)
    gpu_time = time.time() - gpu_start_time

    # Stop monitoring
    monitor.stop_monitoring()
    average_usage = monitor.get_average_usage()

    # Collect resource usage stats
    system_info = monitor.get_system_info()
    system_stats = {
        **system_info,
        "Batch Size": batch_size,
        "CPU Preparation Time": cpu_time,
        "GPU Processing Time": gpu_time,
        **average_usage
    }

    # Save results
    monitor.save_results(system_stats, f"batch_{batch_size}_test", "../../results/results.json", batch_size=batch_size)
    return cpu_time, gpu_time

# Run different tests
def run_all_tests(batch_size):
    test_functions = {
        "Matrix Multiplication": process_data_on_gpu,
        "PCA": pca_test,
        "SVD": svd_test,
        "GPU Stress Test": gpu_stress_test
    }

    for test_name, test_function in test_functions.items():
        print(f"Running {test_name} with batch size: {batch_size}")
        run_pipeline(batch_size, test_function)

# Run the pipeline
if __name__ == "__main__":
    batch_sizes = [100, 500, 1000]  # Example batch sizes for testing

    for batch_size in batch_sizes:
        print(f"Running tests for batch size: {batch_size}")
        run_all_tests(batch_size)
