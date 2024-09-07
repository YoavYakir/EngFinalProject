import time
from ...scripts.utility_functions import ResourceMonitor
from test_cython import matrix_multiplication, iterative_test, fibonacci_test, quicksort_test, pca_test, svd_test, gpu_stress_test

# Cython optimized functions are imported from test_cython.pyx

# Function to run each test and collect results
def run_test(test_name, test_function, batch_size=100):
    monitor = ResourceMonitor()
    system_info = monitor.get_system_info()

    monitor.start_monitoring()
    start_time = time.time()

    # Run the test
    result = test_function(batch_size) if test_name in ["matrix_multiplication_test", "pca_test", "svd_test", "gpu_stress_test"] else test_function()

    end_time = time.time()
    monitor.stop_monitoring()

    average_usage = monitor.get_average_usage()
    elapsed_time = end_time - start_time

    # Collect system stats, including system info
    system_stats = {
        **system_info,  # CPU info
        "Run Type": "CPU",  # Cython runs only on CPU
        "Batch Size": batch_size,  # Include batch size for analysis
        "Elapsed Time": elapsed_time,
        **average_usage
    }

    # Save results for this test
    monitor.save_results(system_stats, f"cython_{test_name}", "../../results/results.json", batch_size=batch_size)

# Function to run all tests
def run_all_tests(batch_size=100):
    tests = {
        "iterative_test": iterative_test,
        "matrix_multiplication_test": lambda batch_size: matrix_multiplication(batch_size, batch_size),
        "pca_test": lambda batch_size: pca_test(batch_size),
        "svd_test": lambda batch_size: svd_test(batch_size),
        "gpu_stress_test": lambda batch_size: gpu_stress_test(batch_size),
        "fibonacci_test": lambda: fibonacci_test(20),
        "quicksort_test": lambda: quicksort_test([5, 3, 8, 6, 7, 2, 1])
    }

    # Run all tests on CPU
    for test_name, test_function in tests.items():
        run_test(test_name, test_function, batch_size=batch_size)

if __name__ == "__main__":
    # Example batch sizes for testing
    batch_sizes = [100, 500, 1000]

    for batch_size in batch_sizes:
        print(f"Running tests with batch size: {batch_size}")
        run_all_tests(batch_size=batch_size)
