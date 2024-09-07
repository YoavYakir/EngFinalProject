import time
from ...scripts.utility_functions import ResourceMonitor
from test_pythran import matrix_multiplication, iterative_test, fibonacci_test, quicksort_test

# Pythran optimized functions are imported from test_pythran.py

# Function to run each test and collect results
def run_test(test_name, test_function):
    monitor = ResourceMonitor()
    system_info = monitor.get_system_info()

    monitor.start_monitoring()
    start_time = time.time()

    # Run the test
    result = test_function()

    end_time = time.time()
    monitor.stop_monitoring()

    average_usage = monitor.get_average_usage()
    elapsed_time = end_time - start_time

    # Collect system stats, including system info
    system_stats = {
        **system_info,  # CPU info
        "Run Type": "CPU",  # Pythran runs only on CPU
        "Elapsed Time": elapsed_time,
        **average_usage
    }

    # Save results for this test
    monitor.save_results(system_stats, f"pythran_{test_name}", "../../results/results.json")

# Function to run all tests
def run_all_tests():
    tests = {
        "iterative_test": iterative_test,
        "matrix_multiplication_test": lambda: matrix_multiplication(100, 100),
        "fibonacci_test": lambda: fibonacci_test(20),
        "quicksort_test": lambda: quicksort_test([5, 3, 8, 6, 7, 2, 1])
    }

    # Run all tests on CPU
    for test_name, test_function in tests.items():
        run_test(test_name, test_function)

if __name__ == "__main__":
    # Run all tests (CPU only, as Pythran doesn't support GPU)
    run_all_tests()
