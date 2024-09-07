import json
import matplotlib.pyplot as plt
import numpy as np

# Function to analyze results from JSON file and produce graphs
def analyze_results(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Collect data for analysis
    cpu_times = []
    gpu_times = []
    batch_sizes = []
    methods = []
    workers_list = []
    run_types = []

    # Collect information from the results
    for entry in data:
        if "Batch Size" in entry:
            batch_sizes.append(entry["Batch Size"])
            cpu_times.append(entry.get("CPU Preparation Time", 0))
            gpu_times.append(entry.get("GPU Processing Time", 0))
            methods.append(entry.get("Test Name", "Unknown"))
            workers_list.append(entry.get("Workers", 1))
            run_types.append(entry.get("Run Type", "CPU"))

    # Plot CPU and GPU times against batch sizes
    plt.figure()
    plt.plot(batch_sizes, cpu_times, label="CPU Preparation Time", color="blue")
    plt.plot(batch_sizes, gpu_times, label="GPU Processing Time", color="green")
    plt.xlabel("Batch Size")
    plt.ylabel("Time (seconds)")
    plt.title("CPU vs GPU Time for Different Batch Sizes")
    plt.legend()
    plt.show()

    # Analyze effect of workers on GPU time
    plt.figure()
    for method in set(methods):
        method_gpu_times = [gpu_times[i] for i in range(len(gpu_times)) if methods[i] == method and run_types[i] == "GPU"]
        method_workers = [workers_list[i] for i in range(len(workers_list)) if methods[i] == method and run_types[i] == "GPU"]
        plt.plot(method_workers, method_gpu_times, label=f"GPU Time for {method}")

    plt.xlabel("Number of Workers")
    plt.ylabel("GPU Processing Time (seconds)")
    plt.title("Effect of Workers on GPU Processing Time")
    plt.legend()
    plt.show()

    # Analyze CPU vs GPU times across different methods
    plt.figure()
    for method in set(methods):
        method_cpu_times = [cpu_times[i] for i in range(len(cpu_times)) if methods[i] == method]
        method_gpu_times = [gpu_times[i] for i in range(len(gpu_times)) if methods[i] == method]
        method_batch_sizes = [batch_sizes[i] for i in range(len(batch_sizes)) if methods[i] == method]
        plt.plot(method_batch_sizes, method_cpu_times, label=f"CPU Time for {method}", linestyle="--")
        plt.plot(method_batch_sizes, method_gpu_times, label=f"GPU Time for {method}")

    plt.xlabel("Batch Size")
    plt.ylabel("Time (seconds)")
    plt.title("CPU vs GPU Times Across Methods")
    plt.legend()
    plt.show()

    # Prediction of batch preparation times based on past data
    def predict_batch_time(batch_size, method_name):
        matching_entries = [(cpu_times[i], gpu_times[i], batch_sizes[i]) for i in range(len(batch_sizes))
                            if methods[i] == method_name]
        if not matching_entries:
            return None, None

        # Linear regression on past data
        cpu_batch_sizes, cpu_times_pred, gpu_times_pred = zip(*matching_entries)
        cpu_coefficients = np.polyfit(cpu_batch_sizes, cpu_times_pred, 1)
        gpu_coefficients = np.polyfit(cpu_batch_sizes, gpu_times_pred, 1)

        # Predict CPU and GPU times for the given batch size
        predicted_cpu_time = np.polyval(cpu_coefficients, batch_size)
        predicted_gpu_time = np.polyval(gpu_coefficients, batch_size)
        return predicted_cpu_time, predicted_gpu_time

    # Example prediction for batch preparation time
    batch_size_to_predict = 500
    method_to_predict = "matrix_multiplication_test"
    predicted_cpu, predicted_gpu = predict_batch_time(batch_size_to_predict, method_to_predict)
    
    if predicted_cpu is not None and predicted_gpu is not None:
        print(f"Predicted CPU time for batch size {batch_size_to_predict} in method {method_to_predict}: {predicted_cpu:.2f} seconds")
        print(f"Predicted GPU time for batch size {batch_size_to_predict} in method {method_to_predict}: {predicted_gpu:.2f} seconds")
    else:
        print(f"No matching data found for method {method_to_predict}")

# Example usage
if __name__ == "__main__":
    analyze_results("../results/results.json")
