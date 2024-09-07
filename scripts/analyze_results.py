import json
import matplotlib.pyplot as plt

# Function to analyze results from JSON file and produce graphs
def analyze_results(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Collect data for analysis
    cpu_times = []
    gpu_times = []
    batch_sizes = []

    for entry in data:
        if "Batch Size" in entry:
            batch_sizes.append(entry["Batch Size"])
            cpu_times.append(entry["CPU Preparation Time"])
            gpu_times.append(entry["GPU Processing Time"])
    
    # Plot CPU and GPU times against batch sizes
    plt.figure()
    plt.plot(batch_sizes, cpu_times, label="CPU Preparation Time")
    plt.plot(batch_sizes, gpu_times, label="GPU Processing Time")
    plt.xlabel("Batch Size")
    plt.ylabel("Time (seconds)")
    plt.title("CPU vs GPU Time for Different Batch Sizes")
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    analyze_results("../results/results.json")
