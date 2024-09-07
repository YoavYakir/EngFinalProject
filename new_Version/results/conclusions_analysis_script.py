import json
from scripts.analyze_results import analyze_results
import numpy as np

def draw_conclusions(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Initialize variables for analysis
    total_entries = len(data)
    batch_size_analysis = {}
    method_analysis = {}
    cpu_vs_gpu_analysis = {}
    worker_analysis = {}

    # Process each entry and build datasets for conclusions
    for entry in data:
        batch_size = entry.get("Batch Size", 0)
        method = entry.get("Test Name", "Unknown")
        workers = entry.get("Workers", 1)
        run_type = entry.get("Run Type", "CPU")
        cpu_time = entry.get("CPU Preparation Time", 0)
        gpu_time = entry.get("GPU Processing Time", 0)

        # Analyze by batch size
        if batch_size not in batch_size_analysis:
            batch_size_analysis[batch_size] = {"cpu_times": [], "gpu_times": []}
        batch_size_analysis[batch_size]["cpu_times"].append(cpu_time)
        batch_size_analysis[batch_size]["gpu_times"].append(gpu_time)

        # Analyze by method
        if method not in method_analysis:
            method_analysis[method] = {"cpu_times": [], "gpu_times": [], "batch_sizes": []}
        method_analysis[method]["cpu_times"].append(cpu_time)
        method_analysis[method]["gpu_times"].append(gpu_time)
        method_analysis[method]["batch_sizes"].append(batch_size)

        # Analyze CPU vs GPU times
        if run_type not in cpu_vs_gpu_analysis:
            cpu_vs_gpu_analysis[run_type] = []
        if run_type == "CPU":
            cpu_vs_gpu_analysis[run_type].append(cpu_time)
        elif run_type == "GPU":
            cpu_vs_gpu_analysis[run_type].append(gpu_time)

        # Analyze by workers
        if workers not in worker_analysis:
            worker_analysis[workers] = {"cpu_times": [], "gpu_times": []}
        worker_analysis[workers]["cpu_times"].append(cpu_time)
        worker_analysis[workers]["gpu_times"].append(gpu_time)

    # Draw conclusions
    print("=== Conclusions ===")

    # Conclusion 1: Batch Size Impact
    for batch_size, times in batch_size_analysis.items():
        avg_cpu_time = np.mean(times["cpu_times"])
        avg_gpu_time = np.mean(times["gpu_times"])
        if avg_gpu_time < avg_cpu_time:
            print(f"For batch size {batch_size}, GPU outperformed CPU by an average of {avg_cpu_time - avg_gpu_time:.2f} seconds.")
        else:
            print(f"For batch size {batch_size}, CPU was faster by an average of {avg_gpu_time - avg_cpu_time:.2f} seconds.")

    # Conclusion 2: Method Performance
    for method, times in method_analysis.items():
        avg_cpu_time = np.mean(times["cpu_times"])
        avg_gpu_time = np.mean(times["gpu_times"])
        if avg_gpu_time < avg_cpu_time:
            print(f"Method {method}: GPU consistently outperformed CPU by an average of {avg_cpu_time - avg_gpu_time:.2f} seconds.")
        else:
            print(f"Method {method}: CPU consistently outperformed GPU by an average of {avg_gpu_time - avg_cpu_time:.2f} seconds.")

    # Conclusion 3: Effect of Workers on GPU
    for workers, times in worker_analysis.items():
        avg_gpu_time = np.mean(times["gpu_times"])
        print(f"With {workers} workers, GPU processing time averaged {avg_gpu_time:.2f} seconds.")

    # Conclusion 4: General CPU vs GPU
    avg_cpu_time_overall = np.mean(cpu_vs_gpu_analysis["CPU"])
    avg_gpu_time_overall = np.mean(cpu_vs_gpu_analysis["GPU"])
    if avg_gpu_time_overall < avg_cpu_time_overall:
        print(f"Overall, GPU outperformed CPU by an average of {avg_cpu_time_overall - avg_gpu_time_overall:.2f} seconds.")
    else:
        print(f"Overall, CPU outperformed GPU by an average of {avg_gpu_time_overall - avg_cpu_time_overall:.2f} seconds.")

# Run the analysis script and draw conclusions
if __name__ == "__main__":
    file_path = "results.json"
    analyze_results(file_path)  # Run the detailed analysis and graph generation
    draw_conclusions(file_path)  # Generate automatic conclusions from the data
