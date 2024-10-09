import json
import matplotlib.pyplot as plt
from collections import defaultdict
from docx import Document
import os

def analyze_results(results_file, output_folder="analysis_output"):
    # Load the results JSON file
    with open(results_file, 'r') as f:
        data = json.load(f)

    # Separate results into CPU and GPU run types
    cpu_results = defaultdict(lambda: defaultdict(list))
    gpu_results = defaultdict(lambda: defaultdict(list))

    # Organize data into CPU and GPU structures
    for entry in data:
        test_name_parts = entry["Test Name"].split("_")
        test_method = test_name_parts[0]  # Method name
        test_name = "_".join(test_name_parts[1:])  # Test name after the method
        run_type = entry["Run Type"]

        # Store in the appropriate section based on run type
        if run_type == "CPU":
            cpu_results[test_name][test_method].append({
                "machine_specs": {
                    "OS": entry["OS"],
                    "CPU": entry["CPU"],
                    "GPU": entry["GPU"].split(",")[0]  # Only take the GPU model
                },
                "time": entry["Elapsed Time"],
                "cache_usage": entry.get("Average Cache Usage (MB)", 0),
                "ram_usage": entry.get("Average Memory Usage (%)", 0),
                "cpu_usage": entry.get("Average CPU Usage (%)", 0),
                "learn_params": {
                    "Workers": entry.get("Workers", None),
                    "Batch Size": entry.get("Batch Size", None),
                    "Learning rate": entry.get("Learning rate", None),
                    "Epochs": entry.get("Epochs", None),
                    "Filter Size": entry.get("Filter Size", None)
                }
            })
        elif run_type == "GPU":
            gpu_results[test_name][test_method].append({
                "machine_specs": {
                    "OS": entry["OS"],
                    "CPU": entry["CPU"],
                    "GPU": entry["GPU"].split(",")[0]  # Only take the GPU model
                },
                "time": entry["Elapsed Time"],
                "cache_usage": entry.get("Average Cache Usage (MB)", 0),
                "ram_usage": entry.get("Average Memory Usage (%)", 0),
                "cpu_usage": entry.get("Average CPU Usage (%)", 0),
                "learn_params": {
                    "Workers": entry.get("Workers", None),
                    "Batch Size": entry.get("Batch Size", None),
                    "Learning rate": entry.get("Learning rate", None),
                    "Epochs": entry.get("Epochs", None),
                    "Filter Size": entry.get("Filter Size", None)
                }
            })

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create a Word document for the report
    doc = Document()
    doc.add_heading("Analysis Report: CPU and GPU Tests", 0)

    # Helper function to plot and save graphs
    def plot_and_save_graph(x_values, y_values, x_label, y_label, title, filename):
        plt.figure()
        plt.bar(x_values, y_values, color='blue')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    # Function to generate graphs for both CPU and GPU sections
    def generate_section(test_results, section_name):
        doc.add_heading(section_name, level=1)

        for test_name, methods in test_results.items():
            doc.add_heading(f"Test: {test_name}", level=2)

            method_names = []
            elapsed_times = []
            cpu_usages = []
            cache_usages = []
            ram_usages = []

            for method, entries in methods.items():
                method_names.append(method)
                elapsed_times.append(entries[0]["time"])
                cpu_usages.append(entries[0]["cpu_usage"])
                cache_usages.append(entries[0]["cache_usage"])
                ram_usages.append(entries[0]["ram_usage"])

            # Graph: Elapsed Time for all methods
            graph_filename = os.path.join(output_folder, f"{section_name}_{test_name}_elapsed_time.png")
            plot_and_save_graph(method_names, elapsed_times, "Methods", "Elapsed Time (seconds)",
                                f"Elapsed Time for {test_name}", graph_filename)
            doc.add_heading(f"Elapsed Time for {test_name} (All Methods)", level=3)
            doc.add_picture(graph_filename)

            # Graph: CPU Usage for all methods
            graph_filename = os.path.join(output_folder, f"{section_name}_{test_name}_cpu_usage.png")
            plot_and_save_graph(method_names, cpu_usages, "Methods", "CPU Usage (%)",
                                f"CPU Usage for {test_name}", graph_filename)
            doc.add_heading(f"CPU Usage for {test_name} (All Methods)", level=3)
            doc.add_picture(graph_filename)

            # Graph: Cache Usage for all methods
            graph_filename = os.path.join(output_folder, f"{section_name}_{test_name}_cache_usage.png")
            plot_and_save_graph(method_names, cache_usages, "Methods", "Cache Usage (MB)",
                                f"Cache Usage for {test_name}", graph_filename)
            doc.add_heading(f"Cache Usage for {test_name} (All Methods)", level=3)
            doc.add_picture(graph_filename)

            # Graph: RAM Usage for all methods
            graph_filename = os.path.join(output_folder, f"{section_name}_{test_name}_ram_usage.png")
            plot_and_save_graph(method_names, ram_usages, "Methods", "RAM Usage (%)",
                                f"RAM Usage for {test_name}", graph_filename)
            doc.add_heading(f"RAM Usage for {test_name} (All Methods)", level=3)
            doc.add_picture(graph_filename)

            # Generate parameter-specific graphs for batch size, filter size, etc.
            if any(key in test_name for key in ["mlist", "matrix_multiplication", "pca", "svd"]):
                batch_size_values = [entry["learn_params"]["Batch Size"] for entry in entries if entry["learn_params"]["Batch Size"] is not None]
                if batch_size_values:
                    graph_filename = os.path.join(output_folder, f"{section_name}_{test_name}_batch_size_elapsed_time.png")
                    plot_and_save_graph(batch_size_values, elapsed_times, "Batch Size", "Elapsed Time (seconds)",
                                        f"Elapsed Time vs Batch Size for {test_name}", graph_filename)
                    doc.add_heading(f"Elapsed Time vs Batch Size for {test_name}", level=3)
                    doc.add_picture(graph_filename)

            if any(key in test_name for key in ["convolution", "fft"]):
                filter_size_values = [entry["learn_params"]["Filter Size"] for entry in entries if entry["learn_params"]["Filter Size"] is not None]
                if filter_size_values:
                    graph_filename = os.path.join(output_folder, f"{section_name}_{test_name}_filter_size_elapsed_time.png")
                    plot_and_save_graph(filter_size_values, elapsed_times, "Filter Size", "Elapsed Time (seconds)",
                                        f"Elapsed Time vs Filter Size for {test_name}", graph_filename)
                    doc.add_heading(f"Elapsed Time vs Filter Size for {test_name}", level=3)
                    doc.add_picture(graph_filename)

            # For mlist: Workers, Learning Rate, Epochs specific graphs
            if "mlist" in test_name:
                workers_values = [entry["learn_params"]["Workers"] for entry in entries if entry["learn_params"]["Workers"] is not None]
                learning_rate_values = [entry["learn_params"]["Learning rate"] for entry in entries if entry["learn_params"]["Learning rate"] is not None]
                epochs_values = [entry["learn_params"]["Epochs"] for entry in entries if entry["learn_params"]["Epochs"] is not None]

                # Workers graph
                if workers_values:
                    graph_filename = os.path.join(output_folder, f"{section_name}_{test_name}_workers_elapsed_time.png")
                    plot_and_save_graph(workers_values, elapsed_times, "Workers", "Elapsed Time (seconds)",
                                        f"Elapsed Time vs Workers for {test_name}", graph_filename)
                    doc.add_heading(f"Elapsed Time vs Workers for {test_name}", level=3)
                    doc.add_picture(graph_filename)

                # Learning Rate graph
                if learning_rate_values:
                    graph_filename = os.path.join(output_folder, f"{section_name}_{test_name}_learning_rate_elapsed_time.png")
                    plot_and_save_graph(learning_rate_values, elapsed_times, "Learning Rate", "Elapsed Time (seconds)",
                                        f"Elapsed Time vs Learning Rate for {test_name}", graph_filename)
                    doc.add_heading(f"Elapsed Time vs Learning Rate for {test_name}", level=3)
                    doc.add_picture(graph_filename)

                # Epochs graph
                if epochs_values:
                    graph_filename = os.path.join(output_folder, f"{section_name}_{test_name}_epochs_elapsed_time.png")
                    plot_and_save_graph(epochs_values, elapsed_times, "Epochs", "Elapsed Time (seconds)",
                                        f"Elapsed Time vs Epochs for {test_name}", graph_filename)
                    doc.add_heading(f"Elapsed Time vs Epochs for {test_name}", level=3)
                    doc.add_picture(graph_filename)

    # Generate CPU section
    generate_section(cpu_results, "CPU")

    # Generate GPU section
    generate_section(gpu_results, "GPU")

    # Save the document
    output_docx = os.path.join(output_folder, "analysis_report.docx")
    doc.save(output_docx)

    print(f"Analysis complete! Report saved to {output_docx}")

os.system("rm -rf EngFinalProject/analysis_output")
analyze_results("EngFinalProject/results/results.json", "EngFinalProject/analysis_output")
