import json
import os
import subprocess
import csv

def run_test_with_nsys(test_name, test_method, test_type, batch_size=None, filter_size=None, sample_rate=None, workers=None, epochs=None, learning_rate=None, model_size=None, dtype=None):
    """
    Run the test under Nsight profiling and return profiling and test's results.
    """    
    # Profiling setup
    output_dir = "./nsight_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Build dynamic nsys file name based on non-None parameters
    params = {
        "batch_size": batch_size,
        "filter_size": filter_size,
        "sample_rate": sample_rate,
        "workers": workers,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "model_size": model_size,
        "data_type": dtype
    }
    
    # Create dynamic parts of the file name and command
    file_parts = [f"{test_name}"]
    for param, value in params.items():
        if value is not None:
            file_parts.append(f"{param}_{value}")
    
    nsys_file = os.path.join(output_dir, "_".join(file_parts) + ".nsys-rep")

    # Copy the current environment variables
    env = os.environ.copy()

    # Ensure the PATH includes /usr/local/cuda-12/bin
    env["PATH"] = "/usr/local/cuda-12/bin:" + env["PATH"]

    # Start constructing the command based on non-None parameters
    cmd = [
        "nsys", "profile",
        "--output", nsys_file.replace(".nsys-rep", ""),
        "--trace=cuda,osrt,cublas",
        "--force-overwrite=true",
        "--show-output=true",
        "python3 -c ",
    ]

    cmd_run_test = f"\"from tests.{test_method}.{test_type} import *; result = run_single_test(test_name='{test_name}', test_function={test_name}, "
    # Append dynamic parameters to the command
    for param, value in params.items():
        if value is not None:
            if param == "data_type":
                cmd_run_test = cmd_run_test + (f"{param}='{value}'")
            else:
                cmd_run_test = cmd_run_test + (f'{param}={str(value)}')
            cmd_run_test = cmd_run_test + ","  # Add a comma after each parameter

    # Remove the last comma and space from the command, and close the print() statement properly
    cmd_run_test = cmd_run_test +  ");\""

    cmd.append(cmd_run_test)

    print(f"Running Nsight profiling: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if process.returncode != 0:
        print(process.stderr)
        print(process.stdout)
        print(f"Nsight Systems profiling failed:\n{process.stderr}")
        raise RuntimeError("Nsight profiling failed")

    # test_output = ""
    # with open('temp', 'r') as f:
    #     for line in f.readlines():
    #         test_output = test_output + line

    # os.remove('temp')

    # Extract JSON result from the child process output
    test_output = process.stdout + process.stderr
    test_result = extract_test_results(test_output)

    # Parse profiling output
    profiling_data = parse_nsys_output(nsys_file)
    print("*"*30)
    print(test_output)
    print("*"*30)
    return test_result, profiling_data

def extract_test_results(test_output):
    test_result = None
    for line in test_output:
        try:
            # Attempt to parse each line as JSON
            test_result = json.loads(line)
            break
        except json.JSONDecodeError:
            continue  # Ignore lines that are not JSON

    if test_result:
        return test_result


def generate_sqlite_file(nsys_file, sqlite_file):
    """
    Generate an SQLite file from the .nsys-rep file.
    """
    if not os.path.exists(sqlite_file):
        cmd = ["nsys", "stats", nsys_file]
        print(f"Generating SQLite file: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("Error generating SQLite file:")
            print(result.stderr)
            raise RuntimeError("Failed to generate SQLite file")


def generate_csv_file(report_name, csv_output_base, sqlite_file):
    """
    Generate a CSV file for a specific Nsight report.
    """
    cmd = [
        "nsys", "stats",
        "--report", report_name,
        "--format", "csv",
        "--output", csv_output_base,  # Specify the base name; Nsight adds the suffix automatically
        sqlite_file,
    ]
    print(f"Generating CSV for report '{report_name}': {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error generating CSV for report '{report_name}':")
        print(result.stderr)
        raise RuntimeError(f"Failed to generate CSV for report '{report_name}'")


def parse_csv_file(csv_file, fields):
    """
    Parse a CSV file and extract specific fields.

    Args:
        csv_file (str): Path to the CSV file.
        fields (list): List of field names to extract.

    Returns:
        list: List of dictionaries with the parsed data.
    """
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return []

    parsed_data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed_row = {field: row[field] for field in fields if field in row}
            parsed_data.append(parsed_row)
    return parsed_data


def cleanup_files(files):
    """
    Remove specified files from the filesystem.

    Args:
        files (list): List of file paths to remove.
    """
    for file in files:
        if os.path.exists(file):
            os.remove(file)


def parse_nsys_output(nsys_file, cleanup=True, top_k=5):
    """
    Parse the Nsight profiling output and extract relevant performance metrics.

    Args:
        nsys_file (str): Path to the Nsight .nsys-rep file.
        cleanup (bool): Whether to clean up intermediate files.
        top_k (int): Number of top kernels to retain based on 'Time (%)'.

    Returns:
        dict: Filtered and summarized performance data.
    """
    # Define paths for SQLite and CSV files
    sqlite_file = nsys_file.replace(".nsys-rep", ".sqlite")
    csv_base = nsys_file.replace(".nsys-rep", "")
    csv_outputs = {
        "kernels": f"{csv_base}_kernels",
        "memory": f"{csv_base}_memory",
    }

    # Step 1: Generate the SQLite file
    generate_sqlite_file(nsys_file, sqlite_file)

    # Step 2: Generate CSV files for specific reports
    reports = {
        "kernels": "cuda_gpu_kern_sum",
        "memory": "cuda_gpu_mem_time_sum",
    }

    for report_key, report_name in reports.items():
        generate_csv_file(report_name, csv_outputs[report_key], sqlite_file)

    # Add Nsight's suffix to expected filenames
    csv_outputs["kernels"] += "_cuda_gpu_kern_sum.csv"
    csv_outputs["memory"] += "_cuda_gpu_mem_time_sum.csv"

    # Step 3: Parse the CSV files
    raw_data = {
        "kernels": parse_csv_file(
            csv_outputs["kernels"],
            fields=["Time (%)", "Total Time (ns)", "Instances", "Avg (ns)", "Name"]
        ),
        "memory": parse_csv_file(
            csv_outputs["memory"],
            fields=["Time (%)", "Total Time (ns)", "Count", "Avg (ns)", "Operation"]
        )
    }

    # Step 4: Filter and summarize the data
    # Filter kernels by top 'Time (%)' or if the name contains 'sgemm'
    filtered_kernels = sorted(
        [
            kernel for kernel in raw_data["kernels"]
            if float(kernel["Time (%)"]) > 0.1 or "sgemm" in kernel["Name"]
        ],
        key=lambda x: float(x["Time (%)"]),
        reverse=True
    )[:top_k]  # Keep only the top_k kernels

    # Summarize memory operations
    memory_summary = {
        "Host-to-Device": next(
            (op for op in raw_data["memory"] if op["Operation"] == "[CUDA memcpy Host-to-Device]"), {}
        ),
        "Device-to-Host": next(
            (op for op in raw_data["memory"] if op["Operation"] == "[CUDA memcpy Device-to-Host]"), {}
        )
    }

    # Optional: Clean up intermediate files
    if cleanup:
        cleanup_files(list(csv_outputs.values()) + [sqlite_file, nsys_file])

    return {
        "top_kernels": filtered_kernels,
        "memory_summary": memory_summary
    }
