import time
import psutil
import threading
import json
import platform

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    cuda.init()
except ImportError:
    cuda = None

class ResourceMonitor:
    def __init__(self):
        self.cpu_usage = []
        self.memory_usage = []
        self.running = False

    def start_monitoring(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def _monitor(self):
        while self.running:
            self.cpu_usage.append(psutil.cpu_percent(interval=1))
            self.memory_usage.append(psutil.virtual_memory().percent)
            time.sleep(1)

    def stop_monitoring(self):
        self.running = False
        self.thread.join()

    def get_average_usage(self):
        return {
            "Average CPU Usage (%)": sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
            "Average Memory Usage (%)": sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
        }

    def get_system_info(self):
        """ Returns the CPU and GPU information of the system. """
        system_info = {
            "OS": platform.system(),
            "OS Version": platform.version(),
            "CPU": platform.processor(),
        }

        if cuda:
            device = cuda.Device(0)
            gpu_info = f"{device.name()}, {device.compute_capability()}"
        else:
            gpu_info = "No GPU"

        system_info["GPU"] = gpu_info
        return system_info

    def save_results(self, system_stats, test_name, result_file_path, workers=1, batch_size=0):
        # Add test name, workers, and batch size to the results
        system_stats["Test Name"] = test_name
        system_stats["Workers"] = workers
        system_stats["Batch Size"] = batch_size

        # Load existing results if the file exists
        try:
            with open(result_file_path, "r") as file:
                results = json.load(file)
        except FileNotFoundError:
            results = []

        # Check if there's an existing result with the same OS, CPU, GPU, Run Type, Workers, and Batch Size
        existing_result = next((r for r in results if
                                r["Test Name"] == test_name and
                                r["OS"] == system_stats["OS"] and
                                r["CPU"] == system_stats["CPU"] and
                                r["GPU"] == system_stats["GPU"] and
                                r["Run Type"] == system_stats["Run Type"] and
                                r["Workers"] == workers and
                                r["Batch Size"] == batch_size), None)

        # If the result already exists, overwrite it; otherwise, append the new result
        if existing_result:
            results = [r if r != existing_result else system_stats for r in results]
        else:
            results.append(system_stats)

        # Save updated results back to the JSON file
        with open(result_file_path, "w") as file:
            json.dump(results, file, indent=4)
