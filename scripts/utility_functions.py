import time
import psutil
import threading
import json
import platform
import os

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    print("Cuda was successfully initialized")
except ImportError:
    print("Error initializing GPU, cuda = None")
    cuda = None

class ResourceMonitor:
    def __init__(self):
        self.cpu_usage = []
        self.memory_usage_percent = []
        self.memory_usage_mb = []
        self.cache_usage = []
        self.running = False

    def start_monitoring(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def _monitor(self):
        while self.running:
            self.cpu_usage.append(psutil.cpu_percent(interval=1))
            mem_info = psutil.virtual_memory()
            self.memory_usage_percent.append(mem_info.percent)
            self.memory_usage_mb.append(mem_info.used / (1024 ** 2))  # Convert memory usage to MB
            self.cache_usage.append(self._get_cache_usage())
            time.sleep(1)

    def stop_monitoring(self):
        self.running = False
        self.thread.join()

    def get_average_usage(self):
        return {
            "Average CPU Usage (%)": sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
            "Average Memory Usage (%)": sum(self.memory_usage_percent) / len(self.memory_usage_percent) if self.memory_usage_percent else 0,
            "Average Memory Usage (MB)": sum(self.memory_usage_mb) / len(self.memory_usage_mb) if self.memory_usage_mb else 0,
            "Average Cache Usage (MB)": sum(self.cache_usage) / len(self.cache_usage) if self.cache_usage else 0
        }

    def _get_cache_usage(self):
        """ Returns the cache usage on Linux by reading /proc/meminfo """
        if platform.system() == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("Cached:"):
                        # Cache usage in KB, convert to MB
                        return int(line.split()[1]) // 1024
        return 0

    def clear_cache(self):
        """ Clears cache (page cache, dentries, and inodes) on Linux """
        if platform.system() == "Linux":
            os.system("sync")  # Ensure filesystem buffers are flushed
            os.system("echo 3 | tee /proc/sys/vm/drop_caches")  # Clear cache
            print("Cache cleared.")

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

    def save_results(self, system_stats, test_name, result_file_path, workers=1, batch_size=0, epochs=10):
        # Add test name, workers, batch size, and epochs to the results
        system_stats["Test Name"] = test_name
        system_stats["Workers"] = workers
        system_stats["Batch Size"] = batch_size
        system_stats["Epochs"] = epochs  # Add epochs to system stats

        # Load existing results, handle empty or invalid JSON batch_sizefiles
        try:
            with open(result_file_path, "r") as file:
                results = json.load(file)  # Load previous results
        except (FileNotFoundError, json.JSONDecodeError):
            results = []  # Initialize with an empty list if file is not found or invalid

        # Check if there's an existing result with the same OS, CPU, GPU, Run Type, Workers, Batch Size, and Epochs
        existing_result = next((r for r in results if
                                r["Test Name"] == test_name and
                                r["OS"] == system_stats["OS"] and
                                r["CPU"] == system_stats["CPU"] and
                                r["GPU"] == system_stats["GPU"] and
                                r["Run Type"] == system_stats["Run Type"] and
                                r["Workers"] == workers and
                                r["Batch Size"] == batch_size and
                                r["Matrix Size"] == system_stats["Matrix Size"] and
                                r["Filter Size"] == system_stats["Filter Size"] and
                                r["Sample Rate"] == system_stats["Sample Rate"] and
                                r["Data Type"] == system_stats["Data Type"] and
                                r["FFT Length"] == system_stats["FFT Length"] and
                                r["Learning rate"] == system_stats["Learning rate"] and
                                r["Epochs"] == epochs), None)

        # If the result already exists, overwrite it; otherwise, append the new result
        if existing_result:
            # Update existing result
            results = [r if r != existing_result else system_stats for r in results]
        else:
            # Append the new result
            results.append(system_stats)

        # Save updated results back to the JSON file
        with open(result_file_path, "w") as file:
            json.dump(results, file, indent=4)  # Write all results, including the new one