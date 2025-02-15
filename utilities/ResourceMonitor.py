import time
import psutil
import threading
import json
import platform
import os
import utilities.gpu_handling as gpu_handling
import cupy as cp  # Use CuPy for GPU handling


class ResourceMonitor:
    def __init__(self, tensorflow=False):
        self.cpu_usage = []
        self.memory_usage_percent = []
        self.memory_usage_mb = []
        self.cache_usage = []
        self.gpu_usage = []
        self.running = False
        self.gpu_index = gpu_handling.setup_gpu(tensorflow)
        self.gpu_init_usage = 0
        self.cpu_init_usage = psutil.cpu_percent(interval=0)
        self.max_samples = 100  # Maximum number of samples to include in the final report

        if self.gpu_index is not None:
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            self.gpu_init_usage = (1 - free_mem / total_mem) * 100  # Initial GPU usage

    def start_monitoring(self):
        """Start monitoring resource usage in a separate thread."""
        print(f"GPU FIRST init state - {self.gpu_init_usage}")
        if self.gpu_index is None:
            raise RuntimeError("There isn't a GPU available.")
        
        self.clear_cache()

        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def _monitor(self):
        """Collect resource usage data at regular intervals."""
        while self.running:
            # CPU usage
            usage = psutil.cpu_percent(interval=0)
            delta_usage = usage - self.cpu_init_usage
            if delta_usage >= 0:
                self.cpu_usage.append(usage)
            else:
                self.cpu_init_usage = usage
            
            # Memory usage
            mem_info = psutil.virtual_memory()
            self.memory_usage_percent.append(mem_info.percent)
            self.memory_usage_mb.append(mem_info.used / (1024 ** 2))  # Convert to MB

            # Cache usage
            self.cache_usage.append(self._get_cache_usage())

            # GPU usage
            if self.gpu_index is not None:
                self.gpu_usage.append(self._get_gpu_usage())

            time.sleep(0.00001)  # Sampling interval

    def stop_monitoring(self):
        """Stop monitoring resource usage."""
        gpu_handling.reset_gpu_memory()
        self.running = False
        self.thread.join()

    def get_average_usage(self):
        """
        Compute and return the average resource usage, along with sampled values.
        """
        cpu_avg = self._calculate_average(self.cpu_usage)
        memory_percent_avg = self._calculate_average(self.memory_usage_percent)
        memory_mb_avg = self._calculate_average(self.memory_usage_mb)
        cache_avg = self._calculate_average(self.cache_usage)
        gpu_avg = self._calculate_average(self.gpu_usage) if self.gpu_index is not None else "No GPU"

        return {
            "Average CPU Usage (%)": cpu_avg,
            "Average Memory Usage (%)": memory_percent_avg,
            "Average Memory Usage (MB)": memory_mb_avg,
            "Average Cache Usage (MB)": cache_avg,
            "Average GPU Usage (%)": gpu_avg,
            "CPU Samples": self._downsample(self.cpu_usage),
            "Memory Percent Samples": self._downsample(self.memory_usage_percent),
            "Memory MB Samples": self._downsample(self.memory_usage_mb),
            "Cache Samples": self._downsample(self.cache_usage),
            "GPU Samples": self._downsample(self.gpu_usage),
        }

    @staticmethod
    def _calculate_average(data_list):
        """Helper method to calculate the average of a list."""
        return sum(data_list) / len(data_list) if data_list else 0

    def _downsample(self, data_list):
        """Downsample the data list to a maximum of self.max_samples."""
        if len(data_list) <= self.max_samples:
            return data_list
        step = len(data_list) // (self.max_samples - 1)
        return [data_list[i] for i in range(0, len(data_list), step)] + [data_list[-1]]

    def _get_cache_usage(self):
        """Returns the cache usage on Linux by reading /proc/meminfo."""
        if platform.system() == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("Cached:"):
                        return int(line.split()[1]) // 1024  # Convert KB to MB
        return 0

    def _get_gpu_usage(self):
        """Returns GPU utilization percentage using CuPy."""
        if self.gpu_index is not None:
            # with cp.cuda.Device(self.gpu_index):
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            current_usage = (1 - free_mem / total_mem) * 100  # GPU usage percentage
            delta_usage = current_usage - self.gpu_init_usage
            # print(f"current usage : {current_usage}, init state : {self.gpu_init_usage}, delta : {delta_usage}")
            if delta_usage >= 0:
                return delta_usage
            else:
                self.gpu_init_usage = current_usage
        return current_usage

    def get_system_info(self):
        """Returns the CPU and GPU information of the system."""
        system_info = {
            "OS": platform.system(),
            "OS Version": platform.version(),
            "CPU": platform.processor(),
        }

        if self.gpu_index is not None:
            # device_info = cp.cuda.Device(self.gpu_index).attributes
            # gpu_info = f"GPU {self.gpu_index} (Compute Capability: {device_info.get(75, 'Unknown')})"
            gpu_info = "GTX2080"
        else:
            gpu_info = "No GPU"

        system_info["GPU"] = gpu_info
        return system_info

    def save_results(self, system_stats, test_name, result_file_path):
        system_stats["Test Name"] = test_name

        # Specify keys to format compactly
        compact_keys = ["CPU Samples", "Memory Percent Samples", "Memory MB Samples", "Cache Samples", "GPU Samples"]

        # Start building the JSON string manually
        result = "{\n"
        for key, value in system_stats.items():
            if key in compact_keys:
                result += f'    "{key}": {json.dumps(value, separators=(",", ":"))},\n'
            else:
                result += f'    "{key}": {json.dumps(value, indent=4)},\n'
        result = result.rstrip(",\n") + "\n}"

        # Write the final formatted string to the file
        with open(result_file_path, "a") as file:
            file.write(result)

    @staticmethod
    def fix_json_file(file_path):
        with open(file_path, 'r') as file:
            data = file.read()

        # Replace '}{' with '},{' and wrap with brackets to make it a valid JSON array
        fixed_data = '[' + data.replace('}{', '},{') + ']'

        # Verify if the fixed data is valid JSON
        json.loads(fixed_data)

        # Save the corrected JSON to a new file
        with open(file_path, 'w') as file:
            file.write(fixed_data)

    def get_gpu_name(self):
        """Returns a standardized GPU name for compatibility."""
        if self.gpu_index is not None:
            # device_info = cp.cuda.Device(self.gpu_index).attributes
            # gpu_info = f"GPU {self.gpu_index} (Compute Capability: {device_info.get(75, 'Unknown')})"
            gpu_info = "2080"
            if "1080" in gpu_info:
                return "gtx1080"
            elif "2080" in gpu_info:
                return "gtx2080"
            elif "a100" in gpu_info:
                return "a100"
            else:
                raise RuntimeError(f"GPU not supported - {gpu_info}")
        return "No GPU"

    def clear_cache(self):
        """ Clears cache (page cache, dentries, and inodes) on Linux """
        if platform.system() == "Linux":
            os.system("echo 1 > /proc/sys/vm/drop_caches")  # Clear cache
            print("Cache cleared.")
