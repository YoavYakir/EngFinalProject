import time
import psutil
import threading
import json
import platform
import os

import pycuda.driver as cuda
import pycuda.autoinit
cuda.init()
if cuda:
    print("Cuda was successfully initialized")


def get_system_info():
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

if __name__ == "__main__":
    print(get_system_info())
