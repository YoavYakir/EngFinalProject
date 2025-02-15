import os
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import subprocess
import tensorflow as tf
import cupy as cp
import gc
import csv
def reset_gpu_memory():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.cuda.runtime.deviceSynchronize()

def setup_tensorflow_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)  # Prevent preallocation
        except RuntimeError as e:
            print(f"TensorFlow GPU setup error: {e}")

def find_available_gpu(threshold=50):
    threshold = 40
    num_gpus = cp.cuda.runtime.getDeviceCount()
    best_gpu = None
    best_free_mem = 0

    for gpu_index in range(num_gpus):
        print(f"Checking GPU {gpu_index}...")        
        try:
            with cp.cuda.Device(gpu_index):
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                free_percent = (free_mem / total_mem) * 100

                print(f"GPU {gpu_index}: {free_percent:.2f}% free memory available.")

                if free_percent > threshold and free_mem > best_free_mem:
                    best_gpu = gpu_index
                    best_free_mem = free_mem

        except cp.cuda.runtime.CUDARuntimeError as e:
            print(f"Error accessing GPU {gpu_index}: {e}")

    if best_gpu is not None:
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)  # Set environment variable
        print(f"Using GPU {best_gpu} with {best_free_mem / (1024 ** 2):.2f} MB free memory.")
        return best_gpu

    return None  # No available GPU found

def reset_tensorflow_gpu():
    """Reset TensorFlow GPU memory."""
    try:
        tf.keras.backend.clear_session()  # Clear TensorFlow session
        print("TensorFlow session cleared.")
    except RuntimeError as e:
        print(f"Error resetting TensorFlow GPU memory: {e}")

def clear_gpu_memory():
    """
    Clears GPU memory using CuPy.
    """
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    print("Cleared GPU memory.")

def setup_gpu(tensorflow=False):
    if tensorflow:
        setup_tensorflow_gpu()
        
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.cuda.runtime.deviceSynchronize()
    
    gpu_index = find_available_gpu()

    if gpu_index is None:
        raise RuntimeError("No available GPU found.")

    cp.cuda.Device(gpu_index).use()
    reset_gpu_memory()
    
    return gpu_index
