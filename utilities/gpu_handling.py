import os
import subprocess
import tensorflow as tf
import pycuda.driver as cuda
import pycuda.autoinit
import cupy as cp
import gc
import csv

def find_available_gpu(threshold=50):
    """
    Finds a GPU with sufficient memory and ensures PyCUDA contexts are released.
    """
    num_gpus = cuda.Device.count()
    for gpu_index in range(num_gpus):
        device = cuda.Device(gpu_index)
        context = device.make_context()  # Create a context for the device
        try:
            free_mem, total_mem = cuda.mem_get_info()
            used_mem_percent = (1 - free_mem / total_mem) * 100
            if used_mem_percent < threshold:
                print(f"GPU {gpu_index}: {100 - used_mem_percent:.2f}% free memory available.")
                return gpu_index
        finally:
            context.pop()  # Always pop the context

    return None

def setup_tensorflow_gpu(gpu_index):
    """Ensure TensorFlow initializes only the target GPU."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
            print(f"Using TensorFlow on GPU {gpu_index}")
        except RuntimeError as e:
            print(f"TensorFlow GPU setup error: {e}")

def reset_tensorflow_gpu():
    """Reset TensorFlow GPU memory."""
    try:
        tf.keras.backend.clear_session()  # Clear TensorFlow session
        print("TensorFlow session cleared.")
    except RuntimeError as e:
        print(f"Error resetting TensorFlow GPU memory: {e}")

def clear_gpu_memory(gpu_index):
    """
    Clears GPU memory and ensures the context is popped.
    """
    try:
        device = cuda.Device(gpu_index)
        context = device.make_context()  # Create a context
        cp.get_default_memory_pool().free_all_blocks()  # Clear CuPy memory pool
        cp.get_default_pinned_memory_pool().free_all_blocks()  # Clear pinned memory pool
        context.synchronize()  # Synchronize device
        print(f"GPU memory cleared for GPU {gpu_index}")
    except cuda.CudaError as e:
        print(f"Error during GPU memory clearing: {e}")
    finally:
        context.pop()  # Pop the context at the end

def setup_gpu(tensorflow=False):
    # Clear memory before allocating new arrays
    # num_gpus = cuda.Device.count()
    # for gpu_index in range(num_gpus):
    #     clear_gpu_memory(gpu_index)
    # reset_tensorflow_gpu()

    gpu_index = find_available_gpu()

    if (gpu_index is None):
        raise("Can't run without GPU")

    if tensorflow:
        setup_tensorflow_gpu(gpu_index)
    
    return gpu_index