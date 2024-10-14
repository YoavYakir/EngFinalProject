import numpy as np
import time
import platform
from numba import cuda


# Numba CUDA kernel for training MList on GPU
@cuda.jit
def train_test_mlist(data, targets, weights, batch_size, learning_rate):
    i = cuda.grid(1)
    if i < batch_size:
        input_size = data.shape[1]
        output_size = weights.shape[1]

        # Forward pass: compute logits
        logits = cuda.local.array(10, dtype=np.float32)  # Assuming 10 output classes
        for j in range(output_size):
            logits[j] = 0.0
            for k in range(input_size):
                logits[j] += data[i, k] * weights[k, j]

        # Softmax activation
        sum_exp = 0.0
        for j in range(output_size):
            logits[j] = np.math.exp(logits[j])
            sum_exp += logits[j]

        for j in range(output_size):
            logits[j] /= sum_exp

        # Backward pass: Cross-entropy loss gradient and weight update
        for j in range(output_size):
            gradient = logits[j]
            if j == targets[i]:
                gradient -= 1  # Cross-entropy gradient adjustment

            # Update weights
            for k in range(input_size):
                weights[k, j] -= learning_rate * gradient * data[i, k]


# Function to run MList training on GPU
def run_test(test_name, test_function_gpu, on_gpu=True, workers=1, batch_size=100, epochs=110, learning_rate=0.01):
    # Simulate dataset and parameters
    input_size = 100  # Assuming 100 input features
    output_size = 10  # Assuming 10 output classes
    data = np.random.rand(batch_size, input_size).astype(np.float32)
    targets = np.random.randint(0, output_size, size=batch_size).astype(np.int32)
    weights = np.random.rand(input_size, output_size).astype(np.float32)

    # Transfer data to GPU
    data_gpu = cuda.to_device(data)
    targets_gpu = cuda.to_device(targets)
    weights_gpu = cuda.to_device(weights)

    # Launch the kernel
    threads_per_block = 128
    blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block

    # Run multiple epochs
    for epoch in range(epochs):
        test_function_gpu[blocks_per_grid, threads_per_block](data_gpu, targets_gpu, weights_gpu, batch_size, learning_rate)

    # Copy results back to CPU for analysis if needed
    weights_result = weights_gpu.copy_to_host()

    # Log system info and performance
    print(f'Completed {test_name} on GPU with batch_size={batch_size}, workers={workers}')


# Function to run all tests (MLIST only in this case)
def run_all_tests():
    tests = {
        "train_test_mlist": train_test_mlist
    }

    # Run the MList test on GPU with varying batch sizes and workers
    for test_name, test_function_gpu in tests.items():
        for batch_size in [256, 512, 1024]:  # Define different batch sizes
            for workers in [1, 2, 4, 8]:  # Define different worker counts
                print(f'Running {test_name} on GPU with batch size {batch_size} and {workers} workers...')
                run_test(test_name, test_function_gpu, on_gpu=True, workers=workers, batch_size=batch_size)


if __name__ == "__main__":
    run_all_tests()
