import numpy as np
cimport numpy as np
cimport cython

# Declare the type of variables for speed optimization
@cython.boundscheck(False)  # Disable bounds-checking for speed
@cython.wraparound(False)   # Disable negative indexing for speed

# Matrix Multiplication using Cython
def matrix_multiplication(int rows, int cols):
    cdef np.ndarray[np.float64_t, ndim=2] A = np.random.rand(rows, cols)
    cdef np.ndarray[np.float64_t, ndim=2] B = np.random.rand(rows, cols)
    return np.dot(A, B)

# PCA using Cython
def pca_test(int batch_size):
    cdef np.ndarray[np.float64_t, ndim=2] data = np.random.rand(batch_size, batch_size)
    data_centered = data - np.mean(data, axis=0)
    cov_matrix = np.cov(data_centered.T)
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    sorted_idx = np.argsort(eigen_values)[::-1]
    return np.dot(data_centered, eigen_vectors[:, :2])  # Keep top 2 components

# SVD using Cython
def svd_test(int batch_size):
    cdef np.ndarray[np.float64_t, ndim=2] data = np.random.rand(batch_size, batch_size)
    return np.linalg.svd(data, full_matrices=False)

# Fibonacci Test using Cython
def fibonacci_test(int n):
    if n <= 1:
        return n
    else:
        return fibonacci_test(n - 1) + fibonacci_test(n - 2)

# Convolution using a sound filter of 1-second length
def convolution_test(int filter_size, int sample_rate=44100):
    # Generate a sound filter (1-second sine wave)
    t = np.linspace(0, 1, sample_rate)  # 1 second duration at 44.1kHz sampling rate
    frequency = 440  # A typical tone at 440 Hz (A4 note)
    sound_filter = np.sin(2 * np.pi * frequency * t)  # Generate sine wave

    # Reshape sound filter to be 2D (filter_size x filter_size)
    cdef np.ndarray[np.float64_t, ndim=2] filter_2d = np.reshape(sound_filter[:filter_size**2], (filter_size, filter_size))

    # Apply convolution
    cdef np.ndarray[np.float64_t, ndim=2] data = np.random.rand(filter_size, filter_size)
    cdef np.ndarray[np.float64_t, ndim=2] result = np.empty_like(data)

    for i in range(1, data.shape[0] - 1):
        for j in range(1, data.shape[1] - 1):
            result[i, j] = np.sum(data[i - 1:i + 2, j - 1:j + 2] * filter_2d)

    return result

# Quicksort using Cython
def quicksort_test(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort_test(left) + middle + quicksort_test(right)

# Simulated train test over MList in Cython
def train_test_mlist(int workers, int batch_size, double learning_rate):
    cdef int data_size = 10000
    cdef int input_size = 100
    cdef int output_size = 10
    cdef int iterations = data_size // batch_size
    
    # Initialize data and targets
    cdef np.ndarray[np.float64_t, ndim=2] data = np.random.rand(data_size, input_size)
    cdef np.ndarray[np.int_t, ndim=1] targets = np.random.randint(0, output_size, size=data_size)
    
    # Initialize random weights
    cdef np.ndarray[np.float64_t, ndim=2] weights = np.random.rand(input_size, output_size)
    
    for epoch in range(10):  # Let's assume 10 epochs
        for i in range(iterations // workers):
            batch = data[i * batch_size:(i + 1) * batch_size]
            target_batch = targets[i * batch_size:(i + 1) * batch_size]
            
            # Forward pass (linear + softmax)
            logits = np.dot(batch, weights)  # Linear transformation
            softmax = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)  # Softmax
            
            # Backward pass (compute gradients and update weights)
            error = softmax
            for j in range(batch_size):
                error[j, target_batch[j]] -= 1  # Cross-entropy gradient
            gradient = np.dot(batch.T, error) / batch_size
            weights -= learning_rate * gradient  # Update weights
    
    return weights  # Return weights after training
