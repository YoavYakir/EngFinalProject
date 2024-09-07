#pythran export iterative_test()
#pythran export matrix_multiplication(int, int)
#pythran export fibonacci_test(int)
#pythran export quicksort_test(int list)
#pythran export pca_test(int)
#pythran export svd_test(int)
#pythran export gpu_stress_test(int)

import numpy as np

def iterative_test():
    result = 0
    for i in range(1000000):
        result += i
    return result

def matrix_multiplication(rows, cols):
    A = np.random.rand(rows, cols)
    B = np.random.rand(rows, cols)
    return np.dot(A, B)

def fibonacci_test(n):
    if n <= 1:
        return n
    else:
        return fibonacci_test(n - 1) + fibonacci_test(n - 2)

def quicksort_test(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort_test(left) + middle + quicksort_test(right)

def pca_test(batch_size):
    data = np.random.rand(batch_size, batch_size)
    data_centered = data - np.mean(data, axis=0)
    cov_matrix = np.cov(data_centered.T)
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    sorted_idx = np.argsort(eigen_values)[::-1]
    return np.dot(data_centered, eigen_vectors[:, :2])

def svd_test(batch_size):
    data = np.random.rand(batch_size, batch_size)
    return np.linalg.svd(data, full_matrices=False)

def gpu_stress_test(batch_size):
    data = np.random.rand(batch_size, batch_size)
    return np.dot(data, data.T)
