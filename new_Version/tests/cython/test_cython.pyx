# cython: boundscheck=False, wraparound=False
import numpy as np
cimport cython
from libc.math cimport pow

@cython.boundscheck(False)
@cython.wraparound(False)
def iterative_test():
    cdef int i, result = 0
    for i in range(1000000):
        result += i
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def matrix_multiplication(int rows, int cols):
    cdef np.ndarray[np.float_t, ndim=2] A = np.random.rand(rows, cols)
    cdef np.ndarray[np.float_t, ndim=2] B = np.random.rand(rows, cols)
    return np.dot(A, B)

@cython.boundscheck(False)
@cython.wraparound(False)
def fibonacci_test(int n):
    if n <= 1:
        return n
    else:
        return fibonacci_test(n - 1) + fibonacci_test(n - 2)

@cython.boundscheck(False)
@cython.wraparound(False)
def quicksort_test(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort_test(left) + middle + quicksort_test(right)

@cython.boundscheck(False)
@cython.wraparound(False)
def matrix_multiplication(int rows, int cols):
    cdef np.ndarray[np.float_t, ndim=2] A = np.random.rand(rows, cols)
    cdef np.ndarray[np.float_t, ndim=2] B = np.random.rand(rows, cols)
    return np.dot(A, B)

@cython.boundscheck(False)
@cython.wraparound(False)
def pca_test(int batch_size, int n_components=2):
    cdef np.ndarray[np.float_t, ndim=2] data = np.random.rand(batch_size, batch_size)
    data_centered = data - np.mean(data, axis=0)
    cov_matrix = np.cov(data_centered.T)
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    sorted_idx = np.argsort(eigen_values)[::-1]
    return np.dot(data_centered, eigen_vectors[:, :n_components])

@cython.boundscheck(False)
@cython.wraparound(False)
def svd_test(int batch_size):
    cdef np.ndarray[np.float_t, ndim=2] data = np.random.rand(batch_size, batch_size)
    return np.linalg.svd(data, full_matrices=False)

@cython.boundscheck(False)
@cython.wraparound(False)
def gpu_stress_test(int batch_size):
    cdef np.ndarray[np.float_t, ndim=2] data = np.random.rand(batch_size, batch_size)
    return np.dot(data, data.T)