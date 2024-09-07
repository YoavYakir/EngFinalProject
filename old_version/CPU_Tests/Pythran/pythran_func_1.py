#pythran export stress_test()
import numpy as np

def stress_test():
    for i in range(50):
        if not i % 10:
            print("Still running calculations")

        # Create a NumPy array with 10 million elements
        data_array = np.zeros(10 ** 8)

        # Vectorized operation to modify each element
        data_array = np.arange(10 ** 8) * 2

        # Sum the elements of the array
        result = np.sum(data_array)