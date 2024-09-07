import os

# Function to run all test scripts
def run_all_tests():
    # Clean Python
    os.system("python ../tests/clean_python/run_all_tests.py")

    # Numba
    os.system("python ../tests/numba/run_all_tests.py")

    # Cython
    os.system("python ../tests/cython/run_all_tests.py")

    # Pythran
    os.system("python ../tests/pythran/run_all_tests.py")

    # Dask + CuDF
    os.system("python ../tests/dask_cudf/run_all_tests.py")

if __name__ == "__main__":
    run_all_tests()
