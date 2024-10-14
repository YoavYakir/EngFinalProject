import os
import sys

# Clearing terminal
os.system("clear")

# Printing system path
print(f'\nSystem Path : {sys.path}\n{"*"*20}\n')

# Function to run all test scripts
def run_all_tests():
    # Clean Python
    # os.system("python3 -m tests.clean_python.run_cpu_tests")
    os.system("python3 -m tests.clean_python.run_gpu_tests")
    # os.system("python3 -m tests.clean_python.run_cpu_tests")

    # Numba
    # try:
    # os.system("python -m EngFinalProject.tests.numba.run_all_tests")
    # except Exception as e:
    #     print("Continue to next method")

    # # Cython
    # os.system("python -m EngFinalProject.tests.cython.run_all_tests")

    # Pythran
    # try:
    #     os.system("python -m EngFinalProject.tests.pythran.run_all_tests")
    # except Exception as e:
    #     print("Continue to next method")

    # # # cupy
    # os.environ["NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY"] = "1"
    # os.system("python3.7 -m EngFinalProject.tests.dask.run_all_tests")

if __name__ == "__main__":
    run_all_tests()
