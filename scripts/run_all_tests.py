import os
import sys

# Clearing terminal
os.system("clear")


# Printing system path
print(f'\nSystem Path : {sys.path}\n{"*"*20}\n')

# Function to run all test scripts
def run_all_tests():
    # Clean Python
    # os.system("python -m EngFinalProject.tests.clean_python.run_all_tests")

    # # Numba
    # os.system("python -m EngFinalProject.tests.numba.run_all_tests")

    # # Cython
    # os.system("python -m EngFinalProject.tests.cython.run_all_tests")

    # # Pythran
    # os.system("python -m EngFinalProject.tests.pythran.run_all_tests")

    # # cupy
    os.system("python -m EngFinalProject.tests.cupy.run_all_tests")

if __name__ == "__main__":
    run_all_tests()
