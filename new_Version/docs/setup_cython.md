# Setup Instructions for Cython
### Note: Cython is optimized for CPU performance and does not provide GPU acceleration.

Cython is used to optimize Python code for **CPU performance**. It translates Python code into C and compiles it into an extension module.

### Steps to Compile and Run Cython Code

1. Install Cython:
```bash
pip install cython
cd /tests/cython
python setup.py build_ext --inplace
python run_all_tests.py
```
