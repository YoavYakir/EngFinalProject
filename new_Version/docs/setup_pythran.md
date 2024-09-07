### Note: Like Cython, Pythran focuses on CPU optimizations and does not provide GPU support.
# Setup Instructions for Pythran

Pythran is a Python-to-C++ compiler designed to accelerate Python code on the **CPU**.

### Steps to Compile and Run Pythran Code

1. Install Pythran:
```bash
pip install pythran
cd /tests/pythran
pythran test_pythran.py
python run_all_tests.py
```