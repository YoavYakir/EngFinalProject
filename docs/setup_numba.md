### This script will run tests on both the CPU and GPU using Numba's @jit and cuda.jit.
### For GPU profiling, ensure that NVIDIA Nsight Systems is installed, and the generated .nsys-rep files can be analyzed using Nsight.

# Setup Instructions for Numba

Numba is used to optimize Python code by compiling it to machine code and enabling **GPU acceleration** with CUDA.

### Steps to Install and Run Numba Tests

1. **Install the CUDA Toolkit**:
   - Follow the official **NVIDIA CUDA Toolkit** installation guide for your operating system: https://developer.nvidia.com/cuda-downloads.
   - Make sure the CUDA Toolkit is properly installed and added to your system's path, run ```python -c from numba import cuda; print(cuda.gpus)```

2. **Install Numba and CUDA libraries:**
```bash
pip install numba
cd /tests/numba
python run_all_tests.py
```
