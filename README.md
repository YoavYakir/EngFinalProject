# CPU/GPU Optimization and Analysis
## Overview

This project focuses on enhancing the performance of Python programs using various techniques to optimize both **CPU** and **GPU** computations. The process involved experimenting with several methods and tools, testing different configurations (such as batch sizes and workers), and analyzing the results to draw meaningful conclusions.

### Project Goals

1. **Optimize CPU and GPU Performance**: Explore multiple methods such as **Pythran**, **Numba**, **Cython**, and **Dask + CuDF** to speed up Python code.
2. **Analyze the Effect of Batch Sizes and Workers**: Investigate how changing the batch sizes and the number of workers affects **CPU** and **GPU** performance.
3. **Performance Analysis and Visualization**: Generate graphs to visualize the performance impact of each method and configuration, and draw conclusions based on the data.
4. **Automated Prediction**: Build a prediction model to estimate the time required to prepare batches of different sizes.

---

## Methods Used

### 1. **Initial Exploration of Methods**:

We started by experimenting with four key methods to improve CPU and GPU performance:

- **Clean Python**: Standard Python code, without any optimizations.
- **Pythran**: A Python-to-C++ compiler that focuses on CPU optimizations but lacks GPU support.
- **Numba**: A Just-In-Time (JIT) compiler that provides both **CPU** and **GPU** support through CUDA.
- **Cython**: A tool that converts Python-like code into C for improved CPU performance.
- **Dask + CuDF**: A Python library that facilitates distributed computation on both CPUs and GPUs, offering scalability and performance improvements for larger datasets.

### 2. **Challenges Faced**:

After testing the above methods, we found that:
- **Pythran** and **Cython** are only effective for **CPU optimization** and don’t provide GPU acceleration.
- **Numba** and **Dask + CuDF** were the only methods that offered **GPU acceleration** via CUDA.

To demonstrate this, we still included **Pythran** and **Cython** to highlight **CPU optimizations**, but clarified that they don’t enhance GPU performance.

### 3. **Worker and Batch Size Analysis**:

We conducted tests on varying **batch sizes** and **numbers of workers** to understand how these variables impact performance. We wanted to know if increasing the **number of workers** would result in a proportional increase in performance when using **GPU**, and how changing the **batch size** affects both **CPU** and **GPU** computation times.

### 4. **Data Analysis and Predictions**:

We generated graphs to compare the results of each test:
- **CPU vs GPU** performance for different methods and configurations.
- The impact of **batch size** and **workers** on **GPU performance**.

Finally, we built a **prediction model** to estimate the time required to process batches of different sizes using various methods.

---

## Project Structure

Here's an overview of the project structure and a description of each key component:
```
EngFinalProject/
    ├── docs/
    │   ├── setup_cython.md
    │   ├── setup_dask_cudf.md
    │   ├── setup_numba.md
    │   └── setup_pythran.md
    ├── results/
    │   ├── conclusions_analysis_script.py
    │   └── results.json
    ├── scripts/
    │   ├── analyze_results.py
    │   ├── run_all_tests.py
    │   └── utility_functions.py
    ├── tests/
    │   ├── clean_python/
    │   │   └── run_all_tests.py
    │   ├── cython/
    │   │   ├── run_all_tests.py
    │   │   ├── setup.py
    │   │   └── test_cython.pyx
    │   ├── dask_cudf/
    │   │   └── run_all_tests.py
    │   ├── numba/
    │   │   └── run_all_tests.py
    │   └── pythran/
    │       ├── run_all_tests.py
    │       └── test_pythran.py
    └── README.md
```

### Key Files:

1. **docs/**: This folder contains setup instructions for each method.
   - `setup_cython.md`: Instructions to compile and run Cython tests.
   - `setup_dask_cudf.md`: Steps to install and run Dask + CuDF for GPU acceleration.
   - `setup_numba.md`: Guidelines for installing CUDA and running Numba on GPU.
   - `setup_pythran.md`: Steps for using Pythran to optimize CPU performance.

2. **results/**: Stores the analysis results and conclusion scripts.
   - `results.json`: JSON file where all test results are saved.
   - `conclusions_analysis_script.py`: Script that generates conclusions from the test results.

3. **scripts/**: Contains utility functions and the main analysis scripts.
   - `analyze_results.py`: Analyzes the results from `results.json` and generates graphs comparing CPU vs GPU performance across methods, batch sizes, and workers.
   - `run_all_tests.py`: Executes all the tests across different methods.
   - `utility_functions.py`: Provides functions for monitoring system resources (CPU, GPU, memory) and saving results to JSON.

4. **tests/**: Contains the actual test scripts for each method.
   - `clean_python/run_all_tests.py`: Runs tests using unoptimized Python code.
   - `cython/run_all_tests.py`: Runs tests using Cython for CPU optimization.
   - `dask_cudf/run_all_tests.py`: Runs tests using Dask + CuDF for distributed GPU computation.
   - `numba/run_all_tests.py`: Runs tests using Numba for both CPU and GPU.
   - `pythran/run_all_tests.py`: Runs tests using Pythran for CPU optimization.

---

## Key Tests

The following key tests were conducted across all methods:

1. **Matrix Multiplication Test**: Evaluates the performance of matrix operations on both **CPU** and **GPU**.
2. **Iterative Test**: Measures the performance of a simple iterative computation.
3. **Fibonacci Test**: Recursively calculates Fibonacci numbers.
4. **Quicksort Test**: Benchmarks the performance of the quicksort algorithm.
5. **PCA (Principal Component Analysis) Test**: Measures performance on dimensionality reduction tasks (implemented for methods that support matrix operations).
6. **SVD (Singular Value Decomposition) Test**: Tests the performance of matrix factorization operations.
7. **GPU Stress Test**: Conducts intensive matrix operations to push the **GPU** to its limits.

---

## Performance Analysis

We gathered performance data from all tests and compared the results in several ways:

- **Batch Size Impact**: We observed that increasing the batch size generally improves **GPU performance**, but the degree of improvement depends on the method used.
  
- **Effect of Workers on GPU**: Increasing the number of workers can improve **GPU performance**, but there are diminishing returns after a certain point.

- **CPU vs GPU**: While **Numba** and **Dask + CuDF** showed significant GPU speed-ups, methods like **Pythran** and **Cython** were more suitable for **CPU optimization**.

---

## Conclusions

1. **GPU is Generally Faster for Large Batch Sizes**: We observed that for larger batch sizes (e.g., 1000+), **GPU** consistently outperformed **CPU**, especially when using **Numba** and **Dask + CuDF**.
   
2. **Workers Impact Performance but With Diminishing Returns**: Increasing the number of workers improved **GPU** performance, but beyond a certain threshold (e.g., 4 workers), the performance gains were marginal.

3. **Method-Specific Optimizations**:
   - **Numba** and **Dask + CuDF**: Best suited for **GPU** acceleration and large-scale distributed computation.
   - **Cython** and **Pythran**: Focused solely on **CPU optimizations** and don’t benefit from GPU resources.
   
4. **Batch Size is Critical**: Larger batch sizes generally reduce computation time, particularly when leveraging **GPU** resources.

5. **Prediction of Batch Preparation Time**: Based on the results, we can now predict that for a batch size of 500, **matrix multiplication** using **Numba** will take approximately 1.2 seconds on the **GPU**.

---

## How to Run the Project

1. Follow the setup instructions in the `docs/` folder for each method.
2. Run the **tests** from the `tests/` folder using the corresponding `run_all_tests.py` file for each method.
3. Analyze the results using the `analyze_results.py` script in the `scripts/` folder.
4. Use the `conclusions_analysis_script.py` in the `results/` folder to automatically draw conclusions from the results.

---
