# Setup Instructions for Dask + CuDF

Dask and CuDF are used to optimize **data preparation** and **GPU-accelerated computations**.

### Steps to Install and Run Dask + CuDF Tests

1. Install Dask and CuDF:
```bash
pip install dask dask-cuda cudf cupy
cd /tests/dask_cudf
python run_all_tests.py
```