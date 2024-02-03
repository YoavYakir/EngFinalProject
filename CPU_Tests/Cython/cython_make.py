from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("stress_test2_cython.pyx")
)