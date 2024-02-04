from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("stress_test4_cython.pyx")
)