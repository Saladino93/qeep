from setuptools import setup, find_packages

setup(
    name="qeep",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "jax",
        "interpax",
        "torchquad",
    ],
    author="Omar Darwish",
    description="Quadratic Estimators for LSS",
) 