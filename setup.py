from setuptools import setup, find_packages

setup(
    name="qeep",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "numba",
        "jax",
        "interpax",
        "torchquad",
        "vegas",
        "matplotlib",
        "pyyaml",
        "tqdm",
    ],
    extras_require={
        # Needed only for the symbolic theory scripts and the equivalence-principle
        # kernels that depend on M(k) (notebooks/paper, scripts/thy.py).
        "theory": ["sympy", "sympy2jax", "quadax"],
        # Needed only to run on Abacus simulations (scripts/abacus_*.py, run_recs*.py).
        "abacus": ["astropy", "classy", "abacusutils"],
    },
    author="Omar Darwish",
    description="Quadratic Estimators for LSS",
) 