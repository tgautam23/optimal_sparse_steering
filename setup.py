from setuptools import setup, find_packages

setup(
    name="optimal_sparse_steering",
    version="0.1.0",
    description="Optimal Sparse Steering via Convex Optimization for Language Models",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformer-lens>=2.0.0",
        "sae-lens>=3.0.0",
        "cvxpy>=1.4.0",
        "datasets>=2.14.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "transformers>=4.35.0",
    ],
)
