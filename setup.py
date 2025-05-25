"""
Setup script for the 3WayCoT framework.
"""

from setuptools import setup, find_packages

setup(
    name="threeway_cot",
    version="2.0.0",
    author="3WayCoT Team",
    description="Three-Way Chain of Thought framework for uncertainty-aware reasoning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/3waycot/framework",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "scikit-learn>=1.0.0",
        "networkx>=2.8.0",
        "matplotlib>=3.5.0",
        "nltk>=3.7",
        "tqdm>=4.64.0",
        "scipy>=1.7.0",
        "concept-py>=0.9.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
)
