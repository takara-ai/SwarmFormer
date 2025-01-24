from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="swarmformer",
    version="0.1.0",
    author="Jordan Legg, Mikus Sturmanis",
    author_email="research@takara.ai",
    description="SwarmFormer: Local-Global Hierarchical Attention via Swarming Token Representations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/takara-ai/SwarmFormer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers",
        "datasets",
        "numpy",
        "tqdm",
        "scikit-learn",
        "huggingface-hub",
        "safetensors",
    ],
) 