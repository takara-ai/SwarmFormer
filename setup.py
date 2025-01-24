from setuptools import setup, find_packages

setup(
    name="swarmformer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.19.0",
        "transformers>=4.0.0",
        "datasets>=2.0.0",
        "nltk>=3.6.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.65.0",
        "safetensors>=0.3.0",
        "huggingface-hub>=0.16.0",
    ],
    author="Jordan Legg",
    description="SwarmFormer: A novel transformer architecture for sentiment analysis",
    python_requires=">=3.8",
) 