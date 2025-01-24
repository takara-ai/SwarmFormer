# SwarmFormer

SwarmFormer is a novel transformer architecture that uses local-global hierarchical attention via swarming token representations. This repository contains the inference code for running sentiment analysis using pre-trained SwarmFormer models.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/takara-ai/SwarmFormer.git
cd SwarmFormer
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install the package:

```bash
pip install -e .
```

## Running Inference

The repository includes a sentiment analysis example that demonstrates how to use SwarmFormer models:

```bash
# Run with base model (default)
python examples/sentiment_analysis/run_inference.py

# Run with small model
python examples/sentiment_analysis/run_inference.py --model-size small
```

The script will:

- Download the model from HuggingFace automatically
- Run inference on the IMDb test set
- Output model size, hyperparameters, accuracy metrics, and performance statistics

## Hardware Support

The models support the following hardware backends:

- NVIDIA CUDA (primary, optimized)
- Apple Silicon MPS
- CPU

Performance metrics listed below are based on CUDA execution. Performance may vary significantly on other backends.

## Pre-trained Models

Pre-trained models are available in the [SwarmFormer Collection](https://huggingface.co/collections/takara-ai/swarmformer-678f8d9baec74b46f9aa3024) on HuggingFace:

### SwarmFormer-Small

- Lightweight model optimized for efficiency
- Architecture:
  - Embedding dimension: 128
  - Number of layers: 2
  - Local update steps: 3
  - Cluster size: 8
  - Sequence length: 256
- Performance:
  - Accuracy: 86.20%
  - Inference time: 0.36s (25k samples)
  - Mean batch latency: 3.67ms
  - Throughput: 45k samples/s
  - VRAM: 8GB minimum

### SwarmFormer-Base

- Standard model with higher capacity
- Architecture:
  - Embedding dimension: 192
  - Number of layers: 2
  - Local update steps: 3
  - Cluster size: 4
  - Sequence length: 768
- Performance:
  - Accuracy: 89.03%
  - Mean batch latency: 4.83ms
  - Peak memory: 9.13GB
  - Precision: 87.22%
  - Recall: 91.46%
  - F1: 89.29%

## Memory Requirements

Memory requirements (with batch size 256):

- Small: 8GB+ VRAM
- Base: 10GB+ VRAM
- Can be reduced by using smaller batch sizes

## TODO

- [ ] Add full training pipeline code
- [ ] Add model architecture documentation
- [ ] Add training configuration guide
- [ ] Add evaluation scripts for custom datasets
- [ ] Add model export utilities
- [ ] Add pip package for SwarmFormer Layer

## Citation

If you use SwarmFormer in your research, please cite:

```bibtex
@article{legg2025swarmformer,
  title={SwarmFormer: Local-Global Hierarchical Attention via Swarming Token Representations},
  author={Legg, Jordan and Sturmanis, Mikus and {Takara.ai}},
  journal={Takara.ai Research},
  year={2025},
  url={https://takara.ai/papers/SwarmFormer-Local-Global-Hierarchical-Attention-via-Swarming-Token-Representations.pdf}
}
```

## Contact

For questions, collaborations, or other inquiries:

- Email: research@takara.ai
- Models: [HuggingFace Collection](https://huggingface.co/collections/takara-ai/swarmformer-678f8d9baec74b46f9aa3024)
