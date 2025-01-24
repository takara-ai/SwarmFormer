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

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run inference using either the small or base model:

```bash
# Run with base model (default)
python run_inference.py

# Run with small model
python run_inference.py --model-size small
```

The script will:

- Download the model from HuggingFace automatically
- Run inference on the IMDb test set
- Output model size, hyperparameters, accuracy metrics, and performance statistics

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

## Model Architecture

Both models follow a hierarchical architecture:

1. Token embedding layer with dropout
2. Multiple SwarmFormer layers, each containing:
   - Local swarm aggregator with gated updates
   - Clustering mechanism
   - Global cluster attention
   - Broadcast updater
3. Mean pooling and classification

## Limitations

- SwarmFormer-Small: Not suitable for sequences >256 tokens
- SwarmFormer-Base: Not suitable for sequences >768 tokens
- Models are trained for English text classification only
- Not designed for text generation or translation tasks
- Memory requirements (with batch size 256):
  - Small: 8GB+ VRAM
  - Base: 10GB+ VRAM
  - Can be reduced by using smaller batch sizes

## TODO

- [ ] Add full training pipeline code
- [ ] Add model architecture documentation
- [ ] Add training configuration guide
- [ ] Add evaluation scripts for custom datasets
- [ ] Add model export utilities

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
