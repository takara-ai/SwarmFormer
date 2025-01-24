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

- SwarmFormer-Small: Lightweight model (2.1M parameters)
- SwarmFormer-Base: Standard model (4.8M parameters)

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
