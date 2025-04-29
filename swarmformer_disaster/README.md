# SwarmFormer: Disaster Tweet Classification

A distilled SwarmFormer transformer tailored for binary classification of disaster-related tweets. Combines efficient local-global hierarchical attention with knowledge distillation from ModernBERT-base to deliver fast, lightweight inference on resource-constrained devices.

## Benchmarks

**Environment**: Apple M1 (MPS)  
**Test samples**: 2,620 | **Sequence length**: 64 | **Batch size**: 32

| Model                       | Params  | Latency (ms/sample) | Memory (MB) | Accuracy | F1     | Precision | Recall |
| --------------------------- | ------- | ------------------- | ----------- | -------- | ------ | --------- | ------ |
| **SwarmFormer (Distilled)** | 6.83M   | 0.718               | 28.89       | 0.8137   | 0.8820 | 0.9470    | 0.8253 |
| DistilBERT-base             | 66.96M  | 19.05               | 258.83      | 0.8424   | 0.9143 | 0.8446    | 0.9964 |
| BERT-base-uncased           | 109.48M | 37.78               | 466.70      | 0.8435   | 0.9151 | 0.8435    | 1.0000 |
| ModernBERT-base (Teacher)   | 149.61M | 66.86               | 570.79      | 0.8790   | 0.9266 | 0.9488    | 0.9054 |

### Speedup vs BERT-base-uncased

| Model                       | Speedup |
| --------------------------- | ------- |
| **SwarmFormer (Distilled)** | 52.61×  |
| DistilBERT-base             | 1.98×   |
| ModernBERT-base             | 0.57×   |

## Training Methodology

- **Dataset**: `disaster_response_messages` (HuggingFace) filtered to binary labels (`related` ≠ 2).
- **Teacher**: ModernBERT-base fine-tuned for 3 epochs (lr=2e-5, weight_decay=1e-2, warmup=10%).
- **Student**: SwarmFormerModel (d_model=128, num_layers=2, cluster_size=4, T_local=2) distilled over 6 epochs (lr=5e-4) using α=0.5 weighted cross-entropy + KL divergence (T=2).
- **Optimizer**: AdamW
- **Scheduler**: Linear warmup (10%)
- **Evaluation**: Accuracy & F1 on held-out test set.

## Model Configuration

```yaml
d_model: 128
num_layers: 2
cluster_size: 4
T_local: 2
sequence_length: 64
precision: torch.float32
```

## Usage

```bash
git clone <repo_url>
cd <repo_root>/swarmformer_disaster
pip install -r requirements.txt

# Distill student model (with optional W&B logging)
python distill_disaster.py --use_wandb

# Run inference on a sample
python distill_disaster.py
```

Or in Python:

```python
import torch
from disaster_data import get_disaster_dataloaders
from swarmformer.model import SwarmFormerModel

# Load student model
model = SwarmFormerModel(
    vocab_size=tokenizer.vocab_size,
    d_model=128,
    seq_len=64,
    cluster_size=4,
    num_layers=2,
    T_local=2
)
model.load_state_dict(torch.load("student_model.pt", map_location="cpu"))
model.eval()

# Prepare data
train_loader, val_loader, test_loader, tokenizer = get_disaster_dataloaders(
    "bert-base-uncased", seq_len=64, batch_size=32
)

# Evaluate
# ...
```

## Directory Structure

```
swarmformer_disaster/
├── README.md
├── distill_disaster.py
├── disaster_data.py
├── teacher_model.pt
├── student_model.pt
└── requirements.txt
```

## License

[MIT](LICENSE)
