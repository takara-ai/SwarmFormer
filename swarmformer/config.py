from dataclasses import dataclass

@dataclass
class ModelConfig:
    d_model: int
    num_layers: int
    T_local: int
    cluster_size: int
    seq_len: int
    batch_size: int
    dropout: float
    learning_rate: float
    weight_decay: float
    hf_model_id: str

SMALL_CONFIG = ModelConfig(
    d_model=128,
    num_layers=2,
    T_local=3,
    cluster_size=8,
    seq_len=256,
    batch_size=48,
    dropout=0.40,
    learning_rate=4.74e-4,
    weight_decay=0.0381,
    hf_model_id="takara-ai/SwarmFormer-Sentiment-Small"
)

BASE_CONFIG = ModelConfig(
    d_model=192,
    num_layers=2,
    T_local=3,
    cluster_size=4,
    seq_len=768,
    batch_size=48,
    dropout=0.40,
    learning_rate=4.74e-4,
    weight_decay=0.0381,
    hf_model_id="takara-ai/SwarmFormer-Sentiment-Base"
)

MODEL_CONFIGS = {
    'small': SMALL_CONFIG,
    'base': BASE_CONFIG
}

INFERENCE_BATCH_SIZE = 256 