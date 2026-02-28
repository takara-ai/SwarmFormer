from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int
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
    vocab_size=30522,
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
    vocab_size=30522,
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


@dataclass
class CausalSwarmFormerConfig:
    vocab_size: int = 128256
    d_model: int = 2048
    num_layers: int = 4           # 4 for testing, 36 for full
    num_attention_heads: int = 16  # Q heads
    num_kv_heads: int = 4          # KV heads (GQA 4:1)
    head_dim: int = 128
    intermediate_size: int = 11008
    cluster_size: int = 64
    T_local: int = 1
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    no_rope_layer_interval: int = 4  # every 4th layer skips RoPE
    max_seq_len: int = 2048
    tie_word_embeddings: bool = True
