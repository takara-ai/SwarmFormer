import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CausalSwarmFormerConfig

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x / rms
        return (self.weight * x).to(dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 5_000_000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class CausalLocalSwarmAggregator(nn.Module):
    def __init__(self, d_model: int, shifts: list[int] | None = None):
        super().__init__()
        self.shifts = shifts or [1, 2, 4, 8]
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(2 * d_model, d_model, bias=True),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=True),
            nn.Sigmoid(),
        )
        self._init_near_zero()

    def _init_near_zero(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
        for i, module in enumerate(self.gate_net):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    if i == len(self.gate_net) - 2:
                        nn.init.constant_(module.bias, -4.0)
                    else:
                        nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        neighbor_sum = x.clone()
        n_valid = torch.ones(B, L, 1, device=x.device, dtype=x.dtype)

        for s in self.shifts:
            if s >= L:
                continue
            shifted = torch.roll(x, shifts=s, dims=1)
            shifted[:, :s, :] = 0.0
            mask = torch.ones(B, L, 1, device=x.device, dtype=x.dtype)
            mask[:, :s, :] = 0.0
            neighbor_sum = neighbor_sum + shifted
            n_valid = n_valid + mask

        neighbor_info = neighbor_sum / n_valid
        update = self.mlp(neighbor_info)
        gates = self.gate_net(torch.cat([x, update], dim=-1))
        return x + gates * (update - x)

def causal_cluster_tokens(x: torch.Tensor, cluster_size: int) -> torch.Tensor:
    B, L, D = x.shape
    assert L % cluster_size == 0
    num_clusters = L // cluster_size
    return x.view(B, num_clusters, cluster_size, D)[:, :, -1, :].contiguous()

class LearnedBroadcast(nn.Module):
    """
    Expands cluster attention output back to token level with per-token
    modulation. Each token gets a UNIQUE output based on both the cluster
    attention result and its own hidden state.

    broadcast(h, attn_out) = proj(cat(h, expanded_attn))

    This is O(L) — just a linear projection per token.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Projects concatenated [token_state, cluster_attn] → attention contribution
        self.proj = nn.Linear(2 * d_model, d_model, bias=False)
        self._init()

    def _init(self):
        # Initialize so the output ≈ the cluster attention part (second half of input)
        # proj.weight is (d_model, 2*d_model). Set first d_model cols ≈ 0, last d_model cols ≈ I
        nn.init.zeros_(self.proj.weight)
        d = self.proj.weight.shape[0]
        # Copy identity into the second half (cluster attn passthrough)
        self.proj.weight.data[:, d:].copy_(torch.eye(d) * 0.9)
        # Small random init for the first half (token modulation, learned)
        nn.init.xavier_uniform_(self.proj.weight.data[:, :d], gain=0.05)

    def forward(self, h: torch.Tensor, cluster_attn_out: torch.Tensor, cluster_size: int) -> torch.Tensor:
        """
        h: (B, L, D) — token-level hidden states (from local_agg)
        cluster_attn_out: (B, C, D) — cluster attention output
        Returns: (B, L, D) — per-token attention contribution (added to residual)
        """
        B, C, D = cluster_attn_out.shape
        # Expand cluster output to token level
        expanded = cluster_attn_out.unsqueeze(2).expand(B, C, cluster_size, D)
        expanded = expanded.reshape(B, C * cluster_size, D)
        # Concatenate token state + cluster attention, project to per-token output
        combined = torch.cat([h, expanded], dim=-1)  # (B, L, 2D)
        return self.proj(combined)  # (B, L, D)

class CausalGQAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, head_dim, use_rope=True):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)

    def _attn(self, x, cos, sin):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_rope and cos is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.full((S, S), float("-inf"), device=q.device, dtype=q.dtype), diagonal=1)
        scores = scores + mask
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)
        return self.o_proj(out)

    def forward_cluster(self, cluster_reps, cos, sin):
        return self._attn(cluster_reps, cos, sin)

    def forward_full(self, x, cos, sin):
        return self._attn(x, cos, sin)

class CausalSwarmFormerLayer(nn.Module):
    def __init__(self, config: CausalSwarmFormerConfig, layer_idx: int):
        super().__init__()
        self.cluster_size = config.cluster_size
        self.T_local = config.T_local

        self.input_layernorm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.d_model, config.rms_norm_eps)

        shifts = [2**i for i in range(int(math.log2(config.cluster_size)))]
        self.local_agg = CausalLocalSwarmAggregator(config.d_model, shifts=shifts)

        use_rope = ((layer_idx + 1) % config.no_rope_layer_interval != 0)
        self.attn = CausalGQAttention(
            d_model=config.d_model,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            use_rope=use_rope,
        )

        self.broadcast = LearnedBroadcast(config.d_model)

        self.mlp = SwiGLUMLP(config.d_model, config.intermediate_size)

    def forward(self, x, cos_full, sin_full, cos_cluster, sin_cluster, use_cluster=True):
        residual = x
        h = self.input_layernorm(x)

        if use_cluster:
            for _ in range(self.T_local):
                h = self.local_agg(h)

            cluster_reps = causal_cluster_tokens(h, self.cluster_size)
            attn_out = self.attn.forward_cluster(cluster_reps, cos_cluster, sin_cluster)

            attn_contrib = self.broadcast(h, attn_out, self.cluster_size)
            x = residual + attn_contrib
        else:
            attn_out = self.attn.forward_full(h, cos_full, sin_full)
            x = residual + attn_out

        # MLP block
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x

class CausalSwarmFormerForLM(nn.Module):
    def __init__(self, config: CausalSwarmFormerConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            CausalSwarmFormerLayer(config, layer_idx=i)
            for i in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.rotary_emb = RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

    def forward(self, input_ids, labels=None, use_cluster=True):
        B, L = input_ids.shape
        cs = self.config.cluster_size
        assert L % cs == 0, f"seq_len {L} must be divisible by cluster_size {cs}"

        x = self.embed_tokens(input_ids)

        positions = torch.arange(L, device=input_ids.device)
        cos_full, sin_full = self.rotary_emb(positions)

        cluster_pos = torch.arange(cs - 1, L, cs, device=input_ids.device)
        cos_cluster, sin_cluster = self.rotary_emb(cluster_pos)

        for layer in self.layers:
            if self.training and use_cluster:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, cos_full, sin_full, cos_cluster, sin_cluster, use_cluster,
                    use_reentrant=False,
                )
            else:
                x = layer(x, cos_full, sin_full, cos_cluster, sin_cluster, use_cluster=use_cluster)

        x = self.norm(x)
        logits = self.lm_head(x)

        result = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            result["loss"] = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
        return result

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.7, top_k=50,
                 eos_token_id=None, use_cluster=True):
        cs = self.config.cluster_size
        for _ in range(max_new_tokens):
            B, L = input_ids.shape
            pad_len = (cs - L % cs) % cs
            if pad_len > 0:
                padded = F.pad(input_ids, (pad_len, 0), value=0)
            else:
                padded = input_ids

            logits = self.forward(padded, use_cluster=use_cluster)["logits"][:, -1, :]

            if temperature > 0:
                logits = logits / temperature
            if top_k > 0:
                topk_vals, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < topk_vals[:, -1:]] = float("-inf")

            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return input_ids
