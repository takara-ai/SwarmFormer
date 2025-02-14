import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --------------------------------------
# 1) HELPER FUNCTIONS & CAUSAL MASKING
# --------------------------------------

def causal_mask(seq_len):
    """
    Returns a [seq_len, seq_len] boolean mask that disallows
    attending to future positions (strictly upper-triangular is masked).
    
    For positions j > i, mask[i, j] = True => large negative in attention logits.
    """
    # mask[i, j] = True if j > i
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

def gating_function(x_local, x_global, dropout_p=0.0):
    """
    Gated merge of local token embedding x_local with broadcast representation x_global:
      x_out = x_local + gate * (x_global - x_local)

    Gate is produced by:
      g = σ( W_g [x_local; x_global] )
    """
    B, N, D = x_local.shape
    gate_inp = torch.cat([x_local, x_global], dim=-1)  # [B, N, 2*D]

    # Create linear layers on the same device as input
    device = x_local.device
    linear1 = nn.Linear(2*D, D, bias=True).to(device)
    linear2 = nn.Linear(D, D, bias=True).to(device)

    # Forward pass through the layers
    hidden = F.gelu(nn.functional.dropout(
        linear1(gate_inp), p=dropout_p
    ))
    gate_logits = linear2(hidden)  # [B, N, D]
    gate = torch.sigmoid(gate_logits)

    return x_local + gate * (x_global - x_local)

# ------------------------------------------------
# 2) LOCAL SWARM AGGREGATOR (Multi-Head, Causal)
# ------------------------------------------------

class LocalSwarmAggregator(nn.Module):
    """
    Multi-head local attention aggregator with causal masking.
    Instead of a simple MLP aggregator, we restrict attention to 
    a local window around each position (with optional half-width).
    """
    def __init__(self, d_model, num_heads, window_size, dropout_p=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size  # half-width of local neighborhood
        self.dropout = nn.Dropout(dropout_p)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: [B, seq_len, d_model]
        Return: [B, seq_len, d_model]
        """
        B, N, D = x.shape
        # (1) Project Q, K, V
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim)

        # (2) We create a full causal mask [N,N], then also block out
        # positions beyond the local window in the past.
        c_mask = causal_mask(N).to(x.device)  # [N, N], True => block
        # Expand for heads: shape => [num_heads, N, N]
        c_mask = c_mask.unsqueeze(0).expand(self.num_heads, N, N)

        # Also block out j < i-window_size
        idxs = torch.arange(N, device=x.device)
        i_idx = idxs.view(N, 1).expand(N, N)  # row index for each (i, j)
        j_idx = idxs.view(1, N).expand(N, N)  # col index for each (i, j)
        left_boundary = i_idx - self.window_size
        # If j < i - window_size => block
        local_block = (j_idx < left_boundary)
        # Combine local_block with the existing causal mask
        combined_block = c_mask.bool() | local_block
        # Convert boolean to -1e9/0 for the attention logits
        block_mask = torch.where(
            combined_block, torch.tensor(-1e9, device=x.device), torch.tensor(0.0, device=x.device)
        )

        # (3) Compute attention scores
        q = q.transpose(1, 2)  # [B, num_heads, N, head_dim]
        k = k.transpose(1, 2)  # [B, num_heads, N, head_dim]
        v = v.transpose(1, 2)  # [B, num_heads, N, head_dim]

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Add block_mask: shape [num_heads, N, N], broadcast batch dimension
        attn_logits = attn_logits + block_mask.unsqueeze(0)  # [B, num_heads, N, N]

        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (4) Apply attention
        out = torch.matmul(attn_weights, v)  # [B, num_heads, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)

        return out

# ------------------------------------------------
# 3) GLOBAL CLUSTER ATTENTION
# ------------------------------------------------

class GlobalClusterAttention(nn.Module):
    """
    Multi-head attention among cluster representatives R,
    with optional causal mask across clusters.
    """
    def __init__(self, d_model, num_heads, dropout_p=0.1, causal_global=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal_global = causal_global

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, reps):
        """
        reps: [B, C, d_model]  (cluster representatives)
        Return updated cluster reps: [B, C, d_model]
        """
        B, C, D = reps.shape
        q = self.q_proj(reps).view(B, C, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(reps).view(B, C, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(reps).view(B, C, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Optional cluster-level causal mask
        if self.causal_global:
            # shape: [C, C]
            c_mask = torch.triu(torch.ones(C, C, device=reps.device), diagonal=1).bool()
            block_mask = torch.where(
                c_mask, torch.tensor(-1e9, device=reps.device), torch.tensor(0.0, device=reps.device)
            )
            # broadcast across batch & heads
            attn_logits = attn_logits + block_mask.unsqueeze(0).unsqueeze(0)

        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # [B, num_heads, C, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, C, D)
        out = self.out_proj(out)
        return out

# ------------------------------------------------
# 4) SWARM BLOCK: Local -> (Clusters) -> Global -> Broadcast
# ------------------------------------------------

class SwarmLMBlock(nn.Module):
    """
    One hierarchical block containing:
      1) T_local steps of local swarm aggregator (causal local MHA)
      2) cluster formation + cross-cluster mixing
      3) global attention among clusters
      4) broadcast with gating back to tokens
    """
    def __init__(
        self,
        d_model,
        num_heads,
        window_size,
        cluster_size,
        t_local=2,
        dropout_p=0.1,
        causal_global=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.cluster_size = cluster_size
        self.t_local = t_local
        self.dropout = nn.Dropout(dropout_p)

        # Repeated local aggregator
        self.local_swarm_layers = nn.ModuleList([
            LocalSwarmAggregator(d_model, num_heads, window_size, dropout_p=dropout_p)
            for _ in range(t_local)
        ])

        # Cross-cluster MLP
        self.cross_cluster_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(d_model, d_model),
        )

        self.global_attn = GlobalClusterAttention(
            d_model, num_heads, dropout_p=dropout_p, causal_global=causal_global
        )

    def forward(self, x):
        """
        x: [B, seq_len, d_model]
        Returns: [B, seq_len, d_model]
        """
        B, N, D = x.shape
        # 1) Local aggregator repeated
        local_res = x
        for layer in self.local_swarm_layers:
            local_attn = layer(local_res)  # [B, N, D]
            local_res = local_res + self.dropout(local_attn)

        x_local = local_res

        # 2) Cluster formation (simple chunk-based)
        C = N // self.cluster_size
        assert N % self.cluster_size == 0, "seq_len not divisible by cluster_size."
        reps = []
        for c in range(C):
            start_idx = c * self.cluster_size
            end_idx = start_idx + self.cluster_size
            chunk = x_local[:, start_idx:end_idx, :]  # [B, cluster_size, d_model]
            rep = chunk.mean(dim=1, keepdim=True)      # [B, 1, d_model]
            reps.append(rep)
        reps = torch.cat(reps, dim=1)  # [B, C, d_model]

        # 2.1) Cross-cluster mixing: each cluster sees sum of others => pass MLP
        reps_sum = reps.sum(dim=1, keepdim=True)  # [B, 1, d_model]
        # quick mixing approach: reps_mixed[c] = reps[c] + MLP( sum_of_all - reps[c] )
        reps_mixed = reps + self.cross_cluster_mlp(reps_sum - reps)

        # 3) Global attention among cluster reps
        reps_new = self.global_attn(reps_mixed)  # [B, C, d_model]

        # 4) Broadcast updated cluster reps back to tokens
        x_out = x_local.clone()
        for c in range(C):
            start_idx = c * self.cluster_size
            end_idx = start_idx + self.cluster_size
            broad = reps_new[:, c, :].unsqueeze(1)  # [B, 1, d_model]
            chunk_local = x_local[:, start_idx:end_idx, :]
            chunk_broadcast = broad.expand(-1, self.cluster_size, -1)

            chunk_out = gating_function(
                chunk_local, chunk_broadcast, dropout_p=self.dropout.p
            )
            x_out[:, start_idx:end_idx, :] = chunk_out

        return x_out

# ------------------------------------------------
# 5) FULL MODEL FOR LANGUAGE MODELING
# ------------------------------------------------

class SwarmFormerLM(nn.Module):
    """
    A language model that stacks multiple SwarmLMBlocks:
      - Token + Positional embedding
      - Repeated hierarchical local/global aggregator blocks
      - Final LayerNorm + linear vocab projection
    """
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        d_model=256,
        n_layers=4,
        num_heads=4,
        window_size=2,
        cluster_size=8,
        dropout_p=0.1,
        causal_global=False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # Token & positional embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)

        # Stacked Swarm blocks
        self.blocks = nn.ModuleList([
            SwarmLMBlock(
                d_model=d_model,
                num_heads=num_heads,
                window_size=window_size,
                cluster_size=cluster_size,
                t_local=2,
                dropout_p=dropout_p,
                causal_global=causal_global
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout_p)

        # Optional initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        """
        input_ids: [B, seq_len]
        Returns: [B, seq_len, vocab_size] (logits)
        """
        B, N = input_ids.shape
        assert N <= self.max_seq_len, "Sequence length exceeds max_seq_len"

        # 1) Token + positional embeddings
        pos_ids = torch.arange(N, device=input_ids.device).unsqueeze(0)  # [1, seq_len]
        tok_emb = self.token_emb(input_ids)  # [B, N, d_model]
        pos_emb = self.pos_emb(pos_ids)      # [1, N, d_model]
        x = tok_emb + pos_emb

        # 2) Pass through the stacked SwarmLMBlocks
        for block in self.blocks:
            x = block(x)

        # 3) Final LN + projection
        x = self.ln_f(x)           # [B, N, d_model]
        logits = self.lm_head(x)   # [B, N, vocab_size]

        return logits
