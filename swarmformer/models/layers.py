"""
Core SwarmFormer layer implementations.
Contains the building blocks of the SwarmFormer architecture.
"""

import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size=10, d_model=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        """
        x shape: (batch_size, seq_len)
        returns: (batch_size, seq_len, d_model)
        """
        return self.embed(x)

class LocalSwarmAggregator(nn.Module):
    def __init__(self, d_model=16):
        super().__init__()
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        # A small MLP to combine local info
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, d_model)
        )
        # Gate network to control information flow
        self.gate_net = nn.Sequential(
            nn.Linear(2 * d_model, d_model),  # Takes concatenated current and update
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, x):
        B, L, D = x.shape
        x = self.dropout1(x)
        
        x_left = torch.roll(x, shifts=1, dims=1)
        x_right = torch.roll(x, shifts=-1, dims=1)
        
        neighbor_info = (x_left + x + x_right) / 3.0
        update = self.mlp(neighbor_info)
        update = self.dropout2(update)
        
        # Compute dynamic gates
        gate_input = torch.cat([x, update], dim=-1)  # (B, L, 2*D)
        gates = self.gate_net(gate_input)  # (B, L, D)
        
        # Gated update
        x_new = x + gates * (update - x)
        return x_new

def cluster_tokens(x, cluster_size):
    """Helper function to cluster tokens into fixed-size groups"""
    B, L, D = x.shape
    assert L % cluster_size == 0, f"Sequence length {L} must be divisible by cluster size {cluster_size}"
    
    # Reshape to group tokens into clusters
    x = x.view(B, L // cluster_size, cluster_size, D)
    
    # Mean pooling within each cluster
    cluster_reps = x.mean(dim=2)  # (B, num_clusters, D)
    return cluster_reps

class GlobalClusterAttention(nn.Module):
    def __init__(self, d_model=16):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        """
        x shape: (batch_size, num_clusters, d_model)
        """
        Q = self.q_proj(x)  # (B, C, D)
        K = self.k_proj(x)  # (B, C, D)
        V = self.v_proj(x)  # (B, C, D)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)  # (B, C, C)
        attn = torch.softmax(scores, dim=-1)  # (B, C, C)
        attn = self.dropout(attn)
        
        # Compute weighted sum
        out = torch.matmul(attn, V)  # (B, C, D)
        return out

class BroadcastUpdater(nn.Module):
    def __init__(self, d_model=16):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        # Gate network to control information flow from clusters back to tokens
        self.gate_net = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, tokens, cluster_reps, cluster_size):
        """
        tokens: (B, L, D) - original token representations
        cluster_reps: (B, C, D) - updated cluster representations
        cluster_size: int - number of tokens per cluster
        """
        B, L, D = tokens.shape
        C = cluster_reps.shape[1]
        
        # Project cluster representations
        cluster_update = self.proj(cluster_reps)  # (B, C, D)
        cluster_update = self.dropout(cluster_update)
        
        # Repeat cluster updates for each token in the cluster
        cluster_update = cluster_update.unsqueeze(2)  # (B, C, 1, D)
        cluster_update = cluster_update.expand(-1, -1, cluster_size, -1)  # (B, C, K, D)
        cluster_update = cluster_update.reshape(B, L, D)  # (B, L, D)
        
        # Compute gates
        gate_input = torch.cat([tokens, cluster_update], dim=-1)  # (B, L, 2*D)
        gates = self.gate_net(gate_input)  # (B, L, D)
        
        # Apply gated update
        tokens_new = tokens + gates * (cluster_update - tokens)
        return tokens_new

class SwarmFormerLayer(nn.Module):
    def __init__(self, d_model=16, cluster_size=4, T_local=2):
        """
        d_model: embedding dimension
        cluster_size: how many tokens per cluster
        T_local: number of micro-iterations for local aggregator
        """
        super().__init__()
        self.local_agg = LocalSwarmAggregator(d_model)
        self.global_attn = GlobalClusterAttention(d_model)
        self.broadcast = BroadcastUpdater(d_model)
        self.cluster_size = cluster_size
        self.T_local = T_local
    
    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        """
        # (1) Multiple micro-iterations of the local swarm
        for _ in range(self.T_local):
            x = self.local_agg(x)
        
        # (2) Form cluster reps
        cluster_reps = cluster_tokens(x, self.cluster_size)
        
        # (3) Global aggregator
        updated_reps = self.global_attn(cluster_reps)
        
        # (4) Broadcast back
        x_out = self.broadcast(x, updated_reps, self.cluster_size)
        
        return x_out 