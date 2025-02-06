import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

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
        
        # Bi-Directional view of tokens

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

def cluster_tokens(x, cluster_size=4):
    """
    x shape: (batch_size, seq_len, d_model)
    We'll chunk seq_len into seq_len//cluster_size blocks.
    Returns a list of cluster embeddings (B, num_clusters, d_model)
    and also a 'chunked_x' for easy grouping.
    """
    B, L, D = x.shape
    assert L % cluster_size == 0, "seq_len must be divisible by cluster_size."
    num_clusters = L // cluster_size
    
    # reshape to (B, num_clusters, cluster_size, d_model)
    x_reshaped = x.view(B, num_clusters, cluster_size, D)
    # average within each cluster -> (B, num_clusters, d_model)
    cluster_reps = x_reshaped.mean(dim=2)
    
    return cluster_reps

class GlobalClusterAttention(nn.Module):
    def __init__(self, d_model=16):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key   = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5
        self.attn_dropout = nn.Dropout(0.3)  # Increased from 0.1
        self.out_dropout = nn.Dropout(0.3)   # Increased from 0.1
    
    def forward(self, cluster_reps):
        """
        cluster_reps: (B, C, d_model)
        Return shape: (B, C, d_model) updated
        """
        B, C, D = cluster_reps.shape
        Q = self.query(cluster_reps)
        K = self.key(cluster_reps)
        V = self.value(cluster_reps)
        
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)  # Add dropout after attention
        
        out = torch.matmul(attn_weights, V)
        out = self.out_dropout(out)  # Add dropout to output
        return out

class BroadcastUpdater(nn.Module):
    def __init__(self, d_model=16):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        # Gate network for controlling global info integration
        self.gate_net = nn.Sequential(
            nn.Linear(2 * d_model, d_model),  # Takes concatenated local and global
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, x, cluster_reps, cluster_size=4):
        B, L, D = x.shape
        C = cluster_reps.shape[1]
        assert L % C == 0, "Mismatch in cluster count."
        
        chunked_x = x.view(B, C, -1, D)  # (B, C, cluster_size, D)
        reps_expanded = cluster_reps.unsqueeze(2).expand(B, C, chunked_x.shape[2], D)
        
        # Process global information
        global_update = self.dropout(self.linear(reps_expanded))
        
        # Prepare inputs for gating
        # Reshape to (B, L, D) for easier concatenation
        chunked_x_flat = chunked_x.view(B, L, D)
        global_update_flat = global_update.view(B, L, D)
        
        # Compute gates
        gate_input = torch.cat([chunked_x_flat, global_update_flat], dim=-1)
        gates = self.gate_net(gate_input).view(B, C, -1, D)
        
        # Apply gated update
        updated_chunk = chunked_x + gates * global_update
        x_new = updated_chunk.view(B, L, D)
        return x_new

class SwarmFormerLayer(nn.Module):
    def __init__(self, d_model=16, cluster_size=4, T_local=2):
        """
        d_model: embedding dimension
        cluster_size: how many tokens per cluster
        T_local: number of micro-iterations for local aggregator
        """
        super().__init__()
        self.local_agg    = LocalSwarmAggregator(d_model)
        self.global_attn  = GlobalClusterAttention(d_model)
        self.broadcast    = BroadcastUpdater(d_model)
        self.cluster_size = cluster_size
        self.T_local      = T_local  # new param: number of local aggregator steps per layer
        
    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        """
        # (1) Multiple micro-iterations of the local swarm
        for _ in range(self.T_local):
            x = self.local_agg(x)  # repeatedly update local embeddings

        # (2) Form cluster reps
        cluster_reps = cluster_tokens(x, self.cluster_size)  # (B, C, d_model)

        # (3) Global aggregator
        updated_reps = self.global_attn(cluster_reps)        # (B, C, d_model)

        # (4) Broadcast back
        x_out = self.broadcast(x, updated_reps, self.cluster_size)  # (B, L, d_model)

        return x_out

class SwarmFormerModel(nn.Module,
                    PyTorchModelHubMixin,
                    library_name="swarmformer",
                    repo_url="https://github.com/takara-ai/SwarmFormer",
                    ):
    def __init__(self, vocab_size=10, d_model=16, seq_len=16, cluster_size=4, num_layers=2, T_local=2):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.dropout_embedding = nn.Dropout(0.4)
        self.layers = nn.ModuleList([
            SwarmFormerLayer(d_model, cluster_size, T_local) for _ in range(num_layers)
        ])
        self.dropout_final = nn.Dropout(0.4)
        self.classifier = nn.Linear(d_model, 2)
        
    def forward(self, x):
        """
        x shape: (B, L)
        """
        out = self.dropout_embedding(self.embedding(x))
        
        for layer in self.layers:
            out = layer(out)
        
        pooled = out.mean(dim=1)
        pooled = self.dropout_final(pooled)  # Add dropout before classification
        logits = self.classifier(pooled)
        return logits 