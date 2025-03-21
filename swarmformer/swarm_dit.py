import torch
import torch.nn as nn

# Import from existing modules
from swarmformer.diffusion import SinusoidalPositionEmbeddings
from swarmformer.model import cluster_tokens, LocalSwarmAggregator, GlobalClusterAttention


class AdaptiveLayerNorm(nn.Module):
    """Layer normalization with adaptive scale and shift based on condition"""
    def __init__(self, dim, condition_dim=None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        condition_dim = condition_dim or dim
        self.scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, dim * 2)
        )
        self.dim = dim
    
    def forward(self, x, condition=None):
        if condition is None:
            return self.norm(x)
            
        scale, shift = self.scale_shift(condition).chunk(2, dim=-1)
        
        # Ensure condition has right shape for broadcasting
        if condition.dim() == 2 and x.dim() == 3:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
            
        normalized = self.norm(x)
        return normalized * (1 + scale) + shift


class SwarmDiT(nn.Module):
    """
    SwarmFormer enhanced with Diffusion Transformer (DiT) techniques
    """
    def __init__(self, input_dim=256, hidden_dim=32, num_layers=2, cluster_size=16):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cluster_size = cluster_size
        self.seq_len = input_dim
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
        )
        
        # Time embedding
        self.time_embedding = SinusoidalPositionEmbeddings(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # DiT-style layers with adaLN
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'norm1': AdaptiveLayerNorm(hidden_dim, hidden_dim),
                'local_agg': LocalSwarmAggregator(hidden_dim),
                'norm2': AdaptiveLayerNorm(hidden_dim, hidden_dim),
                'global_attn': GlobalClusterAttention(hidden_dim)
            }))
        
        # Output layers
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x, t):
        """
        Forward pass for noise prediction in diffusion.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len] or [batch_size, seq_len, 1]
            t: Timestep tensor of shape [batch_size]
            
        Returns:
            Noise prediction of shape [batch_size, seq_len]
        """
        batch_size = x.shape[0]
        
        # Reshape x to [batch_size, seq_len, 1] for embedding
        x = x.view(batch_size, self.seq_len, 1)
        
        # Embed input
        h = self.input_embedding(x)  # [batch_size, seq_len, hidden_dim]
        
        # Get time embedding for conditional adaLN
        time_emb = self.time_mlp(self.time_embedding(t))  # [batch_size, hidden_dim]
        
        # Process through DiT-enhanced SwarmFormer layers
        for layer in self.layers:
            # Local aggregation with adaptive norm
            h_local = layer['norm1'](h, time_emb)
            h_local = layer['local_agg'](h_local)
            h = h + h_local
            
            # Global attention with adaptive norm
            h_global = layer['norm2'](h, time_emb)
            
            # Cluster tokens, apply attention, and broadcast back
            cluster_reps = cluster_tokens(h_global, self.cluster_size)
            updated_reps = layer['global_attn'](cluster_reps)
            
            # Reshape back to original sequence
            h_global_expanded = updated_reps.unsqueeze(2).expand(
                -1, -1, self.cluster_size, -1
            ).reshape(batch_size, self.seq_len, self.hidden_dim)
            
            h = h + h_global_expanded
        
        # Final processing
        h = self.output_norm(h)
        output = self.output_proj(h)  # [batch_size, seq_len, 1]
        
        # Reshape back to [batch_size, input_dim]
        output = output.squeeze(-1)
        
        return output


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Linear schedule for diffusion process betas.
    
    Args:
        timesteps: Number of diffusion timesteps
        beta_start: Starting noise level
        beta_end: Ending noise level
        
    Returns:
        Tensor of betas
    """
    return torch.linspace(beta_start, beta_end, timesteps)


class DiffusionSampler:
    """
    Utility class for sampling from diffusion models.
    """
    def __init__(self, model, timesteps=20, beta_start=0.0001, beta_end=0.02, device=None):
        """
        Initialize the diffusion sampler.
        
        Args:
            model: SwarmDiT model
            timesteps: Number of diffusion steps
            beta_start: Minimum beta value
            beta_end: Maximum beta value
            device: Device to run on
        """
        self.model = model
        self.timesteps = timesteps
        self.device = device or (next(model.parameters()).device)
        
        # Define beta schedule
        self.betas = linear_beta_schedule(timesteps, beta_start, beta_end).to(self.device)
        
        # Pre-compute diffusion values
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]])
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # For sampling
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def add_noise(self, x_start, t):
        """
        Add noise to data according to noise schedule.
        
        Args:
            x_start: Clean data [batch_size, input_dim]
            t: Time step indices [batch_size]
            
        Returns:
            Tuple of (noisy_x, noise)
        """
        t = t.to(self.device)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        # Get relevant pre-computed values
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        
        # Sample random noise
        noise = torch.randn_like(x_start).to(self.device)
        
        # Mix clean input with noise according to schedule
        noisy_x = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        return noisy_x, noise
    
    def sample(self, batch_size=1, steps=None):
        """
        Sample new data from the model.
        
        Args:
            batch_size: Number of samples to generate
            steps: Number of diffusion steps (if None, uses model timesteps)
            
        Returns:
            Generated samples [batch_size, input_dim]
        """
        steps = steps or self.timesteps
        
        # Start with random noise
        x = torch.randn(batch_size, self.model.input_dim, device=self.device)
        
        # Progressively denoise
        for t_idx in reversed(range(0, steps)):
            # Current timestep
            t = torch.full((batch_size,), t_idx, device=self.device, dtype=torch.long)
            
            # Get model's noise prediction
            predicted_noise = self.model(x, t)
            
            # Get the scaling coefficients
            alpha_t = self.alphas[t_idx]
            alpha_cumprod_t = self.alphas_cumprod[t_idx]
            beta_t = self.betas[t_idx]
            
            # No noise at timestep 0
            if t_idx > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
            
            # Update x via reverse diffusion step
            x = 1 / torch.sqrt(alpha_t) * (
                x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
            ) + torch.sqrt(beta_t) * noise
        
        return torch.clamp(x, 0, 1) 