import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from swarmformer.model import LocalSwarmAggregator, GlobalClusterAttention, cluster_tokens


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time step conditioning"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # Pad if odd dimension
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1), mode='constant')
        return embeddings


class SwarmDiffusionModel(nn.Module):
    """
    SwarmFormer architecture for diffusion tasks.
    This is the winning architecture from our experiments with diffusion models.
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
            nn.LayerNorm(hidden_dim)
        )
        
        # Time embedding
        self.time_embedding = SinusoidalPositionEmbeddings(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # SwarmFormer layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(hidden_dim),
                'local_agg': LocalSwarmAggregator(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim),
                'global_attn': GlobalClusterAttention(hidden_dim)
            }))
        
        # Output layers
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights for better optimization."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x, t):
        """Forward pass for noise prediction in diffusion."""
        batch_size = x.shape[0]
        
        # Reshape x to [batch_size, seq_len, 1] for embedding
        x = x.view(batch_size, self.seq_len, 1)
        
        # Embed input
        h = self.input_embedding(x)  # [batch_size, seq_len, hidden_dim]
        
        # Get time embedding and broadcast
        time_emb = self.time_mlp(self.time_embedding(t))  # [batch_size, hidden_dim]
        time_emb = time_emb.unsqueeze(1).expand(-1, self.seq_len, -1)  # [batch_size, seq_len, hidden_dim]
        
        # Add time embedding
        h = h + time_emb
        
        # Process through SwarmFormer layers
        for layer in self.layers:
            # Local aggregation
            h_local = layer['norm1'](h)
            h_local = layer['local_agg'](h_local)
            h = h + h_local
            
            # Global attention
            h_global = layer['norm2'](h)
            
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


class SwarmDiffusion:
    """
    Diffusion process for SwarmFormer models.
    Handles noise scheduling, sampling, and denoising.
    """
    def __init__(self, model, timesteps=20, beta_min=1e-4, beta_max=0.02):
        """
        Initialize the diffusion process.
        
        Args:
            model: The SwarmDiffusionModel for noise prediction
            timesteps: Number of diffusion steps
            beta_min: Minimum beta value for noise schedule
            beta_max: Maximum beta value for noise schedule
        """
        self.model = model
        self.timesteps = timesteps
        self.device = next(model.parameters()).device
        
        # Define beta schedule
        self.betas = torch.linspace(beta_min, beta_max, timesteps, device=self.device)
        
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
    
    def denoise(self, x_t, t):
        """
        Denoise a batch of noisy inputs at specified timesteps.
        
        Args:
            x_t: Noisy inputs [batch_size, input_dim]
            t: Time step indices [batch_size]
            
        Returns:
            Denoised prediction of x0
        """
        t = t.to(self.device)
        if t.dim() == 0:
            t = t.unsqueeze(0)
            
        # Get model's noise prediction
        with torch.no_grad():
            predicted_noise = self.model(x_t, t)
        
        # Get predicted x_0
        alpha_t = self.alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        
        predicted_x0 = (x_t - sqrt_one_minus_alpha_t * predicted_noise) / torch.sqrt(alpha_t)
        
        return predicted_x0
    
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


def create_swarm_diffusion(input_dim=256, hidden_dim=32, timesteps=20):
    """
    Create a complete SwarmDiffusion system with the winning architecture.
    
    Args:
        input_dim: Dimension of input data (default: 256 for 16x16 images)
        hidden_dim: Hidden dimension for model (default: 32)
        timesteps: Number of diffusion steps (default: 20)
        
    Returns:
        SwarmDiffusion object ready for use
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with the winning parameters
    model = SwarmDiffusionModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        cluster_size=16
    ).to(device)
    
    # Create diffusion process
    diffusion = SwarmDiffusion(
        model=model,
        timesteps=timesteps,
        beta_min=1e-4,
        beta_max=0.02
    )
    
    return diffusion 