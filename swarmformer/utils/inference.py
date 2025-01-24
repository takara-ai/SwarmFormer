"""
Utilities for running inference with SwarmFormer models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from ..config.model_configs import ModelConfig
from ..models.classification import SwarmFormerModel, IMDbDataset

def get_device():
    """Get the device to use for inference"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_trained_model(config: ModelConfig, device: str = 'cuda'):
    """Load the trained model with given configuration"""
    # Set seeds and environment variables to match training
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Create test dataset with configuration
    test_dataset = IMDbDataset(
        split='test',
        seq_len=config.seq_len,
        max_samples=25000,
        seed=42,
        augment=False
    )
    
    # Initialize model with configuration
    model = SwarmFormerModel(
        vocab_size=test_dataset.vocab_size(),
        d_model=config.d_model,
        seq_len=config.seq_len,
        cluster_size=config.cluster_size,
        num_layers=config.num_layers,
        T_local=config.T_local
    ).to(device)
    
    # Turn off dropout for inference
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    
    # Download and load model weights from HuggingFace
    try:
        checkpoint_path = hf_hub_download(
            repo_id=config.hf_model_id,
            filename="model.safetensors"
        )
        state_dict = load_file(checkpoint_path)  # Load safetensors file
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from HuggingFace: {str(e)}")
    
    return model, test_dataset

def evaluate_model(model, test_dataset, batch_size, device='cuda'):
    """Evaluate the model on the test dataset"""
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_labels = []
    latencies = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Running inference", unit="batch"):
            x, y = x.to(device), y.to(device)
            start_time = time.time()
            logits = model(x)
            latencies.append((time.time() - start_time) * 1000)  # ms
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    # Calculate latency statistics
    latency_stats = {
        'mean_batch_ms': np.mean(latencies),
        'mean_per_sample_ms': np.mean(latencies) / batch_size,
        'p95_ms': np.percentile(latencies, 95),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies)
    }
    
    # Calculate throughput
    total_time = sum(latencies) / 1000  # seconds
    total_samples = len(all_labels)
    
    throughput_stats = {
        'samples_per_second': total_samples / total_time,
        'total_samples': total_samples,
        'processed_samples': len(all_preds),
        'total_time_seconds': total_time
    }
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'latency': latency_stats,
        'throughput': throughput_stats
    }

def count_parameters(model):
    """Count total and trainable parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params 