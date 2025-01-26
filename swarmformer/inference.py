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
from typing import Dict, Any, Tuple
from transformers import AutoTokenizer

from .model import SwarmFormerModel
from .dataset import IMDbDataset
from .config import ModelConfig

def load_trained_model(config: ModelConfig, device: str = 'cpu', include_dataset=False) -> Tuple[SwarmFormerModel, IMDbDataset]:
    """Load the trained model with given configuration"""
    # Set seeds and environment variables to match training
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if include_dataset:
        test_dataset = IMDbDataset(
            split='test',
            seq_len=config.seq_len,
            max_samples=25000,
            seed=42,
            augment=False
        )
        temp_vocab_size = test_dataset.vocab_size()
    
    if include_dataset == False:
        temp_vocab_size = 30522 # bert-base-uncased vocab size
        
    if config.seq_len % config.cluster_size != 0:
       raise ValueError("seq_len must be divisible by cluster_size.")
   
    model = SwarmFormerModel(
        vocab_size=temp_vocab_size,
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
    
    if include_dataset:
        return model, test_dataset
    else:
        return model

def evaluate_model(model: SwarmFormerModel, test_dataset: IMDbDataset, batch_size: int, device: str = 'cpu') -> Dict[str, Any]:
    """
    Evaluate a SwarmFormer-based model on a dataset (IMDb)
    Args:
        model (SwarmFormerModel): Trained SwarmFormer model
        test_dataset (IMDbDataset): 
        batch_size (int): REPLACE ME!!!
        device (str, optional): Device to run inference on. Defaults to 'cpu'.

    Returns:
        Dict[str, Any]: Evaluation results
    """

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

def inference(model: SwarmFormerModel, tokenizer: AutoTokenizer, text: str | list[str], device: str = 'cpu'):
    """
    Perform inference on either a single text or a list of texts using the SwarmFormer model.
    
    Args:
        model (SwarmFormerModel): Trained SwarmFormer model
        text (str or list[str]): Input text or list of texts for inference
        device (str, optional): Device to run inference on. Defaults to 'cpu'.
    
    Returns:
        torch.Tensor: Logits for the input text(s)
    """
    try:
        if model[1]:
            model = model[0]
    except:
        pass
    model.eval()
    
    if isinstance(text, str):
        text = [text]
    
    with torch.no_grad():
        tokenized_inputs = tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=model.seq_len,
            return_tensors='pt'
        ).to(device)
        
        inputs = tokenized_inputs['input_ids']

        seq_len = inputs.shape[1]
        
        if seq_len % model.cluster_size != 0:
            new_length = (seq_len // model.cluster_size + 1) * model.cluster_size
            padding_length = new_length - seq_len
            inputs = nn.functional.pad(inputs, (0, padding_length), value=tokenizer.pad_token_id)

        logits = model(inputs)
        probs = nn.functional.softmax(logits, dim=-1)
    
        predicted_class = torch.argmax(logits, dim=-1).item()
        predicted_prob = probs.max().item()

        class_labels = ['Negative', 'Positive']
        json_output = {
            'predicted_class': class_labels[predicted_class],
            'predicted_probability': predicted_prob
        }
    
    return logits, probs, json_output

def count_parameters(model: SwarmFormerModel) -> Tuple[int, int]:
    """Count total and trainable parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_device() -> torch.device:
    """Get the device to use for inference"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu") 