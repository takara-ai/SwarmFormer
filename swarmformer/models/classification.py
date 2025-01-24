"""
SwarmFormer model implementation for binary classification tasks.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
import random
from typing import List, Tuple
import os

from .layers import TokenEmbedding, SwarmFormerLayer

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', seq_len=16, max_samples=1000, seed=42, augment=True):
        super().__init__()
        random.seed(seed)
        
        # Load IMDb dataset
        dataset = load_dataset("imdb", split=split)
        
        # Sample if needed
        if max_samples and max_samples < len(dataset):
            indices = random.sample(range(len(dataset)), max_samples)
            dataset = dataset.select(indices)
        
        texts = [item['text'] for item in dataset]
        labels = [item['label'] for item in dataset]
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Process the texts
        self.data = []
        self.labels = []
        
        for text, label in zip(texts, labels):
            encoding = self.tokenizer(
                text,
                max_length=seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            self.data.append(encoding['input_ids'][0])
            self.labels.append(label)
        
        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def vocab_size(self):
        return self.tokenizer.vocab_size

class SwarmFormerModel(nn.Module):
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
        pooled = self.dropout_final(pooled)
        logits = self.classifier(pooled)
        return logits 