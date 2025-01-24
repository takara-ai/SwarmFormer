import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import random
import os
import nltk
import joblib
from nltk.tokenize import sent_tokenize
from typing import List, Tuple
import optuna
from optuna.trial import Trial
import pathlib
from datetime import datetime
import pickle
import psutil
import time
from torch.cuda import max_memory_allocated, reset_peak_memory_stats
import signal
import sys
from torch import autocast, inference_mode
from torch.amp.grad_scaler import GradScaler
import torch.backends.cudnn as cudnn

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class IMDbAugmenter:
    def __init__(self):
        # Download all required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('wordnet')
            nltk.download('omw-1.4')  # Open Multilingual Wordnet
        
        from nltk.corpus import wordnet
        self.wordnet = wordnet
        
    def _shuffle_sentences(self, text: str) -> str:
        """Shuffle sentences while maintaining some local structure"""
        sentences = sent_tokenize(text)
        if len(sentences) <= 2:
            return text
            
        # Keep some sentences together to preserve local context
        chunks = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and random.random() < 0.7:  # 70% chance to keep pairs
                chunks.append(sentences[i:i+2])
                i += 2
            else:
                chunks.append([sentences[i]])
                i += 1
                
        random.shuffle(chunks)
        return ' '.join([' '.join(chunk) for chunk in chunks])
    
    def _swap_synonyms(self, text: str, p: float = 0.1) -> str:
        """Replace words with synonyms while preserving sentiment"""
        words = text.split()
        for i in range(len(words)):
            if random.random() < p:
                synonyms = []
                for syn in self.wordnet.synsets(words[i]):
                    for lemma in syn.lemmas():
                        if lemma.name() != words[i]:
                            synonyms.append(lemma.name())
                if synonyms:
                    words[i] = random.choice(synonyms)
        return ' '.join(words)
    
    def _create_hierarchical_sample(self, texts: List[str]) -> str:
        """Combine multiple reviews in a hierarchical way"""
        # Select 2-3 reviews
        num_reviews = random.randint(2, 3)
        selected = random.sample(texts, num_reviews)
        
        # Create a hierarchical structure
        result = []
        for i, text in enumerate(selected):
            prefix = random.choice([
                "Additionally, ", "Furthermore, ", "Moreover, ",
                "In contrast, ", "Similarly, ", "On the other hand, "
            ]) if i > 0 else ""
            result.append(prefix + text)
            
        return " ".join(result)
    
    def augment_dataset(self, texts: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
        """Apply multiple augmentation strategies"""
        augmented_texts = []
        augmented_labels = []
        
        # Original samples
        augmented_texts.extend(texts)
        augmented_labels.extend(labels)
        
        # Group texts by sentiment
        pos_texts = [t for t, l in zip(texts, labels) if l == 1]
        neg_texts = [t for t, l in zip(texts, labels) if l == 0]
        
        # 1. Sentence shuffling
        for text, label in zip(texts, labels):
            if random.random() < 0.3:  # 30% chance
                augmented_texts.append(self._shuffle_sentences(text))
                augmented_labels.append(label)
        
        # 2. Synonym replacement
        for text, label in zip(texts, labels):
            if random.random() < 0.3:
                augmented_texts.append(self._swap_synonyms(text))
                augmented_labels.append(label)
        
        # 3. Hierarchical combinations
        for _ in range(len(texts) // 4):  # Add 25% more samples
            if random.random() < 0.5:
                text = self._create_hierarchical_sample(pos_texts)
                augmented_texts.append(text)
                augmented_labels.append(1)
            else:
                text = self._create_hierarchical_sample(neg_texts)
                augmented_texts.append(text)
                augmented_labels.append(0)
        
        return augmented_texts, augmented_labels

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
        
        # Apply augmentation for training set
        if split == 'train' and augment:
            augmenter = IMDbAugmenter()
            texts, labels = augmenter.augment_dataset(texts, labels)
        
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
        pooled = self.dropout_final(pooled)  # Add dropout before classification
        logits = self.classifier(pooled)
        return logits

def create_datasets(seq_len, batch_size):
    """Create datasets and dataloaders once"""
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Remove this line
    
    # Create dataset directory if it doesn't exist
    dataset_dir = pathlib.Path("ablations/datasets")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate cache filename based on parameters
    cache_file = dataset_dir / f"dataset_cache_sl{seq_len}.pkl"
    
    if cache_file.exists():
        print(f"Loading cached datasets for seq_len={seq_len}")
        with open(cache_file, 'rb') as f:
            train_dataset, val_dataset = pickle.load(f)
    else:
        print(f"Creating new datasets for seq_len={seq_len}")
        train_dataset = IMDbDataset(split='train', seq_len=seq_len, max_samples=25000, augment=True)
        val_dataset = IMDbDataset(split='test', seq_len=seq_len, max_samples=25000, augment=False)
        
        # Cache the datasets
        with open(cache_file, 'wb') as f:
            pickle.dump((train_dataset, val_dataset), f)
    
    # Create dataloaders with optimized settings
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,  # Prefetch 2 batches per worker
        persistent_workers=True,  # Keep workers alive between epochs
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size * 2,  # Can use larger batch size for validation
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    
    return train_dataset, val_dataset, train_loader, val_loader

def create_dataset_for_config(config, max_samples=25000):
    """Helper function to create datasets for a single configuration"""
    seq_len, batch_size = config
    print(f"Creating datasets for seq_len={seq_len}, batch_size={batch_size}")
    return (seq_len, batch_size), create_datasets(seq_len, batch_size)

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_flops(model, input_size):
    """Estimate FLOPs for a single forward pass"""
    from thop import profile
    
    # Create dummy input
    dummy_input = torch.randint(0, 1000, (1, input_size)).to(next(model.parameters()).device)
    
    # Calculate FLOPs
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    return flops

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process()
    cpu_mem = process.memory_info().rss / 1024 / 1024  # MB
    gpu_mem = max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0  # MB
    return cpu_mem, gpu_mem

def train_SwarmFormer_model(params, trial):
    # Enable cuDNN benchmarking and deterministic mode
    cudnn.benchmark = True
    cudnn.deterministic = True
    
    if torch.cuda.is_available():
        reset_peak_memory_stats()
        # Set higher memory allocation fraction
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available VRAM
    
    # Create checkpoint directory
    checkpoint_dir = pathlib.Path("ablations/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Get cached datasets
    train_dataset, val_dataset, train_loader, val_loader = dataset_cache[
        params['seq_len']][params['batch_size']]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Update model initialization with new dropout value
    model = SwarmFormerModel(
        vocab_size=train_dataset.vocab_size(),
        d_model=params['d_model'],
        seq_len=params['seq_len'],
        cluster_size=params['cluster_size'],
        num_layers=params['num_layers'],
        T_local=params['T_local']
    ).to(device)
    
    # Count parameters and add to params
    params['num_parameters'] = count_parameters(model)
    
    # Update dropout values
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = params['dropout']
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    
    # Add model architecture validation
    assert params['seq_len'] % params['cluster_size'] == 0, "seq_len must be divisible by cluster_size"
    assert params['d_model'] >= 64, "d_model should be large enough for meaningful representations"
    
    # Improved early stopping strategy
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    min_epochs = 5
    
    # Generate unique model identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = f"swarmformer_{timestamp}_trial{trial.number}"
    
    # Keep track of top 3 models
    top_checkpoints = []
    max_checkpoints = 3
    
    # Track compute metrics
    start_time = time.time()
    total_training_samples = 0
    total_forward_passes = 0
    peak_cpu_mem = 0
    peak_gpu_mem = 0
    
    # Calculate FLOPs for one forward pass
    flops_per_forward = count_flops(model, params['seq_len'])
    
    # Create gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Training loop with validation
    for epoch in range(20):
        epoch_start = time.time()
        
        model.train()
        train_loss = 0
        train_steps = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            # Use new autocast API
            with autocast(device_type=device.type):
                logits = model(x)
                loss = criterion(logits, y)
            
            # Scale loss and backprop
            scaler.scale(loss).backward()
            
            # Unscale gradients and clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Step optimizer and update scaler
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_steps += 1
            
            # Track memory usage
            cpu_mem, gpu_mem = get_memory_usage()
            peak_cpu_mem = max(peak_cpu_mem, cpu_mem)
            peak_gpu_mem = max(peak_gpu_mem, gpu_mem)
        
        avg_train_loss = train_loss / train_steps
        
        # Validation (using inference mode and autocast)
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        val_steps = 0
        
        with inference_mode(), autocast(device_type=device.type):
            for x_val, y_val in val_loader:
                x_val = x_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)
                val_logits = model(x_val)
                val_loss += criterion(val_logits, y_val).item()
                preds = torch.argmax(val_logits, dim=-1)
                val_correct += (preds == y_val).sum().item()
                val_total += y_val.size(0)
                val_steps += 1
        
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / val_steps
        
        # Report metrics to Optuna
        trial.report(val_acc, epoch)
        
        # Checkpoint saving logic
        if val_acc > best_val_acc or (len(top_checkpoints) < max_checkpoints and 
            (not top_checkpoints or val_acc > top_checkpoints[-1][0])):
            
            checkpoint_path = checkpoint_dir / f"{model_id}_ep{epoch}_acc{val_acc:.4f}.pt"
            
            # Save model checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': avg_val_loss,
                'params': params,
                'trial_number': trial.number
            }
            
            torch.save(checkpoint, checkpoint_path)
            
            # Update top checkpoints list
            top_checkpoints.append((val_acc, checkpoint_path))
            top_checkpoints.sort(reverse=True)  # Sort by validation accuracy
            
            # Remove excess checkpoints
            while len(top_checkpoints) > max_checkpoints:
                _, old_checkpoint_path = top_checkpoints.pop()
                if old_checkpoint_path.exists():
                    old_checkpoint_path.unlink()  # Delete old checkpoint
        
        # Enhanced early stopping logic
        if epoch >= min_epochs:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        # Handle pruning
        if trial.should_prune():
            # Clean up checkpoints if trial is pruned
            for _, checkpoint_path in top_checkpoints:
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
            raise optuna.TrialPruned()
    
    # Calculate training statistics
    training_time = time.time() - start_time
    total_flops = flops_per_forward * total_forward_passes
    
    # Update params with resource usage before returning
    params.update({
        'training_time': training_time,
        'samples_per_second': total_training_samples / training_time,
        'peak_cpu_memory_mb': peak_cpu_mem,
        'peak_gpu_memory_mb': peak_gpu_mem,
        'total_flops': total_flops,
        'flops_per_forward': flops_per_forward,
        'total_forward_passes': total_forward_passes,
        'num_parameters': count_parameters(model)  # Make sure this is recorded
    })
    
    return best_val_acc

def generate_interim_report(study, ablations_dir, interrupted=True):
    """Generate a report of the study progress so far"""
    if not study.trials:
        print("\nNo trials completed yet.")
        return
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    status = "interrupted" if interrupted else "interim"
    study_name = f"swarmformer_study_{timestamp}_{status}"
    
    print("\nGenerating interim report...")
    print(f"\nTrials completed: {len(study.trials)}")
    
    if study.best_trial:
        print("\nBest trial so far:")
        print(f"  Value: {study.best_trial.value:.4f}")
        print("\nBest parameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
    
    # Save the current state
    study_path = ablations_dir / "studies" / f"{study_name}.pkl"
    joblib.dump(study, study_path)
    print(f"\nPartial study saved to {study_path}")
    
    # Create detailed report
    try:
        study_trials = study.trials_dataframe()
        
        # Add analysis columns
        study_trials['duration'] = study_trials['datetime_complete'] - study_trials['datetime_start']
        study_trials['timestamp'] = study_trials['datetime_complete'].dt.strftime('%Y%m%d_%H%M%S')
        
        # Define valid_configs here
        valid_configs = []
        seq_lengths = [64, 128, 256, 384, 512, 768]
        cluster_sizes = [2, 4, 8, 12, 16]
        for cluster_size in cluster_sizes:
            for seq_len in seq_lengths:
                if seq_len % cluster_size == 0:
                    valid_configs.append((cluster_size, seq_len))
        
        # Add all metrics including cluster_size and seq_len
        metrics = {
            'num_parameters': 0,
            'training_time': 0,
            'samples_per_second': 0,
            'peak_cpu_memory_mb': 0,
            'peak_gpu_memory_mb': 0,
            'total_flops': 0,
            'flops_per_forward': 0,
            'total_forward_passes': 0
        }
        
        # Add each metric to the dataframe, with proper error handling
        for metric in metrics:
            study_trials[metric] = [t.params.get(metric, 0) if t.params else 0 for t in study.trials]
        
        # Add cluster_size and seq_len
        study_trials['cluster_size'] = [valid_configs[t.params['config_idx']][0] if t.params and 'config_idx' in t.params else None 
                                      for t in study.trials]
        study_trials['seq_len'] = [valid_configs[t.params['config_idx']][1] if t.params and 'config_idx' in t.params else None 
                                  for t in study.trials]
        
        # Save detailed report
        report_path = ablations_dir / "reports" / f"{study_name}_report.csv"
        study_trials.to_csv(report_path, index=False)
        print(f"\nDetailed report saved to {report_path}")
        
    except Exception as e:
        print(f"\nError generating detailed report: {e}")
    
    print("\nExiting gracefully...")

if __name__ == "__main__":
    # Create ablations directory and subdirectories
    ablations_dir = pathlib.Path("ablations")
    ablations_dir.mkdir(exist_ok=True)
    
    for subdir in ["checkpoints", "studies", "reports", "datasets"]:
        (ablations_dir / subdir).mkdir(exist_ok=True)
    
    # Create datasets for each sequence length once
    dataset_cache = {}
    seq_lengths = {64, 128, 256, 384, 512, 768}
    batch_sizes = {32, 48, 64, 96, 128, 160}
    
    # First, ensure all datasets are created and cached
    print("Ensuring all datasets are created and cached...")
    for seq_len in seq_lengths:
        # We only need to create the dataset once per sequence length
        # The batch size only affects the dataloader
        train_dataset, val_dataset = None, None
        cache_file = ablations_dir / "datasets" / f"dataset_cache_sl{seq_len}.pkl"
        
        if cache_file.exists():
            print(f"Loading cached datasets for seq_len={seq_len}")
            with open(cache_file, 'rb') as f:
                train_dataset, val_dataset = pickle.load(f)
        else:
            print(f"Creating new datasets for seq_len={seq_len}")
            train_dataset = IMDbDataset(split='train', seq_len=seq_len, max_samples=25000, augment=True)
            val_dataset = IMDbDataset(split='test', seq_len=seq_len, max_samples=25000, augment=False)
            
            # Cache the datasets
            with open(cache_file, 'wb') as f:
                pickle.dump((train_dataset, val_dataset), f)
        
        # Create dataloaders for each batch size
        dataset_cache[seq_len] = {}
        for batch_size in batch_sizes:
            train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            dataset_cache[seq_len][batch_size] = (train_dataset, val_dataset, train_loader, val_loader)
    
    print("Dataset preprocessing complete!")

    def objective(trial: Trial):
        # Pre-defined valid configurations - adjusted for SwarmFormer architecture
        # Each sequence length must be divisible by its cluster size
        valid_configs = []
        
        # All possible sequence lengths and cluster sizes
        seq_lengths = [64, 128, 256, 384, 512, 768]
        cluster_sizes = [2, 4, 8, 12, 16]
        
        # Only add configurations where seq_len is divisible by cluster_size
        for cluster_size in cluster_sizes:
            for seq_len in seq_lengths:
                if seq_len % cluster_size == 0:
                    valid_configs.append((cluster_size, seq_len))
        
        # Print valid configurations for verification
        print("\nValid configurations:")
        for i, (cluster_size, seq_len) in enumerate(valid_configs):
            print(f"Config {i}: cluster_size={cluster_size}, seq_len={seq_len}")
        
        config_idx = trial.suggest_categorical('config_idx', list(range(len(valid_configs))))
        cluster_size, seq_len = valid_configs[config_idx]
        
        # Hyperparameters optimized for SwarmFormer architecture
        params = {
            'seq_len': seq_len,
            'cluster_size': cluster_size,
            # d_model should be large enough for meaningful cluster representations
            'd_model': trial.suggest_categorical('d_model', [64, 96, 128, 160, 192]),
            # More layers allow for deeper hierarchical processing
            'num_layers': trial.suggest_categorical('num_layers', [2, 3, 4]),
            # T_local affects local information propagation
            'T_local': trial.suggest_categorical('T_local', [2, 3, 4, 5]),
            # Larger batch sizes for better cluster statistics
            'batch_size': trial.suggest_categorical('batch_size', [32, 48, 64, 96, 128, 160]),
            # Learning rate adjusted for stable training of attention mechanisms
            'learning_rate': trial.suggest_float('learning_rate', 5e-5, 5e-4, log=True),
            # Weight decay to prevent over-fitting in attention layers
            'weight_decay': trial.suggest_float('weight_decay', 0.02, 0.15, log=True),
            # Dropout values for regularization
            'dropout': trial.suggest_float('dropout', 0.2, 0.5, step=0.1)
        }

        return train_SwarmFormer_model(params, trial)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"swarmformer_study_{timestamp}"
    
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=3
        ),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Add signal handler
    def signal_handler(signum, frame):
        print("\n\nInterrupted by user. Generating final report...")
        generate_interim_report(study, ablations_dir)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        study.optimize(
            objective, 
            n_trials=50, 
            timeout=72000,
            catch=(RuntimeError, ValueError)
        )
    except Exception as e:
        print(f"\nStudy interrupted by error: {e}")
        generate_interim_report(study, ablations_dir, interrupted=False)
        raise
    
    print("\nBest trial:")
    trial = study.best_trial
    
    print(f"  Value: {trial.value:.4f}")
    print("\nBest parameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    
    # Save study results
    study_path = ablations_dir / "studies" / f"{study_name}.pkl"
    joblib.dump(study, study_path)
    print(f"\nStudy saved to {study_path}")
    
    # Create detailed report
    study_trials = study.trials_dataframe()
    
    # Add additional analysis columns
    study_trials['duration'] = study_trials['datetime_complete'] - study_trials['datetime_start']
    study_trials['timestamp'] = study_trials['datetime_complete'].dt.strftime('%Y%m%d_%H%M%S')
    study_trials['num_parameters'] = [t.params.get('num_parameters', 0) for t in study.trials]
    
    # Add resource usage columns
    study_trials['training_time'] = [t.params.get('training_time', 0) for t in study.trials]
    study_trials['samples_per_second'] = [t.params.get('samples_per_second', 0) for t in study.trials]
    study_trials['peak_cpu_memory_mb'] = [t.params.get('peak_cpu_memory_mb', 0) for t in study.trials]
    study_trials['peak_gpu_memory_mb'] = [t.params.get('peak_gpu_memory_mb', 0) for t in study.trials]
    study_trials['total_flops'] = [t.params.get('total_flops', 0) for t in study.trials]
    study_trials['flops_per_forward'] = [t.params.get('flops_per_forward', 0) for t in study.trials]
    
    # Add cluster_size and seq_len to the report
    study_trials['cluster_size'] = [valid_configs[t.params['config_idx']][0] if t.params.get('config_idx') is not None else None 
                                  for t in study.trials]
    study_trials['seq_len'] = [valid_configs[t.params['config_idx']][1] if t.params.get('config_idx') is not None else None 
                              for t in study.trials]
    
    # Update columns order to include cluster_size and seq_len
    columns_order = [
        'number', 'value', 'state', 'timestamp', 'duration',
        'cluster_size', 'seq_len',  # Added these two columns
        'num_parameters', 'training_time', 'samples_per_second',
        'peak_cpu_memory_mb', 'peak_gpu_memory_mb',
        'total_flops', 'flops_per_forward',
        'params_d_model', 'params_num_layers',
        'params_T_local', 'params_batch_size', 'params_learning_rate',
        'params_weight_decay', 'params_dropout'
        # Removed params_config_idx since we now have the actual values
    ]
    columns_order.extend([col for col in study_trials.columns if col not in columns_order])
    study_trials = study_trials[columns_order]
    
    # Save detailed report
    report_path = ablations_dir / "reports" / f"{study_name}_report.csv"
    study_trials.to_csv(report_path, index=False)
    print(f"Detailed report saved to {report_path}")
    
    # Save summary report with parameter count
    summary_data = {
        'study_name': study_name,
        'n_trials': len(study.trials),
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_model_parameters': study.best_trial.params.get('num_parameters', 0),
        'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'start_time': study.trials[0].datetime_start.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': study.trials[-1].datetime_complete.strftime('%Y-%m-%d %H:%M:%S'),
        'parameter_statistics': {
            'min': min(t.params.get('num_parameters', 0) for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
            'max': max(t.params.get('num_parameters', 0) for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
            'mean': sum(t.params.get('num_parameters', 0) for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE) / 
                   len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        },
        'resource_statistics': {
            'avg_training_time': np.mean([t.params.get('training_time', 0) 
                for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'avg_samples_per_second': np.mean([t.params.get('samples_per_second', 0) 
                for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'peak_cpu_memory_mb': max(t.params.get('peak_cpu_memory_mb', 0) 
                for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
            'peak_gpu_memory_mb': max(t.params.get('peak_gpu_memory_mb', 0) 
                for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
            'avg_flops_per_forward': np.mean([t.params.get('flops_per_forward', 0) 
                for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'total_compute_time': sum(t.params.get('training_time', 0) 
                for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        }
    }
    
    # Save summary report with pretty formatting
    summary_path = ablations_dir / "reports" / f"{study_name}_summary.txt"
    with open(summary_path, 'w') as f:
        for key, value in summary_data.items():
            if key in ['parameter_statistics', 'resource_statistics']:
                f.write(f"\n{key.replace('_', ' ').title()}:\n")
                for stat_key, stat_value in value.items():
                    if 'memory' in stat_key:
                        f.write(f"  {stat_key}: {stat_value:.2f} MB\n")
                    elif 'time' in stat_key:
                        f.write(f"  {stat_key}: {stat_value:.2f} seconds\n")
                    elif 'flops' in stat_key:
                        f.write(f"  {stat_key}: {stat_value:,.0f} FLOPs\n")
                    elif isinstance(stat_value, (int, float)):
                        f.write(f"  {stat_key}: {stat_value:,.2f}\n")
            elif key == 'best_model_parameters':
                f.write(f"{key}: {value:,} parameters\n")
            else:
                f.write(f"{key}: {value}\n")
    print(f"Summary report saved to {summary_path}")
