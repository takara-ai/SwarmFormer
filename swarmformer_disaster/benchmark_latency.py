# benchmark_latency.py
import torch
# Ensure CPU dynamic quantization engine is set
torch.backends.quantized.engine = 'qnnpack'
import time
import argparse
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from swarmformer.model import SwarmFormerModel # Assuming SwarmFormerModel is importable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import os
import gc # Garbage collector for memory management
from memory_profiler import memory_usage
import torch.nn as nn
import torch.quantization

def get_device():
    """Selects the best available device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("MPS backend is available. Using MPS.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("CUDA backend is available. Using CUDA.")
        return torch.device("cuda")
    else:
        print("MPS and CUDA not available. Using CPU.")
        return torch.device("cpu")

def prepare_dataloader(tokenizer, seq_len, batch_size):
    """Loads and preprocesses the disaster dataset, returns the test DataLoader."""
    print("Loading and preparing dataset...")
    raw_ds = load_dataset("disaster_response_messages", split="test")
    # Filter out ambiguous labels (related == 2)
    raw_ds = raw_ds.filter(lambda x: x['related'] != 2)
    print(f"Filtered dataset size: {len(raw_ds)}")

    def preprocess(batch):
        enc = tokenizer(batch['message'], truncation=True, padding='max_length', max_length=seq_len)
        # Convert 'related' (0 or 1) to labels
        labels = [1 if r == 1 else 0 for r in batch['related']]
        return {"input_ids": enc['input_ids'], "attention_mask": enc['attention_mask'], "label": labels}

    proc_ds = raw_ds.map(preprocess, batched=True, remove_columns=raw_ds.column_names)
    proc_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    print(f"Test dataset loaded with {len(proc_ds)} samples.")
    return DataLoader(proc_ds, batch_size=batch_size)

def measure_latency(model, dataloader, device, num_batches=20):
    """Measures average inference latency per sample over num_batches."""
    model.eval()
    latencies = []
    total_samples = 0
    print(f"Measuring latency over {num_batches} batches...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            input_ids = batch['input_ids'].to(device)
            # Clamp input_ids to avoid out-of-range indices
            if hasattr(model, 'embedding') and hasattr(model.embedding, 'embed'):
                vocab_size = model.embedding.embed.num_embeddings
                input_ids = input_ids.clamp(0, vocab_size - 1)
            attention_mask = batch['attention_mask'].to(device)
            batch_size = input_ids.shape[0]
            total_samples += batch_size

            # Warm-up GPU and synchronize
            if device.type == 'cuda': torch.cuda.synchronize()
            start_time = time.time()
            # Conditionally pass attention_mask
            if isinstance(model, SwarmFormerModel):
                _ = model(input_ids)
            else:
                _ = model(input_ids, attention_mask=attention_mask)
            # Synchronize GPU for timing
            if device.type == 'cuda': torch.cuda.synchronize()
            end_time = time.time()
            
            batch_latency_ms = (end_time - start_time) * 1000
            latencies.append(batch_latency_ms / batch_size) # Latency per sample

    avg_latency = np.mean(latencies)
    print(f"Average latency: {avg_latency:.3f} ms/sample (averaged over {total_samples} samples)")
    return avg_latency

def get_memory_usage(model, device, seq_len=None):
    """Measures peak memory usage for the model on the specified device. Use seq_len for CPU dummy input."""
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        _ = model(torch.zeros((1, 1), dtype=torch.long).to(device)) # Dummy forward pass
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        memory_metric_name = "Peak Allocated MB (cuda)"
        return peak_memory_bytes / (1024 * 1024), memory_metric_name
    elif device.type == 'mps':
        # memory_profiler is complex for MPS, estimate via current allocation
        # This is less accurate than CUDA peak memory but gives an indication.
        gc.collect()
        torch.mps.empty_cache()
        mem_bytes = torch.mps.current_allocated_memory()
        memory_metric_name = "Current Allocated MB (mps)"
        return mem_bytes / (1024 * 1024), memory_metric_name
    else: # CPU
        # Use memory_profiler for CPU, with correct seq_len if provided
        try:
            length = seq_len if seq_len is not None else 1
            dummy = torch.zeros((1, length), dtype=torch.long).to(device)
            mem_usage = memory_usage((model, (dummy,)), interval=0.1, max_usage=True)
            peak_memory_mb = max(mem_usage)
            memory_metric_name = "Peak Allocated MB (CPU - memory_profiler)"
            return peak_memory_mb, memory_metric_name
        except Exception as e:
            print(f"Could not use memory_profiler for CPU: {e}. Reporting 0.")
            return 0, "Peak Allocated MB (CPU - unknown)"

def count_parameters(model):
    """Counts the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model_performance(model, dataloader, device):
    """Evaluates model performance metrics (accuracy, F1, etc.) on the dataloader."""
    model.eval()
    all_preds = []
    all_labels = []
    print("Evaluating model performance...")
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            # Clamp input_ids to avoid out-of-range indices
            if hasattr(model, 'embedding') and hasattr(model.embedding, 'embed'):
                vocab_size = model.embedding.embed.num_embeddings
                input_ids = input_ids.clamp(0, vocab_size - 1)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Conditionally pass attention_mask
            if isinstance(model, SwarmFormerModel):
                outputs = model(input_ids)
            else:
                outputs = model(input_ids, attention_mask=attention_mask)

            # Handle different output types (dict vs tensor)
            if isinstance(outputs, dict):
                # Common for Hugging Face models (e.g., SequenceClassifierOutput)
                logits = outputs.get('logits')
            else:
                # Assuming raw logits tensor output
                logits = outputs

            if logits is None:
                raise ValueError("Could not extract logits from model output.")

            # Get predictions (assuming binary classification from logits)
            # Check if logits are already probabilities or need sigmoid/softmax
            if logits.shape[-1] == 1: # Single output neuron -> regression or binary with sigmoid applied elsewhere
                preds = (torch.sigmoid(logits) > 0.5).int().squeeze()
            elif logits.shape[-1] == 2: # Two output neurons -> binary classification with softmax/argmax
                preds = torch.argmax(logits, dim=-1)
            else:
                raise ValueError(f"Unexpected logit shape: {logits.shape}")

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary') # Ensure binary average
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    cm = confusion_matrix(all_labels, all_preds).tolist() # Convert to list for JSON

    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark model latency, memory, and performance.")
    # Model Selection Arguments
    parser.add_argument("--model_id", type=str, required=True, help="Unique ID for the model in the results JSON (e.g., 'swarmformer-finetuned').")
    parser.add_argument("--model_name", type=str, required=True, help="Display name for the model (e.g., 'SwarmFormer (Fine-tuned)').")
    parser.add_argument("--model_type", type=str, choices=['hf', 'custom'], default='custom', help="Type of model: 'hf' or 'custom'.")
    parser.add_argument("--hf_id", type=str, default=None, help="Hugging Face model ID (required if model_type is 'hf').")
    parser.add_argument("--model_load_path", type=str, default=None, help="Path to load custom model state_dict from.")

    # Common Arguments
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length for tokenization.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloader.")
    parser.add_argument("--latency_batches", type=int, default=20, help="Number of batches to average latency over.")
    parser.add_argument("--results_file", type=str, default="benchmark_results_comparison.json", help="JSON file to store results.")
    parser.add_argument("--quantize", action="store_true", help="Apply dynamic quantization to the model (CPU-only)")
    parser.add_argument("--quant_gpu", action="store_true", help="Apply FP16 quantization for GPU inference")

    # SwarmFormer Specific Arguments (required if model_type is 'custom')
    parser.add_argument("--d_model", type=int, default=128, help="SwarmFormer d_model.")
    parser.add_argument("--num_layers", type=int, default=2, help="SwarmFormer num_layers.")
    parser.add_argument("--cluster_size", type=int, default=4, help="SwarmFormer cluster_size.")
    parser.add_argument("--T_local", type=int, default=2, help="SwarmFormer T_local.")

    args = parser.parse_args()

    # Determine device and quantization
    if args.quantize:
        print("Dynamic quantization requested: using CPU.")
        device = torch.device('cpu')
    elif args.quant_gpu:
        # Allow FP16 inference on GPU (CUDA) or MPS
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        if not torch.cuda.is_available() and not mps_available:
            raise RuntimeError("No GPU or MPS device available for FP16 quantization.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        print(f"FP16 quantization requested: using {device} with half precision.")
    else:
        device = get_device()

    # --- Tokenizer (Use a consistent one, e.g., from the teacher) ---
    tokenizer_name = "answerdotai/ModernBERT-base" # Or another appropriate tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # --- Load Model ---
    model = None
    model_config = None # Store config for SwarmFormer
    precision = "torch.float32" # Assuming default precision

    if args.model_type == 'hf':
        if not args.hf_id:
            raise ValueError("Must provide --hf_id for model_type 'hf'.")
        print(f"Loading Hugging Face model: {args.hf_id}")
        model = AutoModelForSequenceClassification.from_pretrained(args.hf_id, num_labels=2).to(device)
    elif args.model_type == 'custom':
        print(f"Loading Custom SwarmFormer model...")
        model_config = {
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "cluster_size": args.cluster_size,
            "T_local": args.T_local
        }
        model = SwarmFormerModel(
            vocab_size=tokenizer.vocab_size,
            seq_len=args.seq_len,
            **model_config
        ).to(device)
        if args.model_load_path:
            print(f"Loading state dict from: {args.model_load_path}")
            if not os.path.exists(args.model_load_path):
                 raise FileNotFoundError(f"Model state dict not found at {args.model_load_path}")
            # Load state dict with appropriate map_location
            state_dict = torch.load(args.model_load_path, map_location=device)
            # Handle potential missing keys or size mismatches if necessary
            # Example: model.load_state_dict(state_dict, strict=False)
            model.load_state_dict(state_dict)
            print("State dict loaded successfully.")
        else:
            print("Warning: No --model_load_path provided for custom model. Using initialized weights.")

    if model is None:
        raise ValueError("Model could not be loaded.")

    model.eval() # Ensure model is in eval mode

    # Apply quantization
    if args.quantize:
        print("Applying dynamic quantization to Linear modules on CPU...")
        model.to(device)
        model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )
        print("Dynamic quantization applied.")
    elif args.quant_gpu:
        print(f"Converting model to half-precision for inference on {device}...")
        model = model.to(device).half()
        print(f"FP16 model ready on {device}.")

    # --- Data Loader ---
    test_loader = prepare_dataloader(tokenizer, args.seq_len, args.batch_size)

    # --- Benchmarking ---
    print(f"\n--- Benchmarking: {args.model_name} ({args.model_id}) ---")
    parameters = count_parameters(model)
    print(f"Parameters: {parameters}")

    avg_latency = measure_latency(model, test_loader, device, args.latency_batches)
    # Memory logging: file size for CPU quantized, profile for others
    if args.quantize and args.model_load_path:
        memory_mb = os.path.getsize(args.model_load_path) / (1024 * 1024)
        memory_metric_name = "File Size MB"
    else:
        memory_mb, memory_metric_name = get_memory_usage(model, device, args.seq_len)
    print(f"Memory ({memory_metric_name}): {memory_mb:.2f} MB")

    performance_metrics = evaluate_model_performance(model, test_loader, device)

    # --- Store Results ---
    results_data = {
        "benchmark_params": {
            "device": device.type,
            "test_samples": len(test_loader.dataset),
            "sequence_length": args.seq_len,
            "batch_size": args.batch_size,
            "latency_batches_averaged": args.latency_batches,
            "memory_metric": memory_metric_name
        },
        "models": {}
    }

    # Load existing results if file exists
    if os.path.exists(args.results_file):
        try:
            with open(args.results_file, 'r') as f:
                existing_data = json.load(f)
                # Keep existing benchmark params if they match, otherwise update
                if existing_data.get("benchmark_params") == results_data["benchmark_params"]:
                     results_data["benchmark_params"] = existing_data["benchmark_params"]
                results_data["models"] = existing_data.get("models", {}) # Preserve existing models
        except json.JSONDecodeError:
            print(f"Warning: Could not parse existing results file {args.results_file}. Overwriting.")

    # Quantization info
    quant_type = None
    if args.quantize:
        quant_type = 'dynamic int8 (CPU)'
    elif args.quant_gpu:
        quant_type = f'fp16 ({device.type.upper()})'
    # Add or update the current model's results
    results_data["models"][args.model_id] = {
        "name": args.model_name,
        "id": args.model_id,
        "type": args.model_type,
        "parameters": parameters,
        "precision": precision,
        "hf_id": args.hf_id, # Will be None for custom models
        "load_path": args.model_load_path,
        "config": model_config, # Store SwarmFormer config
        "avg_latency_ms_per_sample": avg_latency,
        "memory_mb": memory_mb,
        "performance": performance_metrics,
        "quantization": quant_type
    }

    # Save updated results
    try:
        with open(args.results_file, 'w') as f:
            json.dump(results_data, f, indent=4)
        print(f"\nBenchmark results saved to {args.results_file}")
    except IOError as e:
        print(f"Error saving results to {args.results_file}: {e}")

    # --- Print Summary Table ---
    print("\n--- Benchmark Summary ---")
    # Basic text table for console output
    print(f"{'Model Name':<25} {'Params (M)':<12} {'Latency (ms)':<14} {'Memory (MB)':<13} {'F1 Score':<10}")
    print("-" * 80)
    for model_id, data in results_data["models"].items():
        params_m = data['parameters'] / 1_000_000
        latency = data['avg_latency_ms_per_sample']
        memory = data['memory_mb']
        f1 = data['performance']['f1_score']
        print(f"{data['name']:<25} {params_m:<12.1f} {latency:<14.3f} {memory:<13.2f} {f1:<10.4f}")
    print("-" * 80)

if __name__ == "__main__":
    main() 