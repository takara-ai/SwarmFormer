#!/usr/bin/env python3
"""
prune_student.py

Prunes the SwarmFormer student model globally and saves the pruned checkpoint.
"""
import os
import argparse
import torch
import torch.nn.utils.prune as prune
import torch.jit
from swarmformer.model import SwarmFormerModel
from transformers import AutoTokenizer

def get_device():
    # We perform pruning on CPU for checkpoint simplicity
    return torch.device('cpu')


def prune_model(model: torch.nn.Module, amount: float) -> torch.nn.Module:
    """
    Applies global unstructured L1 pruning to all Linear layer weights in the model.
    """
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    # Global unstructured pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    # Remove pruning reparameterization to make weights permanent
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')
    return model


def main():
    parser = argparse.ArgumentParser(description="Prune SwarmFormer student model globally.")
    parser.add_argument('--student_save_path', type=str, default='student_model.pt', help='Path to the distilled student model checkpoint')
    parser.add_argument('--output_path', type=str, default='student_pruned.pt', help='Path to save the pruned student checkpoint')
    parser.add_argument('--script_output_path', type=str, default=None, help='Optional path to save TorchScript traced model')
    parser.add_argument('--prune_amount', type=float, default=0.2, help='Global fraction of weights to prune (0 < amount < 1)')
    parser.add_argument('--seq_len', type=int, default=64, help='Sequence length (must match original model)')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension (must match original)')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers (must match original)')
    parser.add_argument('--cluster_size', type=int, default=4, help='Cluster size (must match original)')
    parser.add_argument('--T_local', type=int, default=2, help='T_local parameter (must match original)')
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device} for pruning.")

    # Load tokenizer to get the original vocabulary size
    tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')
    vocab_size = tokenizer.vocab_size
    print(f"Loaded tokenizer; vocab_size={vocab_size}")

    # Instantiate model with correct vocab size
    model = SwarmFormerModel(
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        cluster_size=args.cluster_size,
        num_layers=args.num_layers,
        T_local=args.T_local
    ).to(device)

    # Load distilled student checkpoint
    if not os.path.exists(args.student_save_path):
        raise FileNotFoundError(f"Student checkpoint not found: {args.student_save_path}")
    state_dict = torch.load(args.student_save_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded student model from {args.student_save_path}.")

    # Apply pruning
    print(f"Applying global unstructured L1 pruning (amount={args.prune_amount})...")
    pruned_model = prune_model(model, args.prune_amount)

    # Save pruned checkpoint
    torch.save(pruned_model.state_dict(), args.output_path)
    print(f"Pruned model saved to {args.output_path}.")

    # Optionally, trace and save TorchScript model for production
    if args.script_output_path:
        print(f"Tracing the pruned model for TorchScript (this may take a moment)...")
        example_input = torch.randint(0, vocab_size, (1, args.seq_len), dtype=torch.long)
        pruned_model.eval()
        with torch.no_grad():
            scripted_model = torch.jit.trace(pruned_model, example_input)
        scripted_model.save(args.script_output_path)
        print(f"Scripted pruned model saved to {args.script_output_path}.")

if __name__ == '__main__':
    main() 