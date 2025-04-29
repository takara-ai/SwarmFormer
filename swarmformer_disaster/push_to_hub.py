#!/usr/bin/env python3
"""
push_to_hub.py

Load the best student model (full weights), apply dynamic int8 quantization, and push to Hugging Face Hub as safetensors.
"""
import torch
from transformers import AutoTokenizer
from swarmformer.model import SwarmFormerModel

def main():
    # Repo configuration
    repo_id = "takara-ai/SF-DD-6M"
    private = True

    # Tokenizer for vocab size
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    vocab_size = tokenizer.vocab_size

    # Build SwarmFormer student with correct architecture
    model = SwarmFormerModel(
        vocab_size=vocab_size,
        seq_len=64,
        d_model=128,
        num_layers=2,
        cluster_size=4,
        T_local=2
    )

    # Load full student checkpoint
    print("Loading student_model.pt...")
    state = torch.load("student_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Push to Hugging Face Hub (safetensors by default via ModelHubMixin)
    print(f"Pushing to hub: {repo_id} (private={private})...")
    model.push_to_hub(repo_id, private=private)
    print(f"✅ Successfully pushed {repo_id} (private={private})")

if __name__ == "__main__":
    main() 