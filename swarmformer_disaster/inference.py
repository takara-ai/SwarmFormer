#!/usr/bin/env python3
"""
inference.py

Load the SwarmFormer student model from Hugging Face Hub and classify input text.
"""
import argparse
import torch
from transformers import AutoTokenizer
from swarmformer.model import SwarmFormerModel

def classify_text(model, tokenizer, text, seq_len):
    # Tokenize and prepare inputs
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    padding="max_length", max_length=seq_len)
    input_ids = enc["input_ids"]
    # Clamp any OOV ids to vocab range
    if hasattr(model, 'embedding') and hasattr(model.embedding, 'embed'):
        vs = model.embedding.embed.num_embeddings
        input_ids = input_ids.clamp(0, vs-1)
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
        # Model returns logits directly
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
        pred = torch.argmax(logits, dim=-1).item()
    return pred


def main():
    parser = argparse.ArgumentParser(description="SwarmFormer inference from HF Hub")
    parser.add_argument("--text", type=str, required=False,
                        default="M 5.2 earthquake near Tokyo",
                        help="Text to classify (default: example earthquake)")
    args = parser.parse_args()

    # Load model from Hugging Face Hub
    print("Loading model takara-ai/SF-DD-6M from Hub...")
    model = SwarmFormerModel.from_pretrained("takara-ai/SF-DD-6M")
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    # Use the same max length as during training
    seq_len = 64

    # Classify
    pred = classify_text(model, tokenizer, args.text, seq_len)
    label = "DISASTER" if pred == 1 else "non-disaster"
    print(f"Input: {args.text}\nPrediction: {label}")


if __name__ == '__main__':
    main() 