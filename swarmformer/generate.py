"""
Autoregressive generation with CausalSwarmFormerForLM.

Usage:
    python -m swarmformer.generate --prompt "Once upon a time" [--max-tokens 100]
"""

import argparse
import torch
from transformers import AutoTokenizer

from .config import CausalSwarmFormerConfig
from .causal_model import CausalSwarmFormerForLM
from .convert_smollm3 import convert_smollm3


def generate_text(
    model: CausalSwarmFormerForLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    device: str = "cpu",
) -> str:
    """Generate text from a prompt using the CausalSwarmFormer model."""
    model.eval()
    cluster_size = model.config.cluster_size

    # Tokenize prompt
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    # Pad to at least cluster_size (left-pad with pad token)
    if input_ids.shape[1] < cluster_size:
        pad_len = cluster_size - input_ids.shape[1]
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        input_ids = torch.cat([
            torch.full((1, pad_len), pad_id, dtype=input_ids.dtype, device=device),
            input_ids,
        ], dim=1)
        prompt_start = pad_len
    else:
        prompt_start = 0

    # Generate
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode only the non-padded portion
    generated_ids = output_ids[0, prompt_start:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Generate text with CausalSwarmFormer")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM3-3B")
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--checkpoint", default=None, help="Path to saved state_dict")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device

    if args.checkpoint:
        # Load from saved checkpoint
        config = CausalSwarmFormerConfig(num_layers=args.num_layers)
        model = CausalSwarmFormerForLM(config)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    else:
        # Convert from SmolLM3 on the fly
        model, tokenizer = convert_smollm3(
            hf_model_id=args.model_id,
            num_layers=args.num_layers,
            device=device,
        )

    model = model.to(device).to(torch.bfloat16)

    print(f"\nPrompt: {args.prompt}")
    print(f"Generating {args.max_tokens} tokens (temperature={args.temperature}, top_k={args.top_k})...\n")

    text = generate_text(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print(f"Generated:\n{text}")


if __name__ == "__main__":
    main()
