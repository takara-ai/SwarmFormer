"""
Convert SmolLM3ForCausalLM weights to CausalSwarmFormerForLM.

Memory-efficient: loads one safetensors shard at a time, copies needed
weights directly into model parameters, then deletes the shard.
Model is created in bfloat16 from the start.
"""

import argparse
import gc
import json
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from .config import CausalSwarmFormerConfig
from .causal_model import CausalSwarmFormerForLM


WEIGHT_MAP_GLOBAL = {
    "model.embed_tokens.weight": "embed_tokens.weight",
    "model.norm.weight": "norm.weight",
}


def get_layer_weight_map(src_idx: int, dst_idx: int) -> dict[str, str]:
    src = f"model.layers.{src_idx}"
    dst = f"layers.{dst_idx}"
    return {
        f"{src}.input_layernorm.weight": f"{dst}.input_layernorm.weight",
        f"{src}.self_attn.q_proj.weight": f"{dst}.attn.q_proj.weight",
        f"{src}.self_attn.k_proj.weight": f"{dst}.attn.k_proj.weight",
        f"{src}.self_attn.v_proj.weight": f"{dst}.attn.v_proj.weight",
        f"{src}.self_attn.o_proj.weight": f"{dst}.attn.o_proj.weight",
        f"{src}.post_attention_layernorm.weight": f"{dst}.post_attention_layernorm.weight",
        f"{src}.mlp.gate_proj.weight": f"{dst}.mlp.gate_proj.weight",
        f"{src}.mlp.up_proj.weight": f"{dst}.mlp.up_proj.weight",
        f"{src}.mlp.down_proj.weight": f"{dst}.mlp.down_proj.weight",
    }


def _set_param(model, name: str, tensor: torch.Tensor):
    """Set a parameter by dotted name directly on the model, no state_dict copy."""
    parts = name.split(".")
    obj = model
    for part in parts[:-1]:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    leaf = parts[-1]
    param = getattr(obj, leaf)
    if isinstance(param, torch.nn.Parameter):
        param.data.copy_(tensor)
    else:
        setattr(obj, leaf, tensor)


def convert_smollm3(
    hf_model_id: str = "HuggingFaceTB/SmolLM3-3B",
    num_layers: int = 4,
    output_path: str | None = None,
    device: str = "cpu",
) -> tuple[CausalSwarmFormerForLM, object]:
    """
    Load SmolLM3 weights from safetensors into CausalSwarmFormerForLM.

    Memory strategy: create model in bf16, load one shard at a time,
    copy needed weights, delete shard, repeat.
    """
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

    # Build weight mapping
    full_map: dict[str, str] = dict(WEIGHT_MAP_GLOBAL)
    for i in range(num_layers):
        full_map.update(get_layer_weight_map(src_idx=i, dst_idx=i))

    # Create model in bfloat16 from the start
    config = CausalSwarmFormerConfig(num_layers=num_layers)
    with torch.device("cpu"):
        torch.set_default_dtype(torch.bfloat16)
        model = CausalSwarmFormerForLM(config)
        torch.set_default_dtype(torch.float32)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"CausalSwarmFormer: {total_params:,} params ({num_layers}L, bf16)")

    # Download safetensors index
    print(f"Loading weights from {hf_model_id}...")
    index_path = hf_hub_download(hf_model_id, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Group needed weights by shard
    needed_shards: dict[str, list[str]] = {}
    for src_name in full_map:
        if src_name in weight_map:
            shard = weight_map[src_name]
            needed_shards.setdefault(shard, []).append(src_name)

    # Load -> copy -> delete, one shard at a time
    mapped_count = 0
    for shard_file, src_names in sorted(needed_shards.items()):
        print(f"  Shard {shard_file}: {len(src_names)} weights...", end=" ", flush=True)
        shard_path = hf_hub_download(hf_model_id, shard_file)
        shard_data = load_file(shard_path)

        for src_name in src_names:
            dst_name = full_map[src_name]
            src_tensor = shard_data[src_name]
            _set_param(model, dst_name, src_tensor.to(torch.bfloat16))
            mapped_count += 1

        # Nuke shard from memory
        del shard_data
        gc.collect()
        print("done", flush=True)

    new_count = len([n for n in model.state_dict() if n not in full_map.values()])
    print(f"\nMapped: {mapped_count} | New (near-identity): {new_count}")

    if output_path:
        torch.save(model.state_dict(), output_path)
        print(f"Saved to {output_path}")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Convert SmolLM3 to CausalSwarmFormer")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM3-3B")
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model, tokenizer = convert_smollm3(
        hf_model_id=args.model_id,
        num_layers=args.num_layers,
        output_path=args.output,
    )
    model = model.to(args.device)
    model.eval()

    test_input = tokenizer("Hello, world!", return_tensors="pt", padding="max_length",
                           max_length=model.config.cluster_size, truncation=True)
    input_ids = test_input["input_ids"].to(args.device)

    with torch.no_grad():
        result = model(input_ids)

    print(f"Sanity: {input_ids.shape} -> {result['logits'].shape} OK")


if __name__ == "__main__":
    main()
