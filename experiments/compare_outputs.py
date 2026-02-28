"""
Compare SmolLM3 (original) vs CausalSwarmFormer (4-layer port) outputs.

Tests:
1. Next-token prediction agreement (top-k overlap)
2. Generated text comparison on several prompts
3. Per-token log-probability / perplexity on a reference sentence
"""

import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swarmformer.convert_smollm3 import convert_smollm3


PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a",
    "def fibonacci(n):\n    ",
    "In machine learning, transformers are",
    "The quick brown fox",
]

REFERENCE_SENTENCES = [
    "The Eiffel Tower is located in Paris, France and was built in 1889.",
    "Machine learning models require large amounts of data to train effectively.",
]

HF_MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
DEVICE = "cuda"


def load_smollm3(device):
    """Load original SmolLM3 for comparison."""
    print("Loading original SmolLM3...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID, dtype=torch.bfloat16, device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def load_swarmformer(device, num_layers=4):
    """Load CausalSwarmFormer with converted weights."""
    print(f"Loading CausalSwarmFormer ({num_layers} layers)...")
    t0 = time.time()
    model, tokenizer = convert_smollm3(
        hf_model_id=HF_MODEL_ID, num_layers=num_layers, device=device,
    )
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def get_next_token_probs(model, input_ids, is_swarmformer=False):
    """Get probability distribution over next token."""
    with torch.no_grad():
        if is_swarmformer:
            result = model(input_ids)
            logits = result["logits"][:, -1, :]
        else:
            out = model(input_ids)
            logits = out.logits[:, -1, :]
    return F.softmax(logits.float(), dim=-1)


def compare_top_k(probs_a, probs_b, tokenizer, k=10):
    """Compare top-k token predictions between two models."""
    topk_a = torch.topk(probs_a[0], k)
    topk_b = torch.topk(probs_b[0], k)

    set_a = set(topk_a.indices.tolist())
    set_b = set(topk_b.indices.tolist())
    overlap = set_a & set_b

    print(f"  Top-{k} overlap: {len(overlap)}/{k} tokens in common")
    print()
    print(f"  {'Rank':<5} {'SmolLM3':<20} {'Prob':>8}  |  {'SwarmFormer':<20} {'Prob':>8}")
    print(f"  {'─'*5} {'─'*20} {'─'*8}  |  {'─'*20} {'─'*8}")
    for i in range(k):
        tok_a = tokenizer.decode([topk_a.indices[i].item()])
        tok_b = tokenizer.decode([topk_b.indices[i].item()])
        prob_a = topk_a.values[i].item()
        prob_b = topk_b.values[i].item()
        marker_a = "*" if topk_a.indices[i].item() in overlap else " "
        marker_b = "*" if topk_b.indices[i].item() in overlap else " "
        print(f"  {i+1:<5} {marker_a}{tok_a!r:<19} {prob_a:>8.4f}  |  {marker_b}{tok_b!r:<19} {prob_b:>8.4f}")

    # KL divergence (A || B)
    log_a = torch.log(probs_a + 1e-10)
    log_b = torch.log(probs_b + 1e-10)
    kl = F.kl_div(log_b, probs_a, reduction="batchmean").item()
    print(f"\n  KL(SmolLM3 || SwarmFormer) = {kl:.4f}")


def generate_text(model, tokenizer, prompt, max_new_tokens=50, is_swarmformer=False,
                  temperature=0.01, device="cuda"):
    """Generate text with near-greedy decoding for reproducibility."""
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    if is_swarmformer:
        cluster_size = model.config.cluster_size
        if input_ids.shape[1] < cluster_size:
            pad_len = cluster_size - input_ids.shape[1]
            pad_id = tokenizer.pad_token_id or 0
            input_ids = torch.cat([
                torch.full((1, pad_len), pad_id, dtype=input_ids.dtype, device=device),
                input_ids,
            ], dim=1)
            prompt_start = pad_len
        else:
            prompt_start = 0

        output = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=5,
        )
        return tokenizer.decode(output[0, prompt_start:], skip_special_tokens=True)
    else:
        output = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=5, do_sample=True,
        )
        return tokenizer.decode(output[0], skip_special_tokens=True)


def compute_perplexity(model, tokenizer, text, is_swarmformer=False, device="cuda"):
    """Compute perplexity on a reference text."""
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    if is_swarmformer:
        cluster_size = model.config.cluster_size
        pad_len = (cluster_size - input_ids.shape[1] % cluster_size) % cluster_size
        if pad_len > 0:
            pad_id = tokenizer.pad_token_id or 0
            input_ids = torch.cat([
                torch.full((1, pad_len), pad_id, dtype=input_ids.dtype, device=device),
                input_ids,
            ], dim=1)

    with torch.no_grad():
        if is_swarmformer:
            result = model(input_ids)
            logits = result["logits"]
        else:
            out = model(input_ids)
            logits = out.logits

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]

    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="mean",
    )
    return torch.exp(loss).item()


def pad_for_swarmformer(input_ids, cluster_size, device, pad_id=0):
    """Left-pad input_ids to a multiple of cluster_size."""
    L = input_ids.shape[1]
    pad_len = (cluster_size - L % cluster_size) % cluster_size
    if pad_len == 0 and L >= cluster_size:
        return input_ids, 0
    total = max(cluster_size, L + pad_len)
    pad_len = total - L
    padded = torch.cat([
        torch.full((1, pad_len), pad_id, dtype=input_ids.dtype, device=device),
        input_ids,
    ], dim=1)
    return padded, pad_len


def main():
    print("=" * 70)
    print("SmolLM3 vs CausalSwarmFormer (36-layer) Comparison")
    print("=" * 70)
    print()

    # ── Phase 1: Run SmolLM3, collect results, free GPU ──
    print("Phase 1: Collecting SmolLM3 outputs...")
    smol_model, tokenizer = load_smollm3(DEVICE)

    smol_results = {}

    # Next-token probs
    smol_results["probs"] = {}
    for prompt in PROMPTS:
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(DEVICE)
        probs = get_next_token_probs(smol_model, input_ids, is_swarmformer=False)
        smol_results["probs"][prompt] = probs.cpu()

    # Generation
    smol_results["gen"] = {}
    for prompt in PROMPTS:
        text = generate_text(smol_model, tokenizer, prompt, is_swarmformer=False, device=DEVICE)
        smol_results["gen"][prompt] = text

    # Perplexity
    smol_results["ppl"] = {}
    for text in REFERENCE_SENTENCES:
        ppl = compute_perplexity(smol_model, tokenizer, text, is_swarmformer=False, device=DEVICE)
        smol_results["ppl"][text] = ppl

    # Free GPU
    del smol_model
    torch.cuda.empty_cache()
    print("  SmolLM3 unloaded, GPU freed.\n")

    # ── Phase 2: Run SwarmFormer, collect results ──
    print("Phase 2: Collecting CausalSwarmFormer outputs...")
    swarm_model, _ = load_swarmformer(DEVICE, num_layers=36)
    print(f"  SwarmFormer GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    cluster_size = swarm_model.config.cluster_size

    swarm_results = {}

    # Next-token probs
    swarm_results["probs"] = {}
    for prompt in PROMPTS:
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(DEVICE)
        sw_input, _ = pad_for_swarmformer(input_ids, cluster_size, DEVICE)
        probs = get_next_token_probs(swarm_model, sw_input, is_swarmformer=True)
        swarm_results["probs"][prompt] = probs.cpu()

    # Generation
    swarm_results["gen"] = {}
    for prompt in PROMPTS:
        text = generate_text(swarm_model, tokenizer, prompt, is_swarmformer=True, device=DEVICE)
        swarm_results["gen"][prompt] = text

    # Perplexity
    swarm_results["ppl"] = {}
    for text in REFERENCE_SENTENCES:
        ppl = compute_perplexity(swarm_model, tokenizer, text, is_swarmformer=True, device=DEVICE)
        swarm_results["ppl"][text] = ppl

    del swarm_model
    torch.cuda.empty_cache()
    print("  SwarmFormer unloaded.\n")

    # ── Phase 3: Print comparison ──
    print("=" * 70)
    print("TEST 1: Next-Token Prediction (Top-10 Overlap)")
    print("=" * 70)
    for prompt in PROMPTS:
        print(f'\n  Prompt: "{prompt}"')
        compare_top_k(
            smol_results["probs"][prompt],
            swarm_results["probs"][prompt],
            tokenizer, k=10,
        )
        print()

    print("=" * 70)
    print("TEST 2: Generated Text Comparison (near-greedy, 50 tokens)")
    print("=" * 70)
    for prompt in PROMPTS:
        print(f'\n  Prompt: "{prompt}"')
        print(f"  SmolLM3:      {smol_results['gen'][prompt]}")
        print(f"  SwarmFormer:  {swarm_results['gen'][prompt]}")
        print()

    print("=" * 70)
    print("TEST 3: Perplexity on Reference Sentences")
    print("=" * 70)
    for text in REFERENCE_SENTENCES:
        smol_ppl = smol_results["ppl"][text]
        swarm_ppl = swarm_results["ppl"][text]
        print(f'\n  Text: "{text[:60]}..."')
        print(f"  SmolLM3 perplexity:      {smol_ppl:>10.2f}")
        print(f"  SwarmFormer perplexity:   {swarm_ppl:>10.2f}")
        print(f"  Ratio (Swarm/Smol):      {swarm_ppl/smol_ppl:>10.2f}x")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
