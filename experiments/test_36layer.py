"""Test 36-layer CausalSwarmFormer with converted SmolLM3 weights."""

import torch
import gc
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swarmformer.convert_smollm3 import convert_smollm3


def main():
    model, tokenizer = convert_smollm3(num_layers=36)
    model = model.to("cuda")
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()

    print(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB\n")

    cluster_size = model.config.cluster_size
    pad_id = tokenizer.pad_token_id or 0

    prompts = [
        "The capital of France is",
        "Once upon a time, there was a",
        "def fibonacci(n):\n    ",
        "In machine learning, transformers are",
        "The quick brown fox",
    ]

    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to("cuda")

        L = input_ids.shape[1]
        pad_len = max(0, cluster_size - L)
        if pad_len > 0:
            sw_input = torch.cat([
                torch.full((1, pad_len), pad_id, dtype=input_ids.dtype, device="cuda"),
                input_ids,
            ], dim=1)
            prompt_start = pad_len
        else:
            sw_input = input_ids
            prompt_start = 0

        # Top-5 next token
        with torch.no_grad():
            result = model(sw_input)
            logits = result["logits"][0, -1, :]
        probs = F.softmax(logits.float(), dim=-1)
        top5 = torch.topk(probs, 5)

        top_str = "  ".join(
            f"{repr(tokenizer.decode([t.item()]))}:{p.item():.4f}"
            for t, p in zip(top5.indices, top5.values)
        )
        print(f'PROMPT: "{prompt}"')
        print(f"  TOP5: {top_str}")

        # Generate
        output = model.generate(sw_input, max_new_tokens=40, temperature=0.01, top_k=5)
        gen_text = tokenizer.decode(output[0, prompt_start:], skip_special_tokens=True)
        print(f"  GEN:  {gen_text}")
        print()

    # Perplexity
    refs = [
        "The Eiffel Tower is located in Paris, France and was built in 1889.",
        "Machine learning models require large amounts of data to train effectively.",
    ]
    print("PERPLEXITY:")
    for text in refs:
        encoded = tokenizer(text, return_tensors="pt")
        ids = encoded["input_ids"].to("cuda")
        pad_len = (cluster_size - ids.shape[1] % cluster_size) % cluster_size
        if pad_len > 0:
            ids = torch.cat([
                torch.full((1, pad_len), 0, dtype=ids.dtype, device="cuda"),
                ids,
            ], dim=1)
        with torch.no_grad():
            result = model(ids)
            logits = result["logits"]
        shift_logits = logits[:, :-1, :].float()
        shift_labels = ids[:, 1:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="mean",
        )
        ppl = torch.exp(loss).item()
        short = text[:50] + "..." if len(text) > 50 else text
        print(f'  "{short}" -> PPL: {ppl:.2f}')


if __name__ == "__main__":
    main()
