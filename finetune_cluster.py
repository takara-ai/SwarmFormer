"""
Finetune CausalSwarmFormer cluster attention via self-distillation.

Teacher: same model, full O(L²) attention (no grad)
Student: same model, cluster O(C²) attention (trainable)

Designed for multi-GPU / H100. Uses AdamW, cosine LR, gradient checkpointing,
wandb logging, and proper eval/generation throughout training.

Usage:
    # Single GPU
    python finetune_cluster.py

    # Multi-GPU
    torchrun --nproc_per_node=N finetune_cluster.py

    # Override config
    python finetune_cluster.py --lr 3e-4 --epochs 5 --seq-len 512 --batch-size 4
"""

import argparse
import gc
import math
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from swarmformer.convert_smollm3 import convert_smollm3
from swarmformer.causal_model import CausalSwarmFormerForLM
from swarmformer.config import CausalSwarmFormerConfig

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ---------------------------------------------------------------------------
# Defaults (override via CLI args)
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    hf_model="HuggingFaceTB/SmolLM3-3B",
    num_layers=36,
    cluster_size=64,
    seq_len=512,
    batch_size=2,
    grad_accum=8,
    lr=3e-4,
    weight_decay=0.01,
    warmup_steps=200,
    epochs=5,
    distill_alpha=0.3,
    distill_temp=1.5,
    eval_every=200,
    save_every=500,
    max_grad_norm=1.0,
    train_attn=True,         # also finetune Q/K/V/O (not just local_agg)
    train_mlp=False,         # freeze MLP by default
    train_norms=False,       # freeze norms by default
    save_dir="checkpoints",
    wandb_project="swarmformer-cluster",
    wandb_run=None,          # auto-generated if None
    resume=None,             # path to checkpoint to resume from
    dataset="wikitext",      # "wikitext" or HF dataset name
    dataset_config="wikitext-103-raw-v1",  # larger than wikitext-2
    max_train_tokens=None,   # None = use all
    seed=42,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextChunkDataset(Dataset):
    def __init__(self, token_ids: list[int], seq_len: int):
        self.seq_len = seq_len
        n = (len(token_ids) // seq_len) * seq_len
        self.data = torch.tensor(token_ids[:n], dtype=torch.long).view(-1, seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_text_dataset(tokenizer, dataset_name, dataset_config, split, seq_len, max_tokens=None):
    print(f"Loading {dataset_name}/{dataset_config} ({split})...")
    ds = load_dataset(dataset_name, dataset_config, split=split)
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    print(f"  Raw: {len(text):,} chars")
    tokens = tokenizer.encode(text)
    if max_tokens:
        tokens = tokens[:max_tokens]
    print(f"  Tokens: {len(tokens):,}")
    dataset = TextChunkDataset(tokens, seq_len)
    print(f"  Chunks: {len(dataset)} x {seq_len}")
    return dataset


# ---------------------------------------------------------------------------
# Freeze logic
# ---------------------------------------------------------------------------

def setup_trainable(model, train_attn, train_mlp, train_norms):
    """Freeze/unfreeze parameters. Returns (trainable_params, frozen_count)."""
    for p in model.parameters():
        p.requires_grad = False

    trainable = []
    for layer in model.layers:
        # Always train local_agg (the SwarmFormer-specific component)
        for p in layer.local_agg.parameters():
            p.requires_grad = True
            trainable.append(p)
        if train_attn:
            for p in layer.attn.parameters():
                p.requires_grad = True
                trainable.append(p)
        if train_mlp:
            for p in layer.mlp.parameters():
                p.requires_grad = True
                trainable.append(p)
        if train_norms:
            for p in layer.input_layernorm.parameters():
                p.requires_grad = True
                trainable.append(p)
            for p in layer.post_attention_layernorm.parameters():
                p.requires_grad = True
                trainable.append(p)

    n_trainable = sum(p.numel() for p in trainable)
    n_frozen = sum(p.numel() for p in model.parameters()) - n_trainable
    print(f"Trainable: {n_trainable:,} | Frozen: {n_frozen:,}")
    return trainable


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def distillation_loss(student_logits, teacher_logits, labels, alpha, temp):
    student_soft = F.log_softmax(student_logits / temp, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temp, dim=-1)
    kl = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temp ** 2)
    ce = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1), ignore_index=-100,
    )
    return alpha * kl + (1 - alpha) * ce, kl.item(), ce.item()


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataset, device, max_batches=100):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ids = batch.to(device)
        result = model(ids, labels=ids, use_cluster=True)
        total_loss += result["loss"].item() * (ids.shape[1] - 1)
        total_tokens += ids.shape[1] - 1
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


@torch.no_grad()
def generate_samples(model, tokenizer, device, prompts, max_tokens=60):
    model.eval()
    cs = model.config.cluster_size
    pad_id = tokenizer.pad_token_id or 0
    results = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        pad_len = max(0, cs - ids.shape[1])
        if pad_len > 0:
            ids = torch.cat([torch.full((1, pad_len), pad_id, dtype=ids.dtype, device=device), ids], dim=1)
            start = pad_len
        else:
            start = 0
        out = model.generate(ids, max_new_tokens=max_tokens, temperature=0.7, top_k=50, use_cluster=True)
        text = tokenizer.decode(out[0, start:], skip_special_tokens=True)
        results.append((prompt, text))
    return results


# ---------------------------------------------------------------------------
# Cosine LR with warmup
# ---------------------------------------------------------------------------

def get_lr(step, warmup, total, lr_max):
    if step < warmup:
        return lr_max * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr_max * 0.5 * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        t = type(v) if v is not None else str
        if t == bool:
            parser.add_argument(f"--{k.replace('_', '-')}", default=v, action="store_true" if not v else "store_false")
        else:
            parser.add_argument(f"--{k.replace('_', '-')}", default=v, type=t)
    args = parser.parse_args()
    cfg = {k: getattr(args, k) for k in DEFAULTS}

    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Wandb ──
    if HAS_WANDB:
        wandb.init(
            project=cfg["wandb_project"],
            name=cfg["wandb_run"],
            config=cfg,
        )

    # ── Load model ──
    if cfg["resume"]:
        print(f"Resuming from {cfg['resume']}...")
        config = CausalSwarmFormerConfig(num_layers=cfg["num_layers"], cluster_size=cfg["cluster_size"])
        torch.set_default_dtype(torch.bfloat16)
        model = CausalSwarmFormerForLM(config)
        torch.set_default_dtype(torch.float32)
        model.load_state_dict(torch.load(cfg["resume"], map_location="cpu"))
        tokenizer = AutoTokenizer.from_pretrained(cfg["hf_model"])
    else:
        model, tokenizer = convert_smollm3(hf_model_id=cfg["hf_model"], num_layers=cfg["num_layers"])

    model = model.to(device)
    model.train()

    trainable_params = setup_trainable(model, cfg["train_attn"], cfg["train_mlp"], cfg["train_norms"])

    # ── Data ──
    train_ds = load_text_dataset(tokenizer, cfg["dataset"], cfg["dataset_config"],
                                 "train", cfg["seq_len"], cfg["max_train_tokens"])
    val_ds = load_text_dataset(tokenizer, cfg["dataset"], cfg["dataset_config"],
                               "validation", cfg["seq_len"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    total_steps = (len(train_loader) // cfg["grad_accum"]) * cfg["epochs"]
    print(f"Total optimizer steps: {total_steps}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                                  betas=(0.9, 0.95), fused=True)

    # ── Pre-train eval ──
    pre_ppl = evaluate(model, val_ds, device)
    print(f"Pre-training PPL: {pre_ppl:.2f}")
    if HAS_WANDB:
        wandb.log({"eval/ppl": pre_ppl, "step": 0})

    # ── Training ──
    os.makedirs(cfg["save_dir"], exist_ok=True)
    global_step = 0
    best_ppl = pre_ppl
    test_prompts = [
        "The capital of France is",
        "Once upon a time, there was a",
        "def fibonacci(n):\n    ",
        "In machine learning, transformers are",
    ]
    optimizer.zero_grad()

    for epoch in range(cfg["epochs"]):
        epoch_loss, epoch_kl, epoch_ce, epoch_steps = 0, 0, 0, 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch.to(device)
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100

            # Teacher (full attention, no grad)
            model.eval()
            with torch.no_grad():
                teacher_logits = model(input_ids, use_cluster=False)["logits"][:, :-1, :].clone()
            torch.cuda.empty_cache()

            # Student (cluster attention, grad)
            model.train()
            student_logits = model(input_ids, use_cluster=True)["logits"][:, :-1, :]
            shift_labels = labels[:, :-1]

            loss, kl_val, ce_val = distillation_loss(
                student_logits, teacher_logits, shift_labels,
                cfg["distill_alpha"], cfg["distill_temp"],
            )
            del teacher_logits
            (loss / cfg["grad_accum"]).backward()

            epoch_loss += loss.item()
            epoch_kl += kl_val
            epoch_ce += ce_val
            epoch_steps += 1

            if (batch_idx + 1) % cfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg["max_grad_norm"])

                # Cosine LR
                lr = get_lr(global_step, cfg["warmup_steps"], total_steps, cfg["lr"])
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Log
                if global_step % 10 == 0:
                    avg_loss = epoch_loss / epoch_steps
                    avg_kl = epoch_kl / epoch_steps
                    avg_ce = epoch_ce / epoch_steps
                    elapsed = time.time() - t0
                    tok_s = epoch_steps * cfg["batch_size"] * cfg["seq_len"] / elapsed

                    print(f"  [E{epoch+1} s{global_step}] loss={avg_loss:.1f} "
                          f"kl={avg_kl:.1f} ce={avg_ce:.2f} lr={lr:.2e} "
                          f"tok/s={tok_s:.0f} t={elapsed:.0f}s", flush=True)

                    if HAS_WANDB:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/kl": avg_kl,
                            "train/ce": avg_ce,
                            "train/lr": lr,
                            "train/tok_per_sec": tok_s,
                            "step": global_step,
                            "epoch": epoch + batch_idx / len(train_loader),
                        })

                # Eval
                if global_step % cfg["eval_every"] == 0:
                    ppl = evaluate(model, val_ds, device)
                    print(f"  >>> Eval PPL: {ppl:.2f} (best: {best_ppl:.2f})")

                    log = {"eval/ppl": ppl, "step": global_step}

                    if ppl < best_ppl:
                        best_ppl = ppl
                        torch.save(model.state_dict(), f"{cfg['save_dir']}/best.pt")
                        print(f"  >>> Saved best (PPL={ppl:.2f})")

                    # Generation samples
                    samples = generate_samples(model, tokenizer, device, test_prompts)
                    for prompt, text in samples:
                        print(f"    [{prompt[:30]}...] → {text[:80]}")
                    if HAS_WANDB:
                        table = wandb.Table(columns=["prompt", "generation"])
                        for p, t in samples:
                            table.add_data(p, t)
                        log["eval/generations"] = table

                    if HAS_WANDB:
                        wandb.log(log)
                    model.train()

                # Save checkpoint
                if global_step % cfg["save_every"] == 0:
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": global_step,
                        "epoch": epoch,
                        "best_ppl": best_ppl,
                        "config": cfg,
                    }, f"{cfg['save_dir']}/step_{global_step}.pt")

        # End of epoch
        ppl = evaluate(model, val_ds, device)
        print(f"\n  Epoch {epoch+1} done. PPL: {ppl:.2f} (best: {best_ppl:.2f})")
        if ppl < best_ppl:
            best_ppl = ppl
            torch.save(model.state_dict(), f"{cfg['save_dir']}/best.pt")
        torch.save(model.state_dict(), f"{cfg['save_dir']}/epoch{epoch+1}.pt")

    # ── Final ──
    print("\n" + "=" * 60)
    model.load_state_dict(torch.load(f"{cfg['save_dir']}/best.pt", map_location=device))
    final_ppl = evaluate(model, val_ds, device, max_batches=500)
    print(f"Final PPL: {final_ppl:.2f}")
    print(f"Improvement: {pre_ppl:.2f} → {final_ppl:.2f}")

    samples = generate_samples(model, tokenizer, device, test_prompts)
    print("\nGeneration samples:")
    for p, t in samples:
        print(f"  [{p}] → {t}")

    if HAS_WANDB:
        wandb.log({"eval/final_ppl": final_ppl})
        wandb.finish()


if __name__ == "__main__":
    main()
