"""
CausalSwarmFormer multi-stage finetuning pipeline.

Based on MOHAWK/RADLADS/Longformer research:
  Stage 1: Hidden-state alignment (train new components only, L2 loss per layer)
  Stage 2: End-to-end KL distillation (unfreeze all, KL only, no CE)
  Stage 3: Pure CE continued pretraining (no teacher)

Plus progressive layer activation from SWAA: gradually switch layers
from full attention to cluster attention.

Usage:
    python finetune_cluster.py --stage 1
    python finetune_cluster.py --stage 2 --resume checkpoints/stage1_best.pt
    python finetune_cluster.py --stage 3 --resume checkpoints/stage2_best.pt

    # Or run all stages sequentially:
    python finetune_cluster.py --stage all
"""

import argparse
import gc
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from swarmformer.convert_smollm3 import convert_smollm3
from swarmformer.causal_model import (
    CausalSwarmFormerForLM, CausalSwarmFormerLayer,
    causal_cluster_tokens,
)
from swarmformer.config import CausalSwarmFormerConfig

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    hf_model="HuggingFaceTB/SmolLM3-3B",
    num_layers=36,
    cluster_size=16,
    seq_len=512,
    batch_size=2,
    grad_accum=8,
    warmup_steps=200,
    max_grad_norm=1.0,
    save_dir="checkpoints",
    wandb_project="swarmformer-cluster",
    wandb_run=None,
    resume=None,
    dataset="wikitext",
    dataset_config="wikitext-103-raw-v1",
    max_train_tokens=None,
    seed=42,
    stage="1",  # "1", "2", "3", or "all"
    # Stage-specific defaults (overridable)
    s1_lr=1e-3,
    s1_tokens=200_000_000,    # 200M tokens
    s2_lr=1e-5,
    s2_tokens=500_000_000,    # 500M tokens
    s2_kl_temp=2.0,
    s3_lr=3e-5,
    s3_tokens=1_000_000_000,  # 1B tokens
    # Progressive layer activation
    progressive=True,         # gradually enable cluster layers
    progressive_interval=500, # enable one more layer every N steps
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextChunkDataset(Dataset):
    def __init__(self, token_ids, seq_len):
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
# Freeze helpers
# ---------------------------------------------------------------------------

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_new_components(model):
    """Stage 1: only local_agg + broadcast."""
    freeze_all(model)
    trainable = []
    for layer in model.layers:
        for p in layer.local_agg.parameters():
            p.requires_grad = True
            trainable.append(p)
        for p in layer.broadcast.parameters():
            p.requires_grad = True
            trainable.append(p)
    n = sum(p.numel() for p in trainable)
    print(f"Stage 1 trainable: {n:,} (local_agg + broadcast only)")
    return trainable


def unfreeze_all_params(model):
    """Stage 2/3: everything trainable."""
    trainable = list(model.parameters())
    for p in trainable:
        p.requires_grad = True
    n = sum(p.numel() for p in trainable)
    print(f"Full trainable: {n:,}")
    return trainable


# ---------------------------------------------------------------------------
# Progressive layer activation
# ---------------------------------------------------------------------------

class ProgressiveClusterSchedule:
    """Gradually enable cluster attention layer by layer.
    Starts with all layers using full attention, enables one more
    cluster layer every `interval` steps (from last layer to first)."""

    def __init__(self, num_layers: int, interval: int):
        self.num_layers = num_layers
        self.interval = interval
        self.cluster_layers = set()  # which layers use cluster attention

    def step(self, global_step: int) -> set:
        # Enable layers from the LAST to the FIRST
        # (later layers are easier to adapt since they're closer to output)
        n_active = min(global_step // self.interval, self.num_layers)
        self.cluster_layers = set(range(self.num_layers - n_active, self.num_layers))
        return self.cluster_layers

    def all_active(self):
        self.cluster_layers = set(range(self.num_layers))
        return self.cluster_layers


# ---------------------------------------------------------------------------
# Modified forward for per-layer cluster control
# ---------------------------------------------------------------------------

def forward_with_per_layer_cluster(model, input_ids, cluster_layers: set):
    """Run model with some layers using cluster attention and others using full."""
    B, L = input_ids.shape
    cs = model.config.cluster_size
    assert L % cs == 0

    x = model.embed_tokens(input_ids)
    positions = torch.arange(L, device=input_ids.device)
    cos_full, sin_full = model.rotary_emb(positions)
    cluster_pos = torch.arange(cs - 1, L, cs, device=input_ids.device)
    cos_cluster, sin_cluster = model.rotary_emb(cluster_pos)

    for i, layer in enumerate(model.layers):
        use_cluster = (i in cluster_layers)
        if model.training and use_cluster:
            x = torch.utils.checkpoint.checkpoint(
                layer, x, cos_full, sin_full, cos_cluster, sin_cluster, use_cluster,
                use_reentrant=False,
            )
        else:
            x = layer(x, cos_full, sin_full, cos_cluster, sin_cluster, use_cluster=use_cluster)

    x = model.norm(x)
    return model.lm_head(x)


# ---------------------------------------------------------------------------
# Stage 1: Hidden-state alignment (L2 per layer)
# ---------------------------------------------------------------------------

def compute_layer_alignment_loss(model, input_ids, cluster_layers: set):
    """
    For each layer in cluster_layers, compute L2 between:
      - hidden state from full attention path
      - hidden state from cluster attention path
    Only the cluster path has gradients.
    """
    B, L = input_ids.shape
    cs = model.config.cluster_size

    x = model.embed_tokens(input_ids)
    positions = torch.arange(L, device=input_ids.device)
    cos_full, sin_full = model.rotary_emb(positions)
    cluster_pos = torch.arange(cs - 1, L, cs, device=input_ids.device)
    cos_cluster, sin_cluster = model.rotary_emb(cluster_pos)

    total_loss = torch.tensor(0.0, device=input_ids.device)
    n_aligned = 0

    for i, layer in enumerate(model.layers):
        if i in cluster_layers:
            # Full attention path (no grad) — the target
            with torch.no_grad():
                x_full = layer(x.detach(), cos_full, sin_full, cos_cluster, sin_cluster, use_cluster=False)

            # Cluster attention path (with grad) — the student
            x_cluster = torch.utils.checkpoint.checkpoint(
                layer, x.detach(), cos_full, sin_full, cos_cluster, sin_cluster, True,
                use_reentrant=False,
            )

            # L2 alignment loss
            total_loss = total_loss + F.mse_loss(x_cluster, x_full.detach())
            n_aligned += 1

            # Continue with full attention output for subsequent layers
            x = x_full.detach()
        else:
            with torch.no_grad():
                x = layer(x, cos_full, sin_full, cos_cluster, sin_cluster, use_cluster=False)

    return total_loss / max(n_aligned, 1)


# ---------------------------------------------------------------------------
# Eval + Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataset, device, cluster_layers=None, max_batches=100):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ids = batch.to(device)
        if cluster_layers is not None:
            logits = forward_with_per_layer_cluster(model, ids, cluster_layers)
        else:
            logits = model(ids, use_cluster=True)["logits"]
        shift_logits = logits[:, :-1, :]
        shift_labels = ids[:, 1:]
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        total_loss += loss.item() * (ids.shape[1] - 1)
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


TEST_PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a",
    "def fibonacci(n):\n    ",
    "In machine learning, transformers are",
]


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(step, warmup, total, lr_max):
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    progress = (step - warmup) / max(1, total - warmup)
    return lr_max * 0.5 * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Generic training loop
# ---------------------------------------------------------------------------

def train_loop(
    model, tokenizer, train_ds, val_ds, device, cfg,
    trainable_params, lr, max_tokens, stage_name,
    loss_fn,  # callable(model, input_ids, cluster_layers) -> (loss, log_dict)
    cluster_schedule=None,
):
    """Generic training loop used by all stages."""
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    tokens_per_step = cfg["batch_size"] * cfg["seq_len"]
    max_steps = max_tokens // (tokens_per_step * cfg["grad_accum"])
    print(f"\n{'='*60}")
    print(f"  {stage_name}: {max_steps} steps, LR={lr}")
    print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print(f"{'='*60}\n")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01, betas=(0.9, 0.95))

    global_step = 0
    best_ppl = float("inf")
    prefix = stage_name.replace(" ", "_").lower()
    optimizer.zero_grad()

    # Initial eval
    cl = cluster_schedule.cluster_layers if cluster_schedule else set(range(cfg["num_layers"]))
    ppl = evaluate(model, val_ds, device, cl)
    print(f"  Pre-training PPL: {ppl:.2f}")
    if HAS_WANDB:
        wandb.log({f"{prefix}/ppl": ppl, "step": 0})

    epoch = 0
    while global_step < max_steps:
        epoch += 1
        t0 = time.time()
        epoch_loss, epoch_steps = 0, 0

        for batch in loader:
            if global_step >= max_steps:
                break

            input_ids = batch.to(device)

            # Update progressive schedule
            if cluster_schedule:
                cl = cluster_schedule.step(global_step)
            else:
                cl = set(range(cfg["num_layers"]))

            model.train()
            loss, log_extra = loss_fn(model, input_ids, cl)
            (loss / cfg["grad_accum"]).backward()

            epoch_loss += loss.item()
            epoch_steps += 1

            if epoch_steps % cfg["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg["max_grad_norm"])
                cur_lr = get_lr(global_step, cfg["warmup_steps"], max_steps, lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = cur_lr
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    avg = epoch_loss / epoch_steps
                    elapsed = time.time() - t0
                    tok_s = epoch_steps * tokens_per_step / elapsed
                    n_cl = len(cl) if cluster_schedule else cfg["num_layers"]
                    extra_str = " ".join(f"{k}={v:.3f}" for k, v in log_extra.items())
                    print(f"  [E{epoch} s{global_step}] loss={avg:.3f} {extra_str} "
                          f"lr={cur_lr:.2e} cl={n_cl}/{cfg['num_layers']} "
                          f"tok/s={tok_s:.0f}", flush=True)
                    if HAS_WANDB:
                        log = {f"{prefix}/loss": avg, f"{prefix}/lr": cur_lr,
                               f"{prefix}/cluster_layers": n_cl, "step": global_step}
                        log.update({f"{prefix}/{k}": v for k, v in log_extra.items()})
                        wandb.log(log)

                if global_step % 200 == 0:
                    ppl = evaluate(model, val_ds, device, cl)
                    print(f"  >>> PPL: {ppl:.2f} (best: {best_ppl:.2f})")
                    if ppl < best_ppl:
                        best_ppl = ppl
                        torch.save(model.state_dict(), f"{cfg['save_dir']}/{prefix}_best.pt")
                        print(f"  >>> Saved best")
                    # Generation samples
                    samples = generate_samples(model, tokenizer, device, TEST_PROMPTS)
                    for p, t in samples:
                        print(f"    [{p[:30]}] → {t[:80]}")
                    if HAS_WANDB:
                        wlog = {f"{prefix}/ppl": ppl, "step": global_step}
                        table = wandb.Table(columns=["prompt", "generation"])
                        for p, t in samples:
                            table.add_data(p, t)
                        wlog[f"{prefix}/generations"] = table
                        wandb.log(wlog)
                    model.train()

                if global_step % 1000 == 0:
                    torch.save({"model": model.state_dict(), "step": global_step,
                                "best_ppl": best_ppl}, f"{cfg['save_dir']}/{prefix}_step{global_step}.pt")

    print(f"\n  {stage_name} done. Best PPL: {best_ppl:.2f}")
    return best_ppl


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

def run_stage1(model, tokenizer, train_ds, val_ds, device, cfg):
    """Hidden-state alignment: teach local_agg + broadcast to approximate full attention."""
    trainable = unfreeze_new_components(model)
    schedule = ProgressiveClusterSchedule(cfg["num_layers"], cfg["progressive_interval"]) if cfg["progressive"] else None

    def loss_fn(model, input_ids, cluster_layers):
        loss = compute_layer_alignment_loss(model, input_ids, cluster_layers)
        return loss, {"l2": loss.item()}

    return train_loop(model, tokenizer, train_ds, val_ds, device, cfg,
                      trainable, cfg["s1_lr"], cfg["s1_tokens"], "Stage1_Align",
                      loss_fn, schedule)


def run_stage2(model, tokenizer, train_ds, val_ds, device, cfg):
    """End-to-end KL distillation (KL only, no CE)."""
    trainable = unfreeze_all_params(model)
    temp = cfg["s2_kl_temp"]

    def loss_fn(model, input_ids, cluster_layers):
        # Teacher: full attention (no grad)
        model.eval()
        with torch.no_grad():
            teacher_logits = forward_with_per_layer_cluster(model, input_ids, set()).clone()
        torch.cuda.empty_cache()

        # Student: cluster attention
        model.train()
        student_logits = forward_with_per_layer_cluster(model, input_ids, cluster_layers)

        # KL only (no CE!)
        student_soft = F.log_softmax(student_logits / temp, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temp, dim=-1)
        kl = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temp ** 2)
        del teacher_logits
        return kl, {"kl": kl.item()}

    return train_loop(model, tokenizer, train_ds, val_ds, device, cfg,
                      trainable, cfg["s2_lr"], cfg["s2_tokens"], "Stage2_KL",
                      loss_fn, None)  # all layers cluster in stage 2


def run_stage3(model, tokenizer, train_ds, val_ds, device, cfg):
    """Pure CE continued pretraining (no teacher)."""
    trainable = unfreeze_all_params(model)

    def loss_fn(model, input_ids, cluster_layers):
        logits = model(input_ids, use_cluster=True)["logits"]
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        return ce, {"ce": ce.item()}

    return train_loop(model, tokenizer, train_ds, val_ds, device, cfg,
                      trainable, cfg["s3_lr"], cfg["s3_tokens"], "Stage3_CE",
                      loss_fn, None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        t = type(v) if v is not None else str
        if t == bool:
            parser.add_argument(f"--{k.replace('_', '-')}", default=v,
                                action="store_false" if v else "store_true")
        else:
            parser.add_argument(f"--{k.replace('_', '-')}", default=v, type=t)
    args = parser.parse_args()
    cfg = {k: getattr(args, k) for k in DEFAULTS}
    # Parse int token counts that may come as float from CLI
    for k in ["s1_tokens", "s2_tokens", "s3_tokens"]:
        cfg[k] = int(cfg[k])

    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["save_dir"], exist_ok=True)

    if HAS_WANDB:
        wandb.init(project=cfg["wandb_project"], name=cfg["wandb_run"], config=cfg)

    # Load model
    if cfg["resume"]:
        print(f"Resuming from {cfg['resume']}...")
        config = CausalSwarmFormerConfig(num_layers=cfg["num_layers"], cluster_size=cfg["cluster_size"])
        torch.set_default_dtype(torch.bfloat16)
        model = CausalSwarmFormerForLM(config)
        torch.set_default_dtype(torch.float32)
        state = torch.load(cfg["resume"], map_location="cpu")
        model.load_state_dict(state if isinstance(state, dict) and "model" not in state else state.get("model", state))
        tokenizer = AutoTokenizer.from_pretrained(cfg["hf_model"])
    else:
        model, tokenizer = convert_smollm3(hf_model_id=cfg["hf_model"], num_layers=cfg["num_layers"])
    model = model.to(device)

    # Load data
    train_ds = load_text_dataset(tokenizer, cfg["dataset"], cfg["dataset_config"],
                                 "train", cfg["seq_len"], cfg["max_train_tokens"])
    val_ds = load_text_dataset(tokenizer, cfg["dataset"], cfg["dataset_config"],
                               "validation", cfg["seq_len"])

    # Run stages
    stage = cfg["stage"]

    if stage in ("1", "all"):
        run_stage1(model, tokenizer, train_ds, val_ds, device, cfg)
        if stage == "all":
            torch.save(model.state_dict(), f"{cfg['save_dir']}/stage1_final.pt")

    if stage in ("2", "all"):
        run_stage2(model, tokenizer, train_ds, val_ds, device, cfg)
        if stage == "all":
            torch.save(model.state_dict(), f"{cfg['save_dir']}/stage2_final.pt")

    if stage in ("3", "all"):
        run_stage3(model, tokenizer, train_ds, val_ds, device, cfg)

    # Final eval
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    all_layers = set(range(cfg["num_layers"]))
    final_ppl = evaluate(model, val_ds, device, all_layers, max_batches=500)
    print(f"Final PPL (all layers cluster): {final_ppl:.2f}")

    samples = generate_samples(model, tokenizer, device, TEST_PROMPTS)
    for p, t in samples:
        print(f"  [{p}] → {t}")

    if HAS_WANDB:
        wandb.log({"final_ppl": final_ppl})
        wandb.finish()


if __name__ == "__main__":
    main()
