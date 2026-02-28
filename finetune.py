"""
Finetune CausalSwarmFormer from SmolLM3 weights using self-distillation.

Teacher: same model with full O(L²) attention (no gradients)
Student: same model with cluster O(C²) attention (trainable local_agg)

Loss = α * KL(student || teacher) + (1-α) * CE(student, labels)
"""

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

# ── Config ──
HF_MODEL = "HuggingFaceTB/SmolLM3-3B"
NUM_LAYERS = 36
SEQ_LEN = 256  # 4 clusters of 64
CLUSTER_SIZE = 64
DEVICE = "cuda"
DTYPE = torch.bfloat16

# Training
LR = 0.01
EPOCHS = 2
BATCH_SIZE = 1
GRAD_ACCUM = 4
DISTILL_ALPHA = 0.3  # weight for distillation loss vs CE
DISTILL_TEMP = 1.5   # temperature for distillation
EVAL_EVERY = 100     # steps
SAVE_DIR = "checkpoints"
MAX_TRAIN_TOKENS = None  # None = use all of wikitext-2


class TextChunkDataset(Dataset):
    """Chunks tokenized text into fixed-length segments."""

    def __init__(self, token_ids: list[int], seq_len: int):
        self.seq_len = seq_len
        # Truncate to exact multiple
        n = (len(token_ids) // seq_len) * seq_len
        self.data = torch.tensor(token_ids[:n], dtype=torch.long).view(-1, seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_wikitext(tokenizer, split="train", seq_len=512, max_tokens=None):
    """Load and tokenize wikitext-2."""
    print(f"Loading wikitext-2 ({split})...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    # Concatenate all text
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    print(f"  Raw text: {len(text):,} chars")

    tokens = tokenizer.encode(text)
    if max_tokens:
        tokens = tokens[:max_tokens]
    print(f"  Tokens: {len(tokens):,}")

    dataset = TextChunkDataset(tokens, seq_len)
    print(f"  Chunks: {len(dataset)} × {seq_len}")
    return dataset


def freeze_smollm3_weights(model):
    """Freeze MLP/norm/embedding weights. Train local_agg + attention Q/K/V/O."""
    for p in model.parameters():
        p.requires_grad = False

    trainable = 0
    for layer in model.layers:
        # Unfreeze local_agg (SwarmFormer-specific)
        for p in layer.local_agg.parameters():
            p.requires_grad = True
            trainable += p.numel()
        # Unfreeze attention Q/K/V/O (need to adapt to cluster inputs)
        for p in layer.attn.parameters():
            p.requires_grad = True
            trainable += p.numel()

    frozen = sum(p.numel() for p in model.parameters()) - trainable
    print(f"Trainable: {trainable:,} | Frozen: {frozen:,} | Ratio: {trainable/frozen:.4f}")
    return trainable


@torch.no_grad()
def evaluate(model, dataset, device, max_batches=50):
    """Compute perplexity on a dataset using cluster attention."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        input_ids = batch.to(device)
        labels = input_ids.clone()
        result = model(input_ids, labels=labels, use_cluster=True)
        total_loss += result["loss"].item() * (input_ids.shape[1] - 1)
        total_tokens += input_ids.shape[1] - 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return math.exp(avg_loss)


def distillation_loss(student_logits, teacher_logits, labels, alpha, temp):
    """Combined distillation + CE loss. Returns (total, kl, ce)."""
    student_soft = F.log_softmax(student_logits / temp, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temp, dim=-1)
    kl = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temp ** 2)

    ce = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

    total = alpha * kl + (1 - alpha) * ce
    return total, kl.item(), ce.item()


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Load model ──
    model, tokenizer = convert_smollm3(hf_model_id=HF_MODEL, num_layers=NUM_LAYERS)
    model = model.to(DEVICE).to(DTYPE)
    model.train()

    # Freeze SmolLM3 weights, only train local_agg
    n_trainable = freeze_smollm3_weights(model)

    # ── Load data ──
    train_ds = load_wikitext(tokenizer, "train", SEQ_LEN, MAX_TRAIN_TOKENS)
    val_ds = load_wikitext(tokenizer, "validation", SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # ── Optimizer: SGD (no momentum states = fits in 12GB VRAM) ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, lr=LR)

    # ── Pre-training eval ──
    print("\nPre-training evaluation (cluster attention)...")
    pre_ppl = evaluate(model, val_ds, DEVICE)
    print(f"  PPL before training: {pre_ppl:.2f}")

    # ── Training loop ──
    print(f"\nTraining: {EPOCHS} epochs, {len(train_loader)} batches/epoch")
    print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Distillation alpha={DISTILL_ALPHA}, temp={DISTILL_TEMP}")

    global_step = 0
    best_ppl = pre_ppl
    optimizer.zero_grad()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_ce = 0.0
        epoch_steps = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch.to(DEVICE)
            labels = input_ids.clone()
            # Shift labels for next-token prediction
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100

            # Teacher forward (full attention, no grad) — extract and free
            model.eval()
            with torch.no_grad():
                teacher_logits = model(input_ids, use_cluster=False)["logits"][:, :-1, :].clone()
            torch.cuda.empty_cache()

            # Student forward (cluster attention, with grad)
            model.train()
            student_logits = model(input_ids, use_cluster=True)["logits"][:, :-1, :]
            shift_labels = labels[:, :-1]

            loss, kl_val, ce_val = distillation_loss(
                student_logits, teacher_logits, shift_labels,
                DISTILL_ALPHA, DISTILL_TEMP,
            )
            del teacher_logits
            (loss / GRAD_ACCUM).backward()

            epoch_loss += loss.item()
            epoch_kl += kl_val
            epoch_ce += ce_val
            epoch_steps += 1

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    avg = epoch_loss / epoch_steps
                    avg_kl = epoch_kl / epoch_steps
                    avg_ce = epoch_ce / epoch_steps
                    elapsed = time.time() - t0
                    tok_s = epoch_steps * BATCH_SIZE * SEQ_LEN / elapsed
                    print(f"  [E{epoch+1} s{global_step}] loss={avg:.1f} "
                          f"kl={avg_kl:.1f} ce={avg_ce:.1f} "
                          f"tok/s={tok_s:.0f} t={elapsed:.0f}s", flush=True)

                if global_step % EVAL_EVERY == 0:
                    ppl = evaluate(model, val_ds, DEVICE)
                    print(f"  >>> Eval PPL: {ppl:.2f} (best: {best_ppl:.2f})")
                    if ppl < best_ppl:
                        best_ppl = ppl
                        torch.save(model.state_dict(), f"{SAVE_DIR}/best.pt")
                        print(f"  >>> Saved best model (PPL={ppl:.2f})")
                    model.train()

        # End of epoch eval
        ppl = evaluate(model, val_ds, DEVICE)
        print(f"\n  Epoch {epoch+1} done. Val PPL: {ppl:.2f} (best: {best_ppl:.2f})")
        if ppl < best_ppl:
            best_ppl = ppl
            torch.save(model.state_dict(), f"{SAVE_DIR}/best.pt")
            print(f"  Saved best model")

        torch.save(model.state_dict(), f"{SAVE_DIR}/epoch{epoch+1}.pt")

    # ── Final evaluation ──
    print("\n" + "=" * 60)
    print("Training complete. Loading best checkpoint...")
    model.load_state_dict(torch.load(f"{SAVE_DIR}/best.pt", map_location=DEVICE))
    model.eval()

    final_ppl = evaluate(model, val_ds, DEVICE, max_batches=200)
    print(f"Final PPL (cluster attention): {final_ppl:.2f}")
    print(f"Improvement: {pre_ppl:.2f} → {final_ppl:.2f} ({pre_ppl/final_ppl:.1f}x)")

    # Generation samples
    print("\nGeneration samples:")
    prompts = [
        "The capital of France is",
        "Once upon a time, there was a",
        "def fibonacci(n):\n    ",
    ]
    cs = model.config.cluster_size
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        ids = enc["input_ids"].to(DEVICE)
        pad_len = max(0, cs - ids.shape[1])
        if pad_len > 0:
            ids = F.pad(ids, (pad_len, 0), value=tokenizer.pad_token_id or 0)
            start = pad_len
        else:
            start = 0
        out = model.generate(ids, max_new_tokens=50, temperature=0.01, top_k=5, use_cluster=True)
        text = tokenizer.decode(out[0, start:], skip_special_tokens=True)
        print(f'  "{prompt}" → {text}')

    print(f"\nBest model saved to {SAVE_DIR}/best.pt")


if __name__ == "__main__":
    train()
