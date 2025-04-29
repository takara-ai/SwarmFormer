# distill_disaster.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from swarmformer.model import SwarmFormerModel
import argparse
import time
import wandb
import os

def get_device():
    # Select MPS on Apple M1/M2/M3, then CUDA, then CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def prepare_dataloader(tokenizer, seq_len, batch_size):
    raw_ds = load_dataset("disaster_response_messages")
    raw_ds = raw_ds.filter(lambda x: x['related'] != 2)
    def preprocess(batch):
        enc = tokenizer(batch['message'], truncation=True, padding='max_length', max_length=seq_len)
        labels = [1 if r == 1 else 0 for r in batch['related']]
        return {"input_ids": enc['input_ids'], "attention_mask": enc['attention_mask'], "label": labels}
    proc = raw_ds.map(preprocess, batched=True, remove_columns=raw_ds['train'].column_names)
    proc.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return (
        DataLoader(proc['train'], batch_size=batch_size, shuffle=True),
        DataLoader(proc['validation'], batch_size=batch_size),
        DataLoader(proc['test'], batch_size=batch_size)
    )

def train_teacher(args):
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name)
    train_loader, val_loader, _ = prepare_dataloader(tokenizer, args.seq_len, args.batch_size)
    device = get_device() # Use MPS if available
    print(f"Using device for teacher: {device}")
    teacher = AutoModelForSequenceClassification.from_pretrained(
        args.teacher_model_name, num_labels=2
    ).to(device)
    # W&B watch for teacher model
    if args.use_wandb:
        wandb.watch(teacher, log='all', log_freq=100)
    optimizer = AdamW(teacher.parameters(), lr=args.teacher_lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.teacher_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    best_f1 = 0.0
    print(f"Training teacher '{args.teacher_model_name}' for {args.teacher_epochs} epochs...")
    for epoch in range(1, args.teacher_epochs + 1):
        teacher.train()
        total_loss = 0.0
        for batch in train_loader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = teacher(inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        teacher.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                logits = teacher(inputs, attention_mask=attention_mask).logits
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        val_f1 = f1_score(all_labels, all_preds)
        print(f"Teacher Epoch {epoch}: Loss={avg_loss:.4f}, Val F1={val_f1:.4f}")
        # log teacher metrics to W&B
        if args.use_wandb:
            wandb.log({
                'teacher/train_loss': avg_loss,
                'teacher/val_f1': val_f1,
                'epoch': epoch
            })
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(teacher.state_dict(), args.teacher_save_path)
            print(f"Saved best teacher (F1={best_f1:.4f})")
    teacher.load_state_dict(torch.load(args.teacher_save_path, map_location=device))
    teacher.eval()
    return teacher, tokenizer

def distill_student(args, teacher, tokenizer):
    # setup device and prepare data loaders
    device = get_device() # Use MPS if available
    print(f"Using device for student: {device}")
    train_loader, val_loader, test_loader = prepare_dataloader(tokenizer, args.seq_len, args.batch_size)
    # initialize student model on device
    student = SwarmFormerModel(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        seq_len=args.seq_len,
        cluster_size=args.cluster_size,
        num_layers=args.num_layers,
        T_local=args.T_local
    ).to(device)
    # W&B watch for student model
    if args.use_wandb:
        wandb.watch(student, log='all', log_freq=100)
    optimizer = AdamW(student.parameters(), lr=args.student_lr, weight_decay=args.weight_decay)
    best_f1 = 0.0
    print(f"Distilling student for {args.student_epochs} epochs...")
    for epoch in range(1, args.student_epochs + 1):
        student.train()
        for batch in train_loader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            with torch.no_grad():
                teacher_logits = teacher(inputs, attention_mask=attention_mask).logits
            student_logits = student(inputs)
            T = args.temperature
            loss_kd = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1),
                reduction='batchmean'
            ) * (T * T)
            loss_ce = F.cross_entropy(student_logits, labels)
            loss = args.alpha * loss_ce + (1 - args.alpha) * loss_kd
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        student.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                preds = torch.argmax(student(inputs), dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        val_f1 = f1_score(all_labels, all_preds)
        print(f"Student Epoch {epoch}: Val F1={val_f1:.4f}")
        # log student metrics to W&B
        if args.use_wandb:
            wandb.log({
                'student/val_f1': val_f1,
                'epoch': epoch
            })
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(student.state_dict(), args.student_save_path)
            print(f"Saved best student (F1={best_f1:.4f})")
    student.load_state_dict(torch.load(args.student_save_path, map_location=device))
    student.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            preds = torch.argmax(student(inputs), dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds)
    print(f"Distilled Student Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
    # log test metrics to W&B
    if args.use_wandb:
        wandb.log({
            'student/test_acc': test_acc,
            'student/test_f1': test_f1
        })
    sample = "There is flooding in my neighborhood, need help!"
    enc = tokenizer(sample, truncation=True, padding='max_length', max_length=args.seq_len, return_tensors='pt')
    with torch.no_grad():
        pred = torch.argmax(student(enc['input_ids'].to(device)), dim=1).item()
    print(f"Sample prediction: {pred}")

def main():
    parser = argparse.ArgumentParser(description="Knowledge distillation from BERT to SwarmFormer")
    parser.add_argument("--use_wandb", action='store_true', default=False, help="Enable Weights & Biases logging for experiment tracking")
    parser.add_argument("--teacher_model_name", type=str, default="answerdotai/ModernBERT-base", help="Name of the teacher model to fine-tune (ModernBERT-base recommended)")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--teacher_epochs", type=int, default=3)
    parser.add_argument("--student_epochs", type=int, default=6)
    parser.add_argument("--teacher_lr", type=float, default=2e-5)
    parser.add_argument("--student_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--cluster_size", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--T_local", type=int, default=2)
    parser.add_argument("--teacher_save_path", type=str, default="teacher_model.pt")
    parser.add_argument("--student_save_path", type=str, default="student_model.pt")
    args = parser.parse_args()
    # initialize W&B run
    if args.use_wandb:
        wandb.init(project="swarmformer-distillation", config=vars(args))
    # Load or train teacher
    if os.path.exists(args.teacher_save_path):
        device = get_device() # Use MPS if available
        print(f"Loading saved teacher on device: {device}")
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name)
        teacher = AutoModelForSequenceClassification.from_pretrained(
            args.teacher_model_name, num_labels=2
        ).to(device)
        teacher.load_state_dict(torch.load(args.teacher_save_path, map_location=device))
        teacher.eval()
    else:
        teacher, tokenizer = train_teacher(args)
    # Distill student
    distill_student(args, teacher, tokenizer)
    # finish W&B run
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main() 