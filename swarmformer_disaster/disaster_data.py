# disaster_data.py
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def get_disaster_dataloaders(tokenizer_name="bert-base-uncased", seq_len=64, batch_size=32):
    """
    Returns train, validation, and test DataLoaders (and tokenizer) for binary disaster-related classification
    based on the HuggingFace 'disaster_response_messages' dataset.
    Filters out entries where 'related' == 2, then maps 'related' 1->1, 0->0.
    """
    raw_ds = load_dataset("disaster_response_messages")
    # Filter out ambiguous examples (related == 2)
    raw_ds = raw_ds.filter(lambda example: example["related"] != 2)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def preprocess(batch):
        enc = tokenizer(batch["message"], truncation=True, padding="max_length", max_length=seq_len)
        # Map related labels to binary (1 if disaster-related, else 0)
        labels = [1 if r == 1 else 0 for r in batch["related"]]
        return {"input_ids": enc["input_ids"], "label": labels}

    # Apply tokenization and label mapping
    proc_ds = raw_ds.map(preprocess, batched=True, remove_columns=raw_ds["train"].column_names)
    proc_ds.set_format(type="torch", columns=["input_ids", "label"])

    train_loader = DataLoader(proc_ds["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(proc_ds["validation"], batch_size=batch_size)
    test_loader = DataLoader(proc_ds["test"], batch_size=batch_size)
    return train_loader, val_loader, test_loader, tokenizer 