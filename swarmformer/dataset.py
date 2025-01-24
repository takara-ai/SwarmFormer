import torch
import random
import os
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class IMDbAugmenter:
    def __init__(self):
        # Download all required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('wordnet')
            nltk.download('omw-1.4')  # Open Multilingual Wordnet
        
        from nltk.corpus import wordnet
        self.wordnet = wordnet
        
    def _shuffle_sentences(self, text: str) -> str:
        """Shuffle sentences while maintaining some local structure"""
        sentences = sent_tokenize(text)
        if len(sentences) <= 2:
            return text
            
        # Keep some sentences together to preserve local context
        chunks = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and random.random() < 0.7:  # 70% chance to keep pairs
                chunks.append(sentences[i:i+2])
                i += 2
            else:
                chunks.append([sentences[i]])
                i += 1
                
        random.shuffle(chunks)
        return ' '.join([' '.join(chunk) for chunk in chunks])
    
    def _swap_synonyms(self, text: str, p: float = 0.1) -> str:
        """Replace words with synonyms while preserving sentiment"""
        words = text.split()
        for i in range(len(words)):
            if random.random() < p:
                synonyms = []
                for syn in self.wordnet.synsets(words[i]):
                    for lemma in syn.lemmas():
                        if lemma.name() != words[i]:
                            synonyms.append(lemma.name())
                if synonyms:
                    words[i] = random.choice(synonyms)
        return ' '.join(words)
    
    def _create_hierarchical_sample(self, texts: List[str]) -> str:
        """Combine multiple reviews in a hierarchical way"""
        # Select 2-3 reviews
        num_reviews = random.randint(2, 3)
        selected = random.sample(texts, num_reviews)
        
        # Create a hierarchical structure
        result = []
        for i, text in enumerate(selected):
            prefix = random.choice([
                "Additionally, ", "Furthermore, ", "Moreover, ",
                "In contrast, ", "Similarly, ", "On the other hand, "
            ]) if i > 0 else ""
            result.append(prefix + text)
            
        return " ".join(result)
    
    def augment_dataset(self, texts: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
        """Apply multiple augmentation strategies"""
        augmented_texts = []
        augmented_labels = []
        
        # Original samples
        augmented_texts.extend(texts)
        augmented_labels.extend(labels)
        
        # Group texts by sentiment
        pos_texts = [t for t, l in zip(texts, labels) if l == 1]
        neg_texts = [t for t, l in zip(texts, labels) if l == 0]
        
        # 1. Sentence shuffling
        for text, label in zip(texts, labels):
            if random.random() < 0.3:  # 30% chance
                augmented_texts.append(self._shuffle_sentences(text))
                augmented_labels.append(label)
        
        # 2. Synonym replacement
        for text, label in zip(texts, labels):
            if random.random() < 0.3:
                augmented_texts.append(self._swap_synonyms(text))
                augmented_labels.append(label)
        
        # 3. Hierarchical combinations
        for _ in range(len(texts) // 4):  # Add 25% more samples
            if random.random() < 0.5:
                text = self._create_hierarchical_sample(pos_texts)
                augmented_texts.append(text)
                augmented_labels.append(1)
            else:
                text = self._create_hierarchical_sample(neg_texts)
                augmented_texts.append(text)
                augmented_labels.append(0)
        
        return augmented_texts, augmented_labels

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', seq_len=16, max_samples=1000, seed=42, augment=True):
        super().__init__()
        random.seed(seed)
        
        # Load IMDb dataset
        dataset = load_dataset("imdb", split=split)
        
        # Sample if needed
        if max_samples and max_samples < len(dataset):
            indices = random.sample(range(len(dataset)), max_samples)
            dataset = dataset.select(indices)
        
        texts = [item['text'] for item in dataset]
        labels = [item['label'] for item in dataset]
        
        # Apply augmentation for training set
        if split == 'train' and augment:
            augmenter = IMDbAugmenter()
            texts, labels = augmenter.augment_dataset(texts, labels)
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Process the texts
        self.data = []
        self.labels = []
        
        for text, label in zip(texts, labels):
            encoding = self.tokenizer(
                text,
                max_length=seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            self.data.append(encoding['input_ids'][0])
            self.labels.append(label)
        
        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def vocab_size(self):
        return self.tokenizer.vocab_size 