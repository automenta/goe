import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoTokenizer
import numpy as np
import random



# --- Synthetic Datasets ---
class SequenceBaseDataset(Dataset):
    def __init__(self, num_samples, seq_len, vocab_size):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size # Excludes padding token 0
        self.sequences, self.labels = self._generate_data()

    def _generate_data(self):
        raise NotImplementedError

    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.sequences[idx], dtype=torch.long),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

class SequenceElementParityDataset(SequenceBaseDataset):
    def __init__(self, num_samples, seq_len, vocab_size, num_classes=2, target_token_id=1):
        self.num_classes = num_classes
        self.target_token_id = target_token_id
        super().__init__(num_samples, seq_len, vocab_size)
    
    def _generate_data(self):
        sequences = np.random.randint(1, self.vocab_size + 1, size=(self.num_samples, self.seq_len))
        counts = np.sum(sequences == self.target_token_id, axis=1)
        labels = counts % self.num_classes
        return sequences, labels

class SequenceCopyingDataset(SequenceBaseDataset): # Classify if a short pattern is present
    def __init__(self, num_samples, seq_len, vocab_size, num_classes=2, pattern_len=3):
        self.num_classes = num_classes
        self.pattern_len = min(pattern_len, seq_len // 2)
        self.pattern_to_copy = np.random.randint(1, vocab_size + 1, size=(self.pattern_len,))
        super().__init__(num_samples, seq_len, vocab_size)

    def _generate_data(self):
        sequences = np.random.randint(1, self.vocab_size + 1, size=(self.num_samples, self.seq_len))
        labels = np.zeros(self.num_samples, dtype=int)
        for i in range(self.num_samples):
            if random.random() < 0.5: # 50% chance to embed pattern
                start_idx = random.randint(0, self.seq_len - self.pattern_len)
                sequences[i, start_idx : start_idx + self.pattern_len] = self.pattern_to_copy
                labels[i] = 1 # Class 1 if pattern is present
            else:
                labels[i] = 0 # Class 0 if pattern is not present
        return sequences, labels

SYNTHETIC_DATASETS = {
    "parity": SequenceElementParityDataset,
    "copying": SequenceCopyingDataset,
}

def get_synthetic_data_loaders(dataset_name, batch_size, num_samples=1000, seq_len=20, vocab_size=10):
    if dataset_name not in SYNTHETIC_DATASETS: raise ValueError(f"Unknown synthetic dataset: {dataset_name}")
    
    DatasetClass = SYNTHETIC_DATASETS[dataset_name]
    # Assuming num_classes=2 for these simple synthetic tasks. Can be parameterized.
    train_dataset = DatasetClass(int(num_samples*0.8), seq_len, vocab_size)
    val_dataset = DatasetClass(int(num_samples*0.2), seq_len, vocab_size)
    
    actual_vocab_size = vocab_size + 1 # For padding token 0
    num_classes = train_dataset.num_classes

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)
    return train_loader, val_loader, actual_vocab_size, num_classes

# --- Real-World Datasets ---
TOKENIZER_NAME = "prajjwal1/bert-tiny" # Small tokenizer for VRAM

def get_real_world_data_loaders(dataset_name, max_length=64, batch_size=32, train_samples=1000, val_samples=200):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    dataset_config = None
    text_field = "text"
    label_field = "label"

    if dataset_name == "ag_news":
        pass # default text, label
    elif dataset_name == "imdb":
        pass # default text, label
    elif dataset_name == "sst2": # from GLUE
        dataset_config = "sst2"
        dataset_name = "glue"
        text_field = "sentence"
    elif dataset_name == "trec": # Fine-grained question classification
        # TREC has multiple configs, e.g. 'trec-finegrained'. Default is 'trec'.
        # Let's try default first. It might need specific handling for label names.
        # Check dataset viewer on HuggingFace for exact names.
        # For TREC, 'label-coarse' is often used.
        label_field = "coarse_label" if "coarse_label" in load_dataset(dataset_name, split="train", trust_remote_code=True).features else "label-coarse"
        if label_field not in load_dataset(dataset_name, split="train", trust_remote_code=True).features:
             label_field = "label" # Fallback
    else:
        raise ValueError(f"Unsupported real-world dataset: {dataset_name}")

    raw_datasets = load_dataset(dataset_name, name=dataset_config, trust_remote_code=True)
    
    # Use smaller of requested samples or available samples
    train_count = min(train_samples, len(raw_datasets['train']))
    # Use test set for validation if available and large enough, else train split.
    # Some datasets (like trec) might not have a validation split.
    val_split_name = 'validation' if 'validation' in raw_datasets else 'test'
    if val_split_name not in raw_datasets or len(raw_datasets[val_split_name]) < val_samples :
        val_split_name = 'test' # Fallback to test
        if val_split_name not in raw_datasets or len(raw_datasets[val_split_name]) < val_samples:
             # If test is also too small or non-existent, split from train
             print(f"Warning: Not enough samples in '{val_split_name}' for {dataset_name}. Splitting from train.")
             # This requires more complex splitting logic. For now, just use what's available.
             val_count = min(val_samples, len(raw_datasets[val_split_name]) if val_split_name in raw_datasets else 0)
        else:
            val_count = min(val_samples, len(raw_datasets[val_split_name]))
    else:
        val_count = min(val_samples, len(raw_datasets[val_split_name]))


    train_dataset_raw = raw_datasets["train"].shuffle(seed=42).select(range(train_count))
    val_dataset_raw = raw_datasets[val_split_name].shuffle(seed=42).select(range(val_count))
    
    def tokenize_and_format(examples):
        tokenized_inputs = tokenizer(examples[text_field], padding="max_length", truncation=True, max_length=max_length)
        tokenized_inputs["labels"] = examples[label_field]
        return tokenized_inputs

    cols_to_remove = [col for col in train_dataset_raw.column_names if col not in ["input_ids", "attention_mask", "labels", label_field, text_field]]
    
    train_dataset = train_dataset_raw.map(tokenize_and_format, batched=True, remove_columns=cols_to_remove + [text_field, label_field])
    val_dataset = val_dataset_raw.map(tokenize_and_format, batched=True, remove_columns=cols_to_remove + [text_field, label_field])
    
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)
    
    vocab_size = tokenizer.vocab_size
    # Infer num_classes from the label feature
    num_classes = raw_datasets["train"].features[label_field].num_classes
    
    return train_loader, val_loader, vocab_size, num_classes, tokenizer.pad_token_id

