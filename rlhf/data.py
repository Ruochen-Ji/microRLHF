"""
Preference Dataset Handling

Data formats and utilities for RLHF training.

Preference data format:
    {
        "prompt": "How do I make coffee?",
        "chosen": "Here's a step-by-step guide...",
        "rejected": "Coffee is a beverage..."
    }

Instruction data format (Alpaca):
    {
        "instruction": "What is the capital of France?",
        "input": "",  # Optional additional context
        "output": "The capital of France is Paris."
    }

Phase 2-3 Implementation:
    - InstructionDataset for SFT
    - PreferenceDataset for reward model and DPO
    - Data loading and preprocessing utilities
"""

import json
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class InstructionDataset(Dataset):
    """Dataset for instruction-following (SFT) training."""

    def __init__(self, data_path, tokenizer, max_length=512):
        """
        Args:
            data_path: Path to Alpaca-format JSON
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
        """
        raise NotImplementedError("Phase 2: Instruction dataset")

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        """
        Returns tokenized instruction-response pair.

        Format: "<instruction>\n<input>\n<response>"
        """
        raise NotImplementedError()


class PreferenceDataset(Dataset):
    """Dataset for preference-based training (reward model, DPO)."""

    def __init__(self, tokenizer, max_length=512, split="train"):
        """
        Args:
            tokenizer: tiktoken tokenizer (has .encode() and .eot_token)
            max_length: Max tokens per sequence (truncate/pad to this)
            split: "train" or "test"
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token = tokenizer.eot_token  # GPT-2's <|endoftext|>
        self.data = load_dataset("Anthropic/hh-rlhf", split=split)

    def __len__(self):
        return len(self.data)

    def _tokenize_and_pad(self, text):
        """Tokenize text, truncate/pad to max_length, build attention mask."""
        token_ids = self.tokenizer.encode(text)

        # Truncate if too long
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        real_length = len(token_ids)
        padding_needed = self.max_length - real_length

        # Attention mask: 1 = real token, 0 = padding
        attention_mask = [1] * real_length + [0] * padding_needed

        # Pad token_ids with eot_token
        token_ids = token_ids + [self.pad_token] * padding_needed

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )

    def __getitem__(self, idx):
        """
        Returns tokenized preference pair.

        Returns:
            dict with keys: chosen_ids, chosen_mask, rejected_ids, rejected_mask
        """
        example = self.data[idx]
        chosen_ids, chosen_mask = self._tokenize_and_pad(example["chosen"])
        rejected_ids, rejected_mask = self._tokenize_and_pad(example["rejected"])

        return {
            "chosen_ids": chosen_ids,
            "chosen_mask": chosen_mask,
            "rejected_ids": rejected_ids,
            "rejected_mask": rejected_mask,
        }


def load_alpaca_data(path):
    """Load Alpaca-format instruction data."""
    with open(path, 'r') as f:
        return json.load(f)


def load_preference_data(path):
    """Load preference data (prompt, chosen, rejected)."""
    with open(path, 'r') as f:
        return json.load(f)


def format_instruction(instruction, input_text="", response=""):
    """Format instruction into model input string."""
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{response}"
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
