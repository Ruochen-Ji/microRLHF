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
from torch.utils.data import Dataset

# TODO: Phase 2-3 Implementation


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

    def __init__(self, data_path, tokenizer, max_length=512):
        """
        Args:
            data_path: Path to preference JSON
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
        """
        raise NotImplementedError("Phase 3: Preference dataset")

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        """
        Returns tokenized preference triple.

        Returns:
            dict with keys: prompt_ids, chosen_ids, rejected_ids
        """
        raise NotImplementedError()


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
