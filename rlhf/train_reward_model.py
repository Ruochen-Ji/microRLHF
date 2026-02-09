"""
Train a reward model on Anthropic HH-RLHF human preference data.

Usage:
    python -m rlhf.train_reward_model
"""

import torch
import tiktoken

from model import GPT
from rlhf.reward_model import RewardModel, RewardTrainer
from rlhf.data import PreferenceDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# --- Load model ---
gpt = GPT.from_pretrained("gpt2")
model = RewardModel(gpt).to(device)
del gpt

# --- Load data ---
enc = tiktoken.get_encoding("gpt2")
train_ds = PreferenceDataset(tokenizer=enc, max_length=512, split="train")
eval_ds = PreferenceDataset(tokenizer=enc, max_length=512, split="test")
print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

# --- Train ---
trainer = RewardTrainer(model, lr=1e-5, device=device)
trainer.train(
    train_ds,
    eval_dataset=eval_ds,
    batch_size=4,
    num_epochs=3,
    log_interval=50,
    eval_interval=500,
    save_path="rlhf/logs/reward_model.pt",
)
