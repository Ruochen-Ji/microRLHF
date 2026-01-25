"""
Reward Model

Learns to predict human preferences using the Bradley-Terry model.

Key concepts:
- Architecture: GPT-2 + scalar output head
- Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected))
- The reward model simulates human judgment

Phase 4 Implementation:
- RewardModel architecture (GPT backbone + value head)
- Preference pair training
- Evaluation on held-out preferences
"""

import torch
import torch.nn as nn

# TODO: Phase 4 Implementation


class RewardModel(nn.Module):
    """
    Reward model that predicts scalar rewards for text sequences.

    Architecture:
        GPT-2 backbone (frozen or fine-tuned) + scalar output head
    """

    def __init__(self, gpt_model, config=None):
        """
        Args:
            gpt_model: Pre-trained GPT model to use as backbone
            config: Reward model configuration
        """
        super().__init__()
        raise NotImplementedError("Phase 4: Reward model architecture")

    def forward(self, input_ids, attention_mask=None):
        """
        Compute reward for input sequence.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Optional attention mask

        Returns:
            rewards: Scalar rewards [batch]
        """
        raise NotImplementedError("Phase 4: Reward model forward")


class RewardTrainer:
    """Trainer for reward model using Bradley-Terry loss."""

    def __init__(self, model, config):
        raise NotImplementedError("Phase 4: Reward trainer")

    def compute_loss(self, chosen_rewards, rejected_rewards):
        """
        Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected))

        Args:
            chosen_rewards: Rewards for chosen responses [batch]
            rejected_rewards: Rewards for rejected responses [batch]

        Returns:
            loss: Scalar loss value
        """
        raise NotImplementedError("Phase 4: Bradley-Terry loss")

    def train(self, preference_dataset):
        """Train reward model on preference pairs."""
        raise NotImplementedError("Phase 4: Reward model training")
