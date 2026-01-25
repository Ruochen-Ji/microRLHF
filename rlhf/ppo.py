"""
PPO (Proximal Policy Optimization) for RLHF

Uses the reward model to improve the policy via reinforcement learning.

Architecture:
    - Policy (trainable GPT-2): Generates responses
    - Reference (frozen): Computes KL penalty
    - Reward Model (frozen): Scores responses
    - Value Head (trainable): Estimates expected reward

Key concepts:
    - KL penalty prevents reward hacking
    - R_total = R_reward(y|x) - beta * KL(policy || reference)
    - PPO clipping for stable updates

Phase 5 Implementation:
    5.1 Architecture setup (policy, reference, value head)
    5.2 Training loop (generate, score, compute advantages, update)
    5.3 Ablations (KL penalty effects, beta tuning)
"""

import torch
import torch.nn as nn

# TODO: Phase 5 Implementation


class ValueHead(nn.Module):
    """Value head for PPO advantage estimation."""

    def __init__(self, hidden_size):
        super().__init__()
        raise NotImplementedError("Phase 5.1: Value head")

    def forward(self, hidden_states):
        """Predict value from last hidden state."""
        raise NotImplementedError("Phase 5.1: Value head forward")


class PPOTrainer:
    """
    PPO trainer for RLHF.

    Components:
        - policy_model: Trainable GPT model
        - ref_model: Frozen reference model (initial policy)
        - reward_model: Trained reward model
        - value_head: For advantage estimation
    """

    def __init__(
        self,
        policy_model,
        ref_model,
        reward_model,
        config,
    ):
        """
        Args:
            policy_model: Trainable GPT policy
            ref_model: Frozen reference model
            reward_model: Trained reward model
            config: PPO configuration
        """
        raise NotImplementedError("Phase 5.1: PPO trainer setup")

    def generate_responses(self, prompts):
        """Generate responses from current policy."""
        raise NotImplementedError("Phase 5.2: Response generation")

    def compute_rewards(self, responses):
        """Score responses with reward model."""
        raise NotImplementedError("Phase 5.2: Reward computation")

    def compute_kl_penalty(self, policy_logprobs, ref_logprobs):
        """
        Compute KL divergence penalty.

        KL(policy || ref) = sum(policy * log(policy/ref))
        """
        raise NotImplementedError("Phase 5.2: KL penalty")

    def compute_advantages(self, rewards, values):
        """Compute GAE advantages for PPO update."""
        raise NotImplementedError("Phase 5.2: Advantage estimation")

    def ppo_update(self, batch):
        """
        PPO update with clipped objective.

        L = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
        """
        raise NotImplementedError("Phase 5.2: PPO update")

    def train_step(self, prompts):
        """Single PPO training step."""
        raise NotImplementedError("Phase 5.2: Training step")

    def train(self, prompt_dataset, num_epochs):
        """Full PPO training loop."""
        raise NotImplementedError("Phase 5.2: Training loop")
