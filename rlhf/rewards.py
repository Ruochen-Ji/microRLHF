"""
rewards.py - Collection of reward functions for RLHF experiments

Each reward function demonstrates different behaviors and failure modes:
- LengthReward: Rewards moderate length (50-80 tokens) - leads to verbosity hacking
- BrevityReward: Rewards short responses - leads to EOS spamming
- TargetLengthReward: Rewards specific length range - leads to padding/truncation

All reward functions follow the same interface:
    reward = compute(generated_ids, prompt_length, max_new_tokens, eos_token_id)
    Returns: torch.Tensor of shape (batch_size,)
"""

import torch


class LengthReward:
    """
    Rewards moderate length responses within an optimal range.

    Failure mode: Model learns to generate verbose, rambling text
    to maximize length without saying anything meaningful.

    Args:
        optimal_min (int): Minimum tokens for optimal reward range. Default: 50.
        optimal_max (int): Maximum tokens for optimal reward range. Default: 80.
        no_eos_penalty (float): Penalty applied when response has no EOS token. Default: -0.5.

    Reward schedule:
        - length < 10: reward in [-1.0, -0.5] (very short is bad)
        - length in [10, optimal_min): reward in [-0.5, 0.0]
        - length in [optimal_min, optimal_max]: reward in [0.0, 0.5] (optimal)
        - length > optimal_max: reward decreases from 0.5, capped at -0.5
        - no EOS: additional penalty of no_eos_penalty
    """

    def __init__(self, optimal_min=50, optimal_max=80, no_eos_penalty=-0.5):
        self.optimal_min = optimal_min
        self.optimal_max = optimal_max
        self.no_eos_penalty = no_eos_penalty

    def compute(self, generated_ids, prompt_length, max_new_tokens, eos_token_id):
        """
        Compute rewards for a batch of generated sequences.

        Args:
            generated_ids: Tensor of shape (batch_size, seq_len) containing
                token IDs for prompt + generated response.
            prompt_length: int, number of tokens in the prompt. Response starts
                at index prompt_length.
            max_new_tokens: int, maximum number of tokens that could be generated.
                Used to determine length when no EOS is found.
            eos_token_id: int, the token ID representing end-of-sequence.

        Returns:
            rewards: Tensor of shape (batch_size,) with float rewards in [-1.5, 0.5].
        """
        rewards = []

        for i in range(generated_ids.shape[0]):
            response = generated_ids[i, prompt_length:]

            # Find EOS position
            eos_positions = (response == eos_token_id).nonzero(as_tuple=True)[0]

            if len(eos_positions) > 0:
                length = eos_positions[0].item()
                has_eos = True
            else:
                length = max_new_tokens
                has_eos = False

            # Length-based reward
            if length < 10:
                reward = -1.0 + (length / 10) * 0.5  # Very short is bad
            elif length < self.optimal_min:
                reward = -0.5 + (length - 10) / (self.optimal_min - 10) * 0.5
            elif length <= self.optimal_max:
                reward = 0.0 + (length - self.optimal_min) / (self.optimal_max - self.optimal_min) * 0.5
            else:
                # Longer than optimal - slight penalty
                excess = length - self.optimal_max
                reward = 0.5 - (excess / 100) * 0.5
                reward = max(reward, -0.5)

            # Penalty for no EOS
            if not has_eos:
                reward += self.no_eos_penalty

            rewards.append(reward)

        return torch.tensor(rewards, device=generated_ids.device, dtype=torch.float32)


class BrevityReward:
    """
    Rewards SHORT responses - the shorter the better.

    Failure mode: Model learns to output EOS immediately or after
    just a few tokens, giving useless responses like "Yes." or "".

    Args:
        max_good_length (int): Responses under this length get positive reward. Default: 30.
        eos_bonus (float): Bonus added when response ends with EOS. Default: 0.5.

    Reward schedule:
        - length in [0, max_good_length]: reward = 1.0 - (length / max_good_length)
          (linearly decreases from 1.0 to 0.0)
        - length > max_good_length: reward = negative, capped at -1.0
        - has EOS: additional bonus of eos_bonus
    """

    def __init__(self, max_good_length=30, eos_bonus=0.5):
        self.max_good_length = max_good_length
        self.eos_bonus = eos_bonus

    def compute(self, generated_ids, prompt_length, max_new_tokens, eos_token_id):
        """
        Compute rewards for a batch of generated sequences.

        Args:
            generated_ids: Tensor of shape (batch_size, seq_len) containing
                token IDs for prompt + generated response.
            prompt_length: int, number of tokens in the prompt. Response starts
                at index prompt_length.
            max_new_tokens: int, maximum number of tokens that could be generated.
                Used to determine length when no EOS is found.
            eos_token_id: int, the token ID representing end-of-sequence.

        Returns:
            rewards: Tensor of shape (batch_size,) with float rewards in [-1.0, 1.5].
        """
        rewards = []

        for i in range(generated_ids.shape[0]):
            response = generated_ids[i, prompt_length:]

            # Find EOS position
            eos_positions = (response == eos_token_id).nonzero(as_tuple=True)[0]

            if len(eos_positions) > 0:
                length = eos_positions[0].item()
                has_eos = True
            else:
                length = max_new_tokens
                has_eos = False

            # Reward formula: shorter = better
            # reward = 1.0 at length 0, decreases as length increases
            if length <= self.max_good_length:
                # Linear from 1.0 (at 0) to 0.0 (at max_good_length)
                reward = 1.0 - (length / self.max_good_length)
            else:
                # Negative reward for long responses
                excess = length - self.max_good_length
                reward = -excess / self.max_good_length
                reward = max(reward, -1.0)  # Cap at -1

            # Bonus for having EOS (natural completion)
            if has_eos:
                reward += self.eos_bonus

            rewards.append(reward)

        return torch.tensor(rewards, device=generated_ids.device, dtype=torch.float32)


class TargetLengthReward:
    """
    Rewards responses within a specific target length range.

    Failure mode: Model learns to pad short answers with filler text
    or abruptly cut off long answers to hit the target range.

    Args:
        target_min (int): Minimum tokens for maximum reward. Default: 100.
        target_max (int): Maximum tokens for maximum reward. Default: 150.

    Reward schedule:
        - length in [target_min, target_max]: reward = 1.0 (perfect)
        - length < target_min: reward = -1.0 + (length / target_min)
          (linearly increases toward 0.0)
        - length > target_max: reward decreases from 1.0, capped at -1.0
    """

    def __init__(self, target_min=100, target_max=150):
        self.target_min = target_min
        self.target_max = target_max

    def compute(self, generated_ids, prompt_length, max_new_tokens, eos_token_id):
        """
        Compute rewards for a batch of generated sequences.

        Args:
            generated_ids: Tensor of shape (batch_size, seq_len) containing
                token IDs for prompt + generated response.
            prompt_length: int, number of tokens in the prompt. Response starts
                at index prompt_length.
            max_new_tokens: int, maximum number of tokens that could be generated.
                Used to determine length when no EOS is found.
            eos_token_id: int, the token ID representing end-of-sequence.

        Returns:
            rewards: Tensor of shape (batch_size,) with float rewards in [-1.0, 1.0].
        """
        rewards = []

        for i in range(generated_ids.shape[0]):
            response = generated_ids[i, prompt_length:]

            # Find EOS position
            eos_positions = (response == eos_token_id).nonzero(as_tuple=True)[0]

            if len(eos_positions) > 0:
                length = eos_positions[0].item()
            else:
                length = max_new_tokens

            # Reward: 1.0 if in range, penalty proportional to distance otherwise
            if self.target_min <= length <= self.target_max:
                reward = 1.0
            elif length < self.target_min:
                # Too short
                reward = -1.0 + (length / self.target_min)
            else:
                # Too long
                excess = length - self.target_max
                reward = 1.0 - (excess / 50)
                reward = max(reward, -1.0)

            rewards.append(reward)

        return torch.tensor(rewards, device=generated_ids.device, dtype=torch.float32)


def compute_reward(generated_ids, prompt_length, max_new_tokens, eos_token_id):
    """
    Default reward function using LengthReward with default parameters.

    Args:
        generated_ids: Tensor of shape (batch_size, seq_len) containing
            token IDs for prompt + generated response.
        prompt_length: int, number of tokens in the prompt.
        max_new_tokens: int, maximum number of tokens that could be generated.
        eos_token_id: int, the token ID representing end-of-sequence.

    Returns:
        rewards: Tensor of shape (batch_size,) with float rewards.
    """
    reward_fn = LengthReward()
    return reward_fn.compute(generated_ids, prompt_length, max_new_tokens, eos_token_id)
