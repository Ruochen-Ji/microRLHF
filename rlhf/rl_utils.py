"""
rl_utils.py - Core utilities for PPO training (built from scratch for learning)
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, temperature=1.0):
    """
    Generate response tokens from the model, collecting log probabilities.

    This is different from standard generation because we need to record
    the probability of each action (token) we take. PPO will later compare
    these "old" probabilities with "current" probabilities to compute
    the policy ratio.

    Args:
        model: GPT model (from model.py)
        prompt_ids: (batch_size, prompt_length) - tokenized prompts
        max_new_tokens: how many tokens to generate
        temperature: sampling temperature (higher = more random)

    Returns:
        generated_ids: (batch_size, prompt_length + max_new_tokens) - full sequences
        log_probs: (batch_size, max_new_tokens) - log P(token) for each generated token

    Key insight: We sample token ~ P(token|context), then record log P(token).
    This is the "old" log prob that PPO will use for importance sampling.
    """
    model.eval()
    device = prompt_ids.device

    # Start with the prompt
    # (batch_size, prompt_length)
    generated_ids = prompt_ids.clone()
    log_probs_list = []

    for _ in range(max_new_tokens):
        # Forward pass: get logits for all positions
        # model returns (logits, loss) - we only need logits
        logits, _ = model(generated_ids)

        # We only care about the LAST position (next token prediction)
        # Shape: (batch_size, vocab_size)
        next_token_logits = logits[:, -1, :]

        # Apply temperature (divide logits, not probs!)
        # Higher temperature -> flatter distribution -> more random
        next_token_logits = next_token_logits / temperature

        # Convert to probabilities
        # (batch_size, vocab_size)
        probs = F.softmax(next_token_logits, dim=-1)

        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Compute log probability of the token we just sampled
        # This is log P(next_token | context) under current policy
        # This probability will be used to calculate probablity ratio 
        # Gather is a little tricky, but the mental model is that the output dimension should match the index dimension.
        # The dim in the gather should be the dimension that we are interested in. 
        # When dim=0, we are picking rows. When dim=1, we are picking columns.
        log_prob = torch.log(probs.gather(dim=-1, index=next_token))  # (batch_size, 1)
        log_probs_list.append(log_prob.squeeze(-1))  # (batch_size,)

        # Append token to sequence
        # (batch_size, cur_prompt_length + 1)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

    # Stack log probs: (batch_size, max_new_tokens)
    log_probs = torch.stack(log_probs_list, dim=1)

    return generated_ids, log_probs


def compute_reward(generated_ids, prompt_length, max_new_tokens, eos_token_id):
    """
    Compute reward based on response length (up to first EOS token).

    Our heuristic: longer responses = higher reward.
    This is intentionally simple so we can:
    1. Verify the RL loop is working (reward should increase)
    2. Observe reward hacking (model finds degenerate ways to be "long")

    Args:
        generated_ids: (batch_size, total_length) - prompt + response tokens
        prompt_length: int - where the response starts
        max_new_tokens: int - maximum response length (for normalization)
        eos_token_id: int - the end-of-sequence token ID (50256 for GPT-2)

    Returns:
        rewards: (batch_size,) - scalar reward per example

    Normalization:
        We normalize to roughly [-1, 1] range for stable training.
        reward = (length / max_length) * 2 - 1
        - Length 0 → reward = -1
        - Length max → reward = +1

    Why normalize?
        - Raw lengths (0 to 100) create huge gradients
        - Normalized rewards keep gradient magnitudes stable
        - Makes hyperparameters (like kl_coef) more transferable
    """
    # Extract just the response part (everything after the prompt)
    response_ids = generated_ids[:, prompt_length:]  # (batch_size, max_new_tokens)
    batch_size = response_ids.shape[0]

    # Find the position of first EOS token in each sequence
    # If no EOS found, length = max_new_tokens (full response)
    response_lengths = torch.zeros(batch_size, device=generated_ids.device)

    for i in range(batch_size):
        # Find positions where token == EOS
        # response_ids[i]==eos_token_id will return a boolean tensor of shape (max_new_tokens,)
        # nonzero returns the indices of the true values.
        eos_positions = (response_ids[i] == eos_token_id).nonzero(as_tuple=True)[0]

        if len(eos_positions) > 0:
            # Length is position of first EOS (0-indexed, so position = count of tokens before EOS)
            response_lengths[i] = eos_positions[0].float()
        else:
            # No EOS found - full response length
            response_lengths[i] = max_new_tokens

    # Normalize to [-1, 1], we first make reward between 0 and 1, then stretch out to [0, 2], then shift to [-1, 1].
    # length=0 -> -1, length=max -> +1
    rewards = (response_lengths / max_new_tokens) * 2 - 1

    return rewards 