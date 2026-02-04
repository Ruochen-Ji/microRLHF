"""
rl_utils.py - Core utilities for PPO training (built from scratch for learning)
"""
import torch
import torch.nn.functional as F
import warnings


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


def compute_kl_penalty(model, ref_model, generated_ids, prompt_length):
    """
    Compute KL divergence between current policy and reference model.

    Why do we need this?
    - Without KL penalty, the model finds "shortcuts" to maximize reward
    - Example: To maximize length reward, it could just output "the the the the..."
    - KL penalty says: "stay close to how the original model would respond"
    - This preserves language quality while optimizing for reward

    The math:
        KL(P || Q) = sum over x of P(x) * log(P(x) / Q(x))
                   = sum over x of P(x) * (log P(x) - log Q(x))

        Here:
        - P = current policy (model being trained)
        - Q = reference policy (frozen original model)

    We compute KL at each response token position, then average.

    Args:
        model: current policy (being trained)
        ref_model: reference policy (frozen, original model)
        generated_ids: (batch_size, total_length) - the sequences we generated
        prompt_length: int - where the response starts

    Returns:
        kl: scalar - mean KL divergence across batch and positions
    """
    # Get logits from both models for the generated sequence
    # We need logits at positions [prompt_length-1, ..., total_length-2]
    # because position i predicts token i+1

    # IMPORTANT: nanoGPT's forward() only returns the last position's logits
    # when targets=None. We pass targets to get full logits for all positions.
    input_ids = generated_ids[:, :-1]   
    targets = generated_ids[:, 1:].contiguous()  # Must be contiguous for view()

    # Forward pass through reference model (no gradients needed)
    with torch.no_grad():
        """
        Why generated_ids[:, :-1]?
        This is the "shift by 1" for next-token prediction:
        - Input: tokens [0, 1, 2, 3, 4]
        - Model predicts: [1, 2, 3, 4, 5]
        - Position i in logits predicts token i+1
        """
        ref_logits, _ = ref_model(input_ids, targets=targets)  # (batch, seq_len-1, vocab)
        # Forward pass, keep only response positions (predicting response tokens)
        """
        Why prompt_length - 1?
        - Position prompt_length - 1 predicts the first response token
        - We only care about KL over response tokens (not the prompt)
        """
        ref_logits = ref_logits[:, prompt_length - 1:, :]  # (batch, response_len, vocab)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1) # (batch, response_len, vocab)

    # Forward pass through current model (need gradients for training)
    current_logits, _ = model(input_ids, targets=targets)
    current_logits = current_logits[:, prompt_length - 1:, :]
    current_log_probs = F.log_softmax(current_logits, dim=-1) # (batch, response_len, vocab)

    # Convert current log probs to probs for the KL formula
    # exp() here because of the log_softmax function.
    current_probs = torch.exp(current_log_probs) # (batch, response_len, vocab)

    # KL(current || ref) = sum over vocab of current_prob * (current_log_prob - ref_log_prob)
    # This is computed per position, then we average
    # Shape: (batch, response_len, vocab) -> sum over vocab -> (batch, response_len)
    kl_per_position = (current_probs * (current_log_probs - ref_log_probs)).sum(dim=-1)

    # Average over positions and batch
    kl = kl_per_position.mean()

    return kl


def compute_policy_loss(
    model,
    generated_ids,
    old_log_probs,
    advantages,
    prompt_length,
    clip_epsilon=0.2
):
    """
    Compute the PPO clipped surrogate objective.

    The key idea of PPO:
    - We want to update the policy to increase probability of good actions
    - But large updates can be unstable (policy might change too much)
    - Solution: Clip the update to stay within a "trust region"

    The math:
        ratio = π_θ(a|s) / π_old(a|s)     # How much more/less likely is action now?

        unclipped = ratio * advantage      # Naive policy gradient
        clipped = clip(ratio, 1-ε, 1+ε) * advantage  # Constrained update

        loss = -min(unclipped, clipped)    # Take the pessimistic bound

    Why the min?
        - If advantage > 0 (good action): we want to increase probability
          - But clip prevents ratio from going above 1+ε
          - Prevents "too confident" updates
        - If advantage < 0 (bad action): we want to decrease probability
          - But clip prevents ratio from going below 1-ε
          - Prevents "too aggressive" decreases

    Args:
        model: current policy (being trained)
        generated_ids: (batch_size, total_length) - sequences we generated
        old_log_probs: (batch_size, response_length) - log probs at generation time
        advantages: (batch_size,) - how much better than baseline (reward - baseline)
        prompt_length: int - where response starts
        clip_epsilon: float - PPO clipping parameter (typically 0.2)

    Returns:
        policy_loss: scalar - the PPO loss to minimize
        stats: dict - ratio statistics for monitoring
            - ratio_mean, ratio_min, ratio_max
            - clip_fraction: how often we hit the clipping boundary
            - approx_kl: approximate KL from ratio (for early stopping)
    """
    model.train()

    # =========================================================================
    # Token-Level PPO Implementation
    # =========================================================================
    # Instead of treating the whole response as one action (sequence-level),
    # we compute the PPO objective for each token separately, then average.
    # This prevents ratio explosion from summing log probs.
    #
    # Key insight:
    #   Sequence-level: ratio = exp(Σ log_diff) → can explode to 10^9 or collapse to 10^-9
    #   Token-level:    ratio[t] = exp(log_diff[t]) → stays in [0.1, 10] range
    # =========================================================================

    # Get current policy's log probs for the same tokens we generated
    # IMPORTANT: nanoGPT's forward() only returns the last position's logits
    # when targets=None (an inference optimization). We pass targets to get full logits.
    input_ids = generated_ids[:, :-1]
    targets = generated_ids[:, 1:].contiguous()  # Must be contiguous for view()
    logits, _ = model(input_ids, targets=targets)  # (batch, seq_len-1, vocab)

    # Keep only response positions
    # Position (prompt_length - 1) predicts the first response token
    logits = logits[:, prompt_length - 1:, :]  # (batch, response_len, vocab)
    log_probs = F.log_softmax(logits, dim=-1)

    # Get the log prob of each token we actually generated
    response_ids = generated_ids[:, prompt_length:]  # (batch, response_len)

    # Gather the log prob of each generated token
    # current_log_probs[i, t] = log P(token_t | context) under current policy
    current_log_probs = log_probs.gather(
        dim=-1,
        index=response_ids.unsqueeze(-1)  # (batch, response_len, 1)
    ).squeeze(-1)  # (batch, response_len)

    # =========================================================================
    # Token-level ratio computation (the key change!)
    # =========================================================================
    # old_log_probs is (batch, response_len) from generate()
    # Compute ratio per token: π_new(token_t) / π_old(token_t)
    ratio = torch.exp(current_log_probs - old_log_probs)  # (batch, response_len)

    # Broadcast advantages to all tokens in each response
    # advantages is (batch,) → (batch, response_len)
    # All tokens in a response share the same advantage (simplified credit assignment)
    response_len = current_log_probs.shape[1]
    advantages_expanded = advantages.unsqueeze(1).expand(-1, response_len)  # (batch, response_len)

    # =========================================================================
    # PPO clipped objective (per-token)
    # =========================================================================
    # Unclipped objective: ratio * advantage
    unclipped = ratio * advantages_expanded  # (batch, response_len)

    # Clipped objective: constrain ratio to [1-ε, 1+ε]
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    clipped = clipped_ratio * advantages_expanded  # (batch, response_len)

    # Take the minimum (pessimistic bound) per token, then average over all tokens
    # This is the key: we AVERAGE token losses instead of computing one sequence ratio
    policy_loss = -torch.min(unclipped, clipped).mean()

    # =========================================================================
    # Statistics for monitoring
    # =========================================================================
    with torch.no_grad():
        # Clip fraction: what fraction of tokens hit the clipping boundary?
        clipped_mask = (ratio < 1 - clip_epsilon) | (ratio > 1 + clip_epsilon)
        clip_fraction = clipped_mask.float().mean().item()

        # Approximate KL divergence per token, then average
        approx_kl = (0.5 * (ratio - 1) ** 2).mean().item()

        stats = {
            "ratio_mean": ratio.mean().item(),
            "ratio_min": ratio.min().item(),
            "ratio_max": ratio.max().item(),
            "clip_fraction": clip_fraction,
            "approx_kl": approx_kl,
        }

    return policy_loss, stats


def check_training_health(stats, step, thresholds=None):
    """
    Check PPO training health and log warnings if metrics are off.

    This helps catch training issues early:
    - Ratio too extreme → policy changed too much, stale data
    - Clip fraction too high → updates always hitting boundary
    - Approx KL too high → consider early stopping this PPO epoch

    Args:
        stats: dict from compute_policy_loss()
        step: current training step (for logging)
        thresholds: dict of thresholds (optional, uses defaults)

    Returns:
        healthy: bool - True if all metrics are within thresholds
    """
    if thresholds is None:
        thresholds = {
            "ratio_min": 0.1,      # ratio below this is concerning
            "ratio_max": 10.0,     # ratio above this is concerning
            "clip_fraction": 0.5,  # more than 50% clipped is too much
            "approx_kl": 0.1,      # KL above this suggests early stopping
        }

    healthy = True
    issues = []

    # Check ratio bounds
    if stats["ratio_min"] < thresholds["ratio_min"]:
        issues.append(
            f"ratio_min={stats['ratio_min']:.4f} < {thresholds['ratio_min']} "
            "(policy probability collapsed for some samples)"
        )
        healthy = False

    if stats["ratio_max"] > thresholds["ratio_max"]:
        issues.append(
            f"ratio_max={stats['ratio_max']:.4f} > {thresholds['ratio_max']} "
            "(policy probability exploded for some samples)"
        )
        healthy = False

    # Check clip fraction
    if stats["clip_fraction"] > thresholds["clip_fraction"]:
        issues.append(
            f"clip_fraction={stats['clip_fraction']:.2%} > {thresholds['clip_fraction']:.0%} "
            "(too many samples hitting clip boundary)"
        )
        healthy = False

    # Check approximate KL (soft warning, might want early stopping)
    if stats["approx_kl"] > thresholds["approx_kl"]:
        issues.append(
            f"approx_kl={stats['approx_kl']:.4f} > {thresholds['approx_kl']} "
            "(consider reducing PPO epochs or learning rate)"
        )
        # This is a soft warning, don't mark as unhealthy
        # healthy = False

    # Log warnings
    if issues:
        warnings.warn(
            f"\n[Step {step}] Training health check failed:\n  - " +
            "\n  - ".join(issues)
        )

    return healthy 