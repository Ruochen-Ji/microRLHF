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
    """
    Value head for PPO advantage estimation.

    Takes hidden states from the transformer and predicts expected future reward.
    This is a simple single-layer linear projection:

        hidden_state (768-dim) --> linear --> scalar value

    Why so simple?
    - The transformer has already learned rich representations
    - The value function just needs to "read off" the expected reward
    - More complex heads (MLP) work too but aren't necessary for learning
    """

    def __init__(self, hidden_size):
        super().__init__()
        # Linear projection: hidden_size -> 1
        # Each position gets its own value estimate
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        """
        Predict value from hidden states.

        Args:
            hidden_states: (batch, seq_len, hidden_size) from transformer

        Returns:
            values: (batch, seq_len) - value estimate per position

        Example:
            hidden_states shape: (4, 100, 768)  # batch=4, seq_len=100, hidden=768
            output shape: (4, 100)              # one value per position
        """
        # (batch, seq_len, hidden_size) -> (batch, seq_len, 1) -> (batch, seq_len)
        return self.linear(hidden_states).squeeze(-1)


def compute_gae(
    values: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).

    This is the heart of per-token credit assignment. Instead of giving all tokens
    the same advantage (reward - baseline), GAE computes how much each token
    contributed to the final outcome.

    The Algorithm:
    ==============
    1. Compute TD errors (Temporal Difference):
       δ_t = r_t + γ·V(s_{t+1}) - V(s_t)

       - r_t is the reward at step t (usually 0 except at the last token)
       - V(s_t) is our value estimate at position t
       - This measures: "Was this transition better or worse than expected?"

    2. Compute GAE (weighted sum of future TD errors):
       A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...

       - λ (lambda) controls bias-variance tradeoff
       - λ=0: Only use immediate TD error (high bias, low variance)
       - λ=1: Use all future TD errors (low bias, high variance)
       - λ=0.95: Sweet spot

    3. Efficient backward computation:
       A_{T-1} = δ_{T-1}
       A_{T-2} = δ_{T-2} + (γλ)·A_{T-1}
       ...and so on

    Args:
        values: (batch, response_len) - V(s_t) for each position in response
        rewards: (batch, response_len) - r_t for each position
                 (typically zeros except last position has the actual reward)
        gamma: discount factor (0.99 = care about future rewards)
        lam: GAE lambda for bias-variance tradeoff (0.95 is common)

    Returns:
        advantages: (batch, response_len) - A_t for each token
        returns: (batch, response_len) - V_target = A_t + V(s_t), used for value loss

    Example:
        Suppose we have a 5-token response with reward=1.0 at the end:

        rewards = [0, 0, 0, 0, 1.0]
        values  = [0.3, 0.4, 0.5, 0.7, 0.9]  (model's predictions)

        TD errors (with γ=1.0 for simplicity):
        δ_4 = 1.0 + 0 - 0.9 = 0.1       (last token: got reward, minus value)
        δ_3 = 0 + 0.9 - 0.7 = 0.2       (value increased, good transition)
        δ_2 = 0 + 0.7 - 0.5 = 0.2
        δ_1 = 0 + 0.5 - 0.4 = 0.1
        δ_0 = 0 + 0.4 - 0.3 = 0.1

        GAE advantages (with λ=1.0 for simplicity):
        A_4 = 0.1
        A_3 = 0.2 + 0.1 = 0.3
        A_2 = 0.2 + 0.3 = 0.5
        A_1 = 0.1 + 0.5 = 0.6
        A_0 = 0.1 + 0.6 = 0.7

        Notice: Earlier tokens get higher advantages because they "set up" the
        good ending. This is credit assignment in action!
    """
    batch_size, response_len = values.shape
    device = values.device

    # Initialize advantage tensor
    advantages = torch.zeros_like(values)

    # =========================================================================
    # Step 1: Compute TD errors
    # =========================================================================
    # δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
    #
    # For the last token: V(s_{T}) = 0 (episode ends, no future value)
    # So δ_{T-1} = r_{T-1} + γ·0 - V(s_{T-1}) = r_{T-1} - V(s_{T-1})

    # Create V(s_{t+1}) by shifting values left and padding with 0
    # values:      [V_0, V_1, V_2, V_3, V_4]
    # next_values: [V_1, V_2, V_3, V_4, 0  ]  (0 = terminal state value)
    next_values = torch.zeros_like(values)
    next_values[:, :-1] = values[:, 1:]
    # Last position: next_values[:, -1] = 0 (already initialized)

    # TD errors: δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
    td_errors = rewards + gamma * next_values - values

    # =========================================================================
    # Step 2: Compute GAE (backward pass)
    # =========================================================================
    # A_t = δ_t + (γλ)·A_{t+1}
    # Start from the end and work backwards

    # Last position: A_{T-1} = δ_{T-1}
    advantages[:, -1] = td_errors[:, -1]

    # Work backwards: t = T-2, T-3, ..., 0
    for t in range(response_len - 2, -1, -1):
        # A_t = δ_t + (γλ)·A_{t+1}
        advantages[:, t] = td_errors[:, t] + gamma * lam * advantages[:, t + 1]

    # =========================================================================
    # Step 3: Compute returns for value function training
    # =========================================================================
    # The value function should predict V(s_t) = E[returns from t onwards]
    # returns_t = A_t + V(s_t)  (advantage + baseline = actual return)
    # This is the target for our value function loss
    returns = advantages + values

    return advantages, returns


def make_per_token_rewards(
    rewards: torch.Tensor,
    response_length: int,
) -> torch.Tensor:
    """
    Convert scalar rewards to per-token rewards for GAE.

    In RLHF, we typically get a single reward for the entire response
    (e.g., from a reward model). But GAE expects per-token rewards.

    The standard approach:
    - Reward at all positions except the last = 0
    - Reward at the last position = the actual reward

    This is like a sparse reward signal: you only find out how well
    you did at the very end.

    Args:
        rewards: (batch,) - scalar reward per response
        response_length: int - number of tokens in each response

    Returns:
        per_token_rewards: (batch, response_length)
            - All zeros except the last position which has the reward

    Example:
        rewards = [0.8, -0.3]  # batch of 2
        response_length = 5
        output = [[0, 0, 0, 0, 0.8],
                  [0, 0, 0, 0, -0.3]]
    """
    batch_size = rewards.shape[0]
    device = rewards.device

    # Initialize all rewards to 0
    per_token_rewards = torch.zeros(batch_size, response_length, device=device)

    # Put the actual reward at the last position
    per_token_rewards[:, -1] = rewards

    return per_token_rewards


def compute_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor = None,
    clip_value: float = None,
) -> torch.Tensor:
    """
    Compute value function loss (MSE between predictions and targets).

    The value head should learn to predict V(s) = expected future reward.
    We train it by minimizing the error between its predictions and
    the actual returns (computed via GAE).

    Args:
        values: (batch, seq_len) - current value predictions V(s_t)
        returns: (batch, seq_len) - target values (from GAE: A_t + V_old(s_t))
        old_values: (batch, seq_len) - value predictions from before PPO update
                    (optional, used for clipping)
        clip_value: float - clip value predictions to old_values ± clip_value
                    (optional, helps stability like PPO's policy clipping)

    Returns:
        value_loss: scalar - MSE loss to minimize

    Why clip values?
    ================
    Similar to PPO's policy clipping, we can clip value predictions to prevent
    the value function from changing too much in a single update.

    Without clipping:
        value_loss = (values - returns)²

    With clipping:
        clipped_values = clip(values, old_values - ε, old_values + ε)
        value_loss = max((values - returns)², (clipped_values - returns)²)

    This is optional but can improve stability.
    """
    if clip_value is not None and old_values is not None:
        # Clipped value loss (like PPO)
        # Clip current values to be within [old - ε, old + ε]
        clipped_values = torch.clamp(
            values,
            old_values - clip_value,
            old_values + clip_value
        )

        # Take the worse (larger) of the two losses
        # This prevents the value function from changing too much
        value_loss_unclipped = (values - returns) ** 2
        value_loss_clipped = (clipped_values - returns) ** 2
        value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
    else:
        # Simple MSE loss
        value_loss = ((values - returns) ** 2).mean()

    return value_loss


class PolicyWithValueHead(nn.Module):
    """
    Wrapper that combines a GPT model with a value head.

    Architecture:
    =============
                        GPT Transformer Blocks
                                │
                    hidden_states (batch, seq, n_embd)
                           ┌────┴────┐
                           │         │
                      lm_head    value_head
                           │         │
                    logits (vocab)  values (scalar)

    Why a wrapper?
    - Keeps the original GPT model unchanged
    - Clean separation: policy (lm_head) vs value (value_head)
    - Easy to freeze/unfreeze different parts
    - All RLHF-specific code stays in rlhf/ directory

    Usage:
        model = PolicyWithValueHead(gpt_model, n_embd=768)
        logits, values = model(input_ids)
        # logits: (batch, seq, vocab_size) - for policy/generation
        # values: (batch, seq) - for GAE advantage estimation
    """

    def __init__(self, gpt_model: nn.Module, n_embd: int):
        """
        Args:
            gpt_model: A nanoGPT model (has transformer, lm_head)
            n_embd: Hidden dimension (768 for GPT-2 small)
        """
        super().__init__()
        self.gpt = gpt_model
        self.value_head = ValueHead(n_embd)

        # Store config for convenience
        self.config = gpt_model.config

    def get_hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Run the transformer and return hidden states (before lm_head).

        This replicates the first part of GPT.forward() but returns
        the hidden states instead of logits.

        Args:
            idx: (batch, seq_len) - input token ids

        Returns:
            hidden_states: (batch, seq_len, n_embd)
        """
        device = idx.device
        b, t = idx.size()

        # Check sequence length
        assert t <= self.gpt.config.block_size, \
            f"Sequence length {t} > block_size {self.gpt.config.block_size}"

        # Position indices
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Embeddings
        tok_emb = self.gpt.transformer.wte(idx)  # (b, t, n_embd)
        pos_emb = self.gpt.transformer.wpe(pos)  # (t, n_embd)
        x = self.gpt.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.gpt.transformer.h:
            x = block(x)

        # Final layer norm
        hidden_states = self.gpt.transformer.ln_f(x)  # (b, t, n_embd)

        return hidden_states

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning logits, values, and optionally loss.

        Args:
            idx: (batch, seq_len) - input token ids
            targets: (batch, seq_len) - target token ids (optional, for loss)

        Returns:
            logits: (batch, seq_len, vocab_size) - next token predictions
            values: (batch, seq_len) - value estimates per position
            loss: scalar or None - cross-entropy loss if targets provided
        """
        # Get hidden states from transformer
        hidden_states = self.get_hidden_states(idx)  # (batch, seq, n_embd)

        # Policy head: hidden -> logits
        logits = self.gpt.lm_head(hidden_states)  # (batch, seq, vocab_size)

        # Value head: hidden -> values
        values = self.value_head(hidden_states)  # (batch, seq)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            import torch.nn.functional as F
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, values, loss

    def generate_with_values(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate tokens while collecting log probs AND value estimates.

        This extends the basic generate() to also return values,
        which we need for GAE computation.

        Args:
            prompt_ids: (batch, prompt_len) - tokenized prompts
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature

        Returns:
            generated_ids: (batch, prompt_len + max_new_tokens) - full sequences
            log_probs: (batch, max_new_tokens) - log P(token) for each generated token
            values: (batch, max_new_tokens) - V(s_t) for each generated position
        """
        import torch.nn.functional as F

        self.eval()
        device = prompt_ids.device

        generated_ids = prompt_ids.clone()
        log_probs_list = []
        values_list = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get hidden states for current sequence
                hidden_states = self.get_hidden_states(generated_ids)

                # Get logits and values for the LAST position only
                last_hidden = hidden_states[:, -1, :]  # (batch, n_embd)
                next_token_logits = self.gpt.lm_head(last_hidden)  # (batch, vocab)
                value = self.value_head.linear(last_hidden).squeeze(-1)  # (batch,)

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

                # Record log prob of sampled token
                log_prob = torch.log(probs.gather(dim=-1, index=next_token))
                log_probs_list.append(log_prob.squeeze(-1))  # (batch,)

                # Record value estimate
                values_list.append(value)  # (batch,)

                # Append token to sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

        # Stack results
        log_probs = torch.stack(log_probs_list, dim=1)  # (batch, max_new_tokens)
        values = torch.stack(values_list, dim=1)  # (batch, max_new_tokens)

        return generated_ids, log_probs, values


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
