# GAE (Generalized Advantage Estimation) Learning Notes

This document walks through the concepts and implementation of GAE for RLHF training.

---

## Part 1: The Problem GAE Solves

### Current Approach: "Whole Response = One Action"

```
Prompt: "What is 2+2?"
Response: "The answer is four." (5 tokens)
Reward: +0.8

Current: ALL tokens get advantage = 0.8
  "The"    → advantage = 0.8
  "answer" → advantage = 0.8
  "is"     → advantage = 0.8
  "four"   → advantage = 0.8
  "."      → advantage = 0.8
```

**Problem**: What if only the last token ("four") was actually good, and the rest was filler? We're giving equal credit to all tokens — this is the **credit assignment problem**.

### What We Want: Per-Token Credit

```
  "The"    → advantage = 0.1  (filler, low credit)
  "answer" → advantage = 0.2  (okay)
  "is"     → advantage = 0.2  (okay)
  "four"   → advantage = 0.9  (this is what earned the reward!)
  "."      → advantage = 0.5  (good to end properly)
```

Per-token advantages help training because:
- Tokens that directly contribute to reward get higher credit
- Filler tokens get less credit, so we don't reinforce them as strongly
- This leads to faster, more stable learning

---

## Part 2: The Value Function

To assign per-token credit, we need to know: **"How good is it to be at this position?"**

This is the **Value Function** `V(s_t)`:
- Input: The state at position t (all tokens so far)
- Output: Expected future reward from this point

```
Prompt: "What is 2+2?" → Response: "The answer is four."

Position 0 (after "The"):     V(s_0) = 0.3  "We're on track"
Position 1 (after "answer"):  V(s_1) = 0.4  "Still good"
Position 2 (after "is"):      V(s_2) = 0.5  "Getting closer"
Position 3 (after "four"):    V(s_3) = 0.9  "Nailed it!"
Position 4 (after "."):       V(s_4) = 0.8  "Done, reward coming"
```

### Implementation: Value Head

We add a small network on top of GPT's hidden states:

```python
# GPT outputs hidden states of shape (batch, seq_len, n_embd)
# Value head predicts a scalar for each position

class ValueHead(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear = nn.Linear(n_embd, 1)

    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, n_embd)
        # output: (batch, seq_len)
        return self.linear(hidden_states).squeeze(-1)
```

**Why a separate value head?**
- Policy outputs probability over actions (next token)
- Value outputs expected cumulative reward (a scalar)
- These are fundamentally different objectives, so separate heads learn better

---

## Part 3: TD Error (Temporal Difference)

Once we have `V(s_t)`, we can compute how "surprising" each transition was:

```
TD Error: δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
```

Where:
- `r_t` = reward at step t (usually 0, except at the last token)
- `γ` (gamma) = discount factor (e.g., 0.99)
- `V(s_{t+1})` = value of next state, or the expected total future reward of step t+1
- `V(s_t)` = value of current state, or the expected total future reward of step t

**Intuition**:
- If `δ_t > 0`: Things went **better** than expected → increase this action's probability
- If `δ_t < 0`: Things went **worse** than expected → decrease this action's probability

### Example

```
Position 2 → 3: Generating "four"
  r_2 = 0 (no reward yet)
  V(s_3) = 0.9 (after "four", looking good!)
  V(s_2) = 0.5 (before "four")

  δ_2 = 0 + 0.99 × 0.9 - 0.5 = 0.391

  Interpretation: "four" was a GREAT token choice (+0.39 better than expected)
```

---

## Part 4: GAE Formula

TD error gives us one-step credit. **GAE** looks at multiple future steps:

```
A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
```

Where `λ` (lambda) controls the tradeoff:
- `λ = 0`: Only use immediate TD error (high bias, low variance)
- `λ = 1`: Use all future TD errors (low bias, high variance), the math formula becomes A_t = r_t_n(the final reward) - V(s_t_0)
- `λ = 0.95`: Sweet spot (what most implementations use)

### Efficient Computation (Backward Pass)

Instead of computing the sum forward, we compute backward:

```python
# Start from the last token and work backwards
A_{T-1} = δ_{T-1}
A_{T-2} = δ_{T-2} + (γλ) × A_{T-1}
A_{T-3} = δ_{T-3} + (γλ) × A_{T-2}
...
```

This is just one line in code:
```python
advantages[t] = td_errors[t] + gamma * lam * advantages[t + 1]
```

---

## Part 5: Implementation Plan

Here's the step-by-step implementation plan:

1. **Modify model to output hidden states** (or use a wrapper)
2. **Create ValueHead class**
3. **Write `compute_advantages_gae()` function**
4. **Add value loss to training**
5. **Update training loop**

### Architecture Diagram

```
   GPT hidden states (batch, seq_len, 768)
           │
           ├──→ lm_head ──→ logits (batch, seq_len, vocab_size)  [existing]
           │
           └──→ value_head ──→ values (batch, seq_len)           [new!]
```

---

## Implementation Code

All code is in `ppo.py`. Here's a summary of what we implemented:

### 1. ValueHead (simple linear projection)

```python
class ValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        # (batch, seq, hidden) -> (batch, seq)
        return self.linear(hidden_states).squeeze(-1)
```

### 2. compute_gae (the core algorithm)

```python
def compute_gae(values, rewards, gamma=0.99, lam=0.95):
    # Step 1: TD errors
    next_values = torch.zeros_like(values)
    next_values[:, :-1] = values[:, 1:]  # Shift left, pad with 0
    td_errors = rewards + gamma * next_values - values

    # Step 2: GAE (backward pass)
    advantages = torch.zeros_like(values)
    advantages[:, -1] = td_errors[:, -1]
    for t in range(response_len - 2, -1, -1):
        advantages[:, t] = td_errors[:, t] + gamma * lam * advantages[:, t + 1]

    # Step 3: Returns for value loss
    returns = advantages + values
    return advantages, returns
```

### 3. PolicyWithValueHead (wrapper class)

```python
class PolicyWithValueHead(nn.Module):
    def __init__(self, gpt_model, n_embd):
        self.gpt = gpt_model
        self.value_head = ValueHead(n_embd)

    def get_hidden_states(self, idx):
        # Run transformer blocks, return hidden states before lm_head
        ...

    def forward(self, idx, targets=None):
        hidden_states = self.get_hidden_states(idx)
        logits = self.gpt.lm_head(hidden_states)  # Policy
        values = self.value_head(hidden_states)   # Value
        return logits, values, loss

    def generate_with_values(self, prompt_ids, max_new_tokens, temperature):
        # Generate + collect log_probs + collect values
        return generated_ids, log_probs, values
```

---

## Training Loop Integration

Here's how to integrate GAE into the training loop. The key changes from the baseline:

### Before (Uniform Advantages)

```python
# Generate
generated_ids, old_log_probs = generate(model, prompt_ids, max_new_tokens)

# Compute scalar rewards
rewards = reward_fn.compute(generated_ids, ...)  # (batch,)

# Uniform advantage: all tokens get the same credit
advantages = rewards - baseline  # (batch,)

# Loss
policy_loss, stats = compute_policy_loss(model, generated_ids, old_log_probs, advantages, ...)
loss = policy_loss + kl_coef * kl
```

### After (GAE Per-Token Advantages)

```python
from ppo import PolicyWithValueHead, compute_gae, compute_value_loss, make_per_token_rewards

# Wrap model with value head
policy_model = PolicyWithValueHead(gpt_model, n_embd=768)

# Generate with values
generated_ids, old_log_probs, old_values = policy_model.generate_with_values(
    prompt_ids, max_new_tokens, temperature
)

# Compute scalar rewards
scalar_rewards = reward_fn.compute(generated_ids, ...)  # (batch,)

# Convert to per-token rewards
per_token_rewards = make_per_token_rewards(scalar_rewards, response_length=max_new_tokens)

# Compute GAE advantages (per-token!)
advantages, returns = compute_gae(
    values=old_values,           # (batch, response_len)
    rewards=per_token_rewards,   # (batch, response_len)
    gamma=0.99,
    lam=0.95
)
# advantages is now (batch, response_len) - different credit per token!

# Normalize advantages (helps stability)
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# PPO updates
for epoch in range(ppo_epochs):
    # Policy loss (now uses per-token advantages)
    policy_loss, stats = compute_policy_loss(
        model, generated_ids, old_log_probs, advantages, prompt_length
    )

    # Value loss (train value head to predict returns)
    # Need to re-compute values with current model
    _, current_values, _ = policy_model(generated_ids[:, :-1], targets=generated_ids[:, 1:])
    response_values = current_values[:, prompt_length-1:]  # Only response positions
    value_loss = compute_value_loss(response_values, returns, old_values, clip_value=0.2)

    # KL penalty
    kl = compute_kl_penalty(policy_model.gpt, ref_model, generated_ids, prompt_length)

    # Total loss
    loss = policy_loss + value_coef * value_loss + kl_coef * kl

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### New Hyperparameters

```python
gamma = 0.99      # Discount factor (how much to care about future rewards)
lam = 0.95        # GAE lambda (bias-variance tradeoff)
value_coef = 0.5  # Weight for value loss (typically 0.5)
```

---

## Summary

| Component | Purpose | Location |
|-----------|---------|----------|
| `ValueHead` | Predicts V(s) for each position | ppo.py |
| `compute_gae()` | Computes per-token advantages | ppo.py |
| `compute_value_loss()` | MSE loss for training value head | ppo.py |
| `make_per_token_rewards()` | Converts scalar reward to per-token | ppo.py |
| `PolicyWithValueHead` | Wraps GPT + ValueHead | ppo.py |
| `compute_policy_loss()` | Updated to handle (batch, seq) advantages | rl_utils.py |
