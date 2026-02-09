# RLHF Training Scripts

This folder contains PPO training scripts for RLHF (Reinforcement Learning from Human Feedback).

## Scripts

### 1. `train_ppo.py` - Baseline PPO

Basic PPO implementation with **uniform credit assignment** (all tokens in a response get the same advantage).

```bash
cd /home/ruochen/projects/nanoGPT
python rlhf/train_ppo.py
```

### 2. `train_ppo_gae.py` - PPO with GAE

Advanced PPO with **Generalized Advantage Estimation** for per-token credit assignment.

```bash
cd /home/ruochen/projects/nanoGPT
python rlhf/train_ppo_gae.py
```

### 3. `train_reward_model.py` - Reward Model Training

Trains a reward model on [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) human preference data. Uses GPT-2 as backbone with a scalar reward head, trained with Bradley-Terry pairwise ranking loss.

```bash
cd /home/ruochen/projects/nanoGPT
python -m rlhf.train_reward_model
```

Saves checkpoint to `rlhf/reward_model.pt`. Takes ~1 hour for 1 epoch on a single GPU.

## Key Differences

| Aspect | train_ppo.py (baseline) | train_ppo_gae.py (GAE) |
|--------|------------------------|------------------------|
| Model | `GPT` | `PolicyWithValueHead(GPT)` |
| Generation | `generate()` | `generate_with_values()` |
| Advantages | `rewards - baseline` (uniform) | `compute_gae()` (per-token) |
| Losses | `policy_loss + kl` | `policy_loss + value_loss + kl` |
| Trainable params | LoRA only | LoRA + ValueHead |

## GAE Hyperparameters

```python
gamma = 0.99      # Discount factor: how much to value future rewards
lam = 0.95        # GAE lambda: bias-variance tradeoff (0.95 is standard)
value_coef = 0.5  # Weight for value loss
```

## Prerequisites

Both scripts require an SFT checkpoint at `out-alpaca-lora/ckpt.pt`.

## Learning Resources

See `learning_notes.md` for a detailed walkthrough of GAE concepts and implementation.
