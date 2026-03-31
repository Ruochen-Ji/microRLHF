# MicroRLHF: A Minimal RLHF Implementation for Beginners

Building on [nanoGPT](https://github.com/karpathy/nanoGPT), this project implements the post-training pipeline — LoRA, Reward Modeling, and PPO — with minimal, readable code designed for learning.

## Why This Project?

I personally had lots of fun learning from Andrej's nanoGPT project and it was truly a gem. At the end of the video of nanoGPT, Andrej pulled out OpenAI's [article](https://openai.com/index/chatgpt/) that demonstrates how OpenAI trained the model to have assistant-like behavior.

This repo is a continuation of nanoGPT that tries to replicate what OpenAI does with minimum hardware requirement (you'll still need a GPU to run this).

MicroRLHF follows nanoGPT's ethos: **minimal code, maximum insight**. Every component is implemented from scratch with clear explanations.

## What You'll Learn

```
Pretrained LLM → LoRA Finetuning → Reward Modeling → PPO → Aligned Model
├── How LoRA enables parameter-efficient finetuning
├── How reward models learn to predict human judgment
├── Why PPO needs a KL penalty (and what happens without it)
├── How GAE enables per-token credit assignment
└── The failure modes that make alignment hard (reward hacking, EOS spamming)
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8GB | 24GB+ |
| Model Size | GPT-2 Small (124M) | GPT-2 Medium (355M) |

**Memory Budget for PPO (GPT-2 Small):**
```
Policy (trainable):     ~2GB
Reference (frozen):     ~0.5GB
Reward Model:           ~0.5GB
Value Head:             ~0.1GB
Activations:            ~1-2GB
────────────────────────────────
Total:                  ~5-6GB
```

---

## What's Implemented

### LoRA (Low-Rank Adaptation)
Parameter-efficient finetuning that decomposes weight updates into low-rank matrices. Integrated into the nanoGPT training and inference loop.

```python
# Full finetuning: 768 x 768 = 589,824 parameters
# LoRA (r=8):  768 x 8 + 8 x 768 = 12,288 parameters (~48x fewer!)
W_new = W + B @ A   # B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)
```

See `lora.py` for the full implementation with merge support.

### Reward Modeling
Trains a scalar reward head on top of GPT-2 using the Bradley-Terry model and the [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) preference dataset.

```python
# Core insight: train reward(chosen) > reward(rejected)
loss = -log(sigmoid(reward_chosen - reward_rejected))
```

Train it with:
```bash
python -m rlhf.train_reward_model
```

### PPO with KL Penalty
The full PPO loop: generate responses, score them, compute advantages, update the policy. Includes a frozen reference model to prevent reward hacking via KL penalty.

```
┌─────────────────────────────────────────────────────────────┐
│                    PPO Architecture                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Policy (θ)      →  Generates responses                     │
│  Reference (θ₀)  →  Frozen; computes KL penalty             │
│  Reward (φ)      →  Scores responses                        │
│  Value (ψ)       →  Estimates expected reward (baseline)    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
# The core RLHF objective
reward_total = reward_model(response) - beta * KL(policy || reference)
#              ↑                              ↑
#              Be helpful                     Don't drift from base model
```

Two training scripts are provided:
```bash
python -m rlhf.train_ppo       # Basic PPO with uniform advantages
python -m rlhf.train_ppo_gae   # PPO with GAE for per-token credit assignment
```

### Naive Reward Functions (for Ablations)
Heuristic reward functions that demonstrate RLHF failure modes:

| Reward | What it rewards | Failure mode |
|--------|----------------|--------------|
| `LengthReward` | Moderate length (50-80 tokens) | Verbosity hacking |
| `BrevityReward` | Short responses | EOS spamming |
| `TargetLengthReward` | Specific length range | Padding/truncation |

These are useful for experimenting with PPO before training a full reward model.

---

## Project Structure

```
microRLHF/
├── model.py                    # GPT-2 architecture (from nanoGPT)
├── train.py                    # Pretraining script (+ LoRA support)
├── sample.py                   # Inference/generation (+ LoRA support)
├── lora.py                     # LoRA implementation
├── configurator.py             # Config file parsing
│
├── rlhf/                       # RLHF implementation
│   ├── reward_model.py         # RewardModel + RewardTrainer (Bradley-Terry)
│   ├── train_reward_model.py   # Train reward model on HH-RLHF data
│   ├── ppo.py                  # ValueHead, GAE, PolicyWithValueHead
│   ├── rl_utils.py             # PPO utilities (generate, KL, policy loss)
│   ├── train_ppo.py            # PPO training loop (basic)
│   ├── train_ppo_gae.py        # PPO training loop (with GAE)
│   ├── naive_reward.py         # Heuristic rewards for ablations
│   ├── data.py                 # PreferenceDataset (Anthropic HH-RLHF)
│   ├── plots/                  # Training visualizations
│   └── analysis/               # Analysis scripts and notes
│
├── config/                     # nanoGPT training configs
│   ├── train_gpt2.py
│   ├── train_shakespeare_char.py
│   ├── finetune_shakespeare.py
│   ├── finetune_shakespeare_lora.py
│   └── finetune_alpaca_lora.py
│
├── configs/                    # RLHF hyperparameter configs
│   ├── reward_config.yaml
│   └── ppo_config.yaml
│
├── data/                       # Dataset preparation
│   ├── shakespeare/
│   ├── shakespeare_char/
│   └── openwebtext/
│
└── chat/                       # Chat interface
    └── gradio_app.py
```

---

## Key Equations

### Bradley-Terry Model (Reward Training)
$$P(\text{chosen} \succ \text{rejected}) = \sigma(r(\text{chosen}) - r(\text{rejected}))$$

### PPO Objective
$$\mathcal{L}_{\text{PPO}} = \mathbb{E}\left[\min\left(\frac{\pi_\theta}{\pi_{\text{old}}} A, \text{clip}\left(\frac{\pi_\theta}{\pi_{\text{old}}}, 1-\epsilon, 1+\epsilon\right) A\right)\right]$$

### RLHF Reward with KL Penalty
$$R_{\text{total}} = R_{\phi}(y|x) - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

### DPO Loss (for reference)
$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

---

## References

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (DPO)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (PPO)
- [nanoGPT](https://github.com/karpathy/nanoGPT) (Karpathy)
- [The N Implementation Details of RLHF with PPO](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo)

---

## License

MIT
