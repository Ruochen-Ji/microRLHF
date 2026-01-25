# MicroRLHF: A Minimal RLHF Implementation for beginners

Building on [nanoGPT](https://github.com/karpathy/nanoGPT), this project implements the complete post-training pipeline—SFT, LoRA, Reward Modeling, PPO, and DPO—with minimal, readable code designed for learning.

## Why This Project?

I personally had lots of fun learning from Andrej's nanoGPT project and it was truly a gem. At the end of the video of nanoGPT, Andrej pulled out OpenAI's (article)[https://openai.com/index/chatgpt/] that demonstrates how OpenAI trained the model to have assistant-like behavior. 

This repo is a continuation of nanoGPT that tries to replicate what OpenAI does with minimum hardware requirement(you'll still need GPU to run this). 

NanoRLHF tries to follow nano-gpt's etho: **minimal code, maximum insight**. Every component is implemented from scratch with clear explanations.

## What we'll Learn

```
Pretrained LLM → SFT → RLHF → Aligned Model
├── How instruction-following is trained (SFT)
├── How to collect and use human preferences
├── How reward models learn to predict human judgment
├── Why PPO needs a KL penalty (and what happens without it)
├── How DPO eliminates the reward model entirely
└── The failure modes that make alignment hard
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

## Roadmap

### Phase 1: Infrastructure
> Goal: Minimal UI for training, finetuning, and inference

- [ ] Set up Gradio/Streamlit app with tab navigation
- [ ] **Inference Tab**: Chat interface with tokens/sec display
- [ ] **Training Tab**: Streaming loss curves, basic hyperparameter controls
- [ ] **Finetune Tab**: SFT and LoRA options
- [ ] Checkpoint save/load functionality

### Phase 2: Supervised Fine-Tuning (SFT)
> Goal: Turn base GPT-2 into an instruction-following model

- [ ] Prepare instruction dataset (Alpaca format: instruction → response)
- [ ] Implement SFT training loop
- [ ] Add LoRA as memory-efficient finetuning option
- [ ] Compare base vs. SFT model outputs in chat interface

**Key Concept**: SFT teaches the model the *format* of helpful responses, but not necessarily *what* humans prefer.

### Phase 3: Preference Data Collection
> Goal: Build a dataset of human preferences for training the reward model

- [ ] **Annotation UI**: Side-by-side response comparison
  - Generate 2 responses for same prompt
  - Human selects: A > B, B > A, or Tie
  - Export as preference dataset
- [ ] Implement preference data format:
  ```python
  {
      "prompt": "How do I make coffee?",
      "chosen": "Here's a step-by-step guide...",
      "rejected": "Coffee is a beverage..."
  }
  ```
- [ ] (Optional) Synthetic preferences using stronger model as judge

### Phase 4: Reward Modeling
> Goal: Train a model to predict human preferences

- [ ] Reward model architecture (GPT-2 + scalar output head)
- [ ] Implement Bradley-Terry loss:
  ```python
  # Core insight: train reward(chosen) > reward(rejected)
  loss = -log(sigmoid(reward_chosen - reward_rejected))
  ```
- [ ] Training pipeline with preference pairs
- [ ] Evaluation: accuracy on held-out preferences
- [ ] Visualization: reward distribution histograms

**Key Concept**: The reward model learns to *simulate* human judgment, enabling us to score millions of responses without human labelers.

### Phase 5: RLHF with PPO
> Goal: Use the reward model to improve the policy via reinforcement learning

#### 5.1 Architecture Setup
- [ ] Policy model (trainable GPT-2)
- [ ] Reference model (frozen copy of initial policy)
- [ ] Value head (for PPO advantage estimation)

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

#### 5.2 Training Loop
- [ ] Generate responses from current policy
- [ ] Score with reward model
- [ ] Compute KL penalty: `KL(policy || reference)`
- [ ] PPO update with advantage estimation
- [ ] Logging: reward, KL divergence, policy loss

```python
# The core RLHF objective
reward_total = reward_model(response) - beta * KL(policy || reference)
#              ↑                              ↑
#              Be helpful                     Don't drift from base model
```

#### 5.3 Ablations & Failure Modes
- [ ] **Experiment**: What happens without KL penalty?
  - Demonstrate reward hacking (high reward, garbage output)
- [ ] **Experiment**: What happens with beta too high/low?
- [ ] Visualize KL divergence over training

**Key Concept**: The KL penalty prevents the policy from finding adversarial responses that "hack" the reward model while producing gibberish.

### Phase 6: Direct Preference Optimization (DPO)
> Goal: Achieve RLHF results without reward model or PPO

- [ ] Implement DPO loss:
  ```python
  # The elegant insight: reward is implicit in policy ratios
  log_ratio_w = log_prob(policy, chosen) - log_prob(ref, chosen)
  log_ratio_l = log_prob(policy, rejected) - log_prob(ref, rejected)
  loss = -log(sigmoid(beta * (log_ratio_w - log_ratio_l)))
  ```
- [ ] Train on same preference data as reward model
- [ ] Compare DPO vs PPO:
  - Training stability
  - Final model quality
  - Compute requirements

**Key Concept**: DPO shows that the optimal RLHF policy has a closed form—we can skip reward modeling and RL entirely by training directly on preferences.

### Phase 7: Evaluation & Visualization
> Goal: Demonstrate what was learned and make it tangible

- [ ] **Chat Comparison**: Base → SFT → RLHF side-by-side
- [ ] **Training Curves Dashboard**:
  - Reward over time
  - KL divergence
  - Policy/value loss
- [ ] **Ablation Summary**: Table comparing all approaches
- [ ] **Failure Mode Gallery**: Examples of reward hacking, mode collapse

---

## Project Structure

```
nanoRLHF/
├── app/
│   ├── app.py              # Gradio/Streamlit entry point
│   ├── inference_tab.py    # Chat interface
│   ├── training_tab.py     # Training visualization
│   ├── finetune_tab.py     # SFT/LoRA controls
│   └── annotate_tab.py     # Preference collection UI
├── nanogpt/                # Base nanoGPT code (wrapper module)
│   └── __init__.py         # Exports GPT, GPTConfig from root
├── data/
│   ├── shakespeare/        # Character-level Shakespeare data
│   │   └── prepare.py
│   ├── shakespeare_char/   # Token-level Shakespeare data
│   │   └── prepare.py
│   ├── alpaca/             # Instruction-following dataset (SFT)
│   │   └── prepare.py
│   ├── openwebtext/        # Large-scale pretraining data
│   │   └── prepare.py
│   ├── preferences/        # Human preference data (RLHF)
│   │   ├── train.json      # Training preferences
│   │   └── val.json        # Validation preferences
│   └── prompts/            # Prompts for PPO generation
│       └── train.json
├── rlhf/
│   ├── sft.py              # Supervised fine-tuning
│   ├── lora.py             # LoRA implementation
│   ├── reward_model.py     # Reward model architecture & training
│   ├── ppo.py              # PPO trainer
│   ├── dpo.py              # DPO trainer
│   └── data.py             # Preference dataset handling
├── configs/
│   ├── sft_config.yaml
│   ├── reward_config.yaml
│   ├── ppo_config.yaml
│   └── dpo_config.yaml
├── scripts/
│   ├── train_sft.py        # SFT training script
│   ├── collect_preferences.py
│   ├── train_reward_model.py
│   ├── train_ppo.py
│   └── train_dpo.py
├── model.py                # Core GPT model (from nanoGPT)
├── train.py                # Base training script
├── sample.py               # Inference/generation script
├── lora.py                 # LoRA implementation
└── README.md
```

---

## Key Equations

### Bradley-Terry Model (Reward Training)
$$P(\text{chosen} \succ \text{rejected}) = \sigma(r(\text{chosen}) - r(\text{rejected}))$$

### PPO Objective
$$\mathcal{L}_{\text{PPO}} = \mathbb{E}\left[\min\left(\frac{\pi_\theta}{\pi_{\text{old}}} A, \text{clip}\left(\frac{\pi_\theta}{\pi_{\text{old}}}, 1-\epsilon, 1+\epsilon\right) A\right)\right]$$

### RLHF Reward with KL Penalty
$$R_{\text{total}} = R_{\phi}(y|x) - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

### DPO Loss
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