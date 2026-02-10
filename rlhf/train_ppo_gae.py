"""
train_ppo_gae.py - PPO training with Generalized Advantage Estimation (GAE)

This script extends train_ppo.py with per-token credit assignment using GAE.

Key differences from train_ppo.py:
=================================
1. Uses PolicyWithValueHead wrapper (adds value head to GPT)
2. generate_with_values() returns values alongside log_probs
3. compute_gae() gives per-token advantages instead of uniform
4. Adds value loss to train the value head
5. Per-token advantages = better credit assignment = faster learning

Architecture:
                    GPT Transformer
                          │
              hidden_states (batch, seq, 768)
                     ┌────┴────┐
                     │         │
                lm_head    value_head
                     │         │
              logits (vocab)  values (scalar)
                     │         │
               Policy Loss   Value Loss
"""
import os
import sys
import csv
import random
import torch
import tiktoken
from datasets import load_dataset

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig
from lora import apply_lora_to_model
from rl_utils import compute_kl_penalty, compute_policy_loss, check_training_health
from naive_reward import BrevityReward
from reward_model import RewardModel, TrainedRewardModel
from ppo import (
    PolicyWithValueHead,
    compute_gae,
    compute_value_loss,
    make_per_token_rewards,
)


def print_header(text, char="=", width=60):
    """Print a formatted section header."""
    print("\n" + char * width)
    print(text)
    print(char * width)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# SFT checkpoint path
sft_checkpoint_path = "out-alpaca-lora/ckpt.pt"

# Generation config
max_new_tokens = 200
temperature = 1.0

# GPT-2 special tokens
eos_token_id = 50256

# PPO Training config
num_steps = 200
batch_size = 8
ppo_epochs = 2
learning_rate = 1e-5
clip_epsilon = 0.2
max_grad_norm = 1.0

# KL penalty
kl_coef = 0.05

# GAE hyperparameters (NEW!)
gamma = 0.99      # Discount factor: how much to value future rewards
lam = 0.95        # GAE lambda: bias-variance tradeoff (0.95 is standard)
value_coef = 0.5  # Weight for value loss (0.5 is standard)

# Logging config
log_interval = 10
sample_interval = 50

# Reward function — trained reward model from Anthropic HH-RLHF
# To use a naive reward instead, uncomment:
#   reward_fn = BrevityReward(max_good_length=30, eos_bonus=0.5)
print("\nLoading trained reward model...")
reward_gpt = GPT.from_pretrained("gpt2")
reward_model = RewardModel(reward_gpt).to(device)
reward_model.load_state_dict(torch.load("rlhf/logs/reward_model.pt", map_location=device))
reward_model.eval()
for param in reward_model.parameters():
    param.requires_grad = False
del reward_gpt
reward_fn = TrainedRewardModel(reward_model)
print(f"Reward function: {reward_fn.__class__.__name__} (trained on Anthropic HH-RLHF)")

# -----------------------------------------------------------------------------
# Step 1: Load SFT'd GPT-2 with LoRA
# -----------------------------------------------------------------------------
print_header("Step 1: Loading Models")

print(f"Loading SFT checkpoint from {sft_checkpoint_path}...")
checkpoint = torch.load(sft_checkpoint_path, map_location=device)
lora_config = checkpoint['lora_config']
print(f"LoRA config: rank={lora_config['rank']}, alpha={lora_config['alpha']}")

# Create base GPT model
gptconf = GPTConfig(**checkpoint['model_args'])
base_gpt = GPT(gptconf)

# Apply LoRA
base_gpt = apply_lora_to_model(
    base_gpt,
    rank=lora_config['rank'],
    alpha=lora_config['alpha'],
    dropout=0.0,
    target_modules=lora_config['target_modules']
)

# Load weights
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
base_gpt.load_state_dict(state_dict)

# -----------------------------------------------------------------------------
# Step 2: Wrap with PolicyWithValueHead (NEW!)
# -----------------------------------------------------------------------------
print("\nWrapping model with ValueHead...")
policy_model = PolicyWithValueHead(base_gpt, n_embd=gptconf.n_embd)
policy_model.to(device)
policy_model.eval()

# Count parameters
total_params = sum(p.numel() for p in policy_model.parameters())
trainable_params = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
value_head_params = sum(p.numel() for p in policy_model.value_head.parameters())
print(f"Policy model: {total_params:,} total, {trainable_params:,} trainable (LoRA + ValueHead)")
print(f"Value head params: {value_head_params:,}")

# -----------------------------------------------------------------------------
# Step 3: Create frozen reference model
# -----------------------------------------------------------------------------
print("\nCreating frozen reference model...")
ref_model = GPT(gptconf)
ref_model = apply_lora_to_model(
    ref_model,
    rank=lora_config['rank'],
    alpha=lora_config['alpha'],
    dropout=0.0,
    target_modules=lora_config['target_modules']
)

# Reload state dict
state_dict = checkpoint['model']
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
ref_model.load_state_dict(state_dict)

for param in ref_model.parameters():
    param.requires_grad = False
ref_model.to(device)
ref_model.eval()
print("Reference model created and frozen")

# -----------------------------------------------------------------------------
# Step 4: Load tokenizer and prompts
# -----------------------------------------------------------------------------
print("\nLoading tokenizer...")
tokenizer = tiktoken.get_encoding("gpt2")


def format_prompt(instruction, input_text=""):
    """Format a prompt in Alpaca style."""
    if input_text:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


# Test prompts for monitoring
test_prompts = [
    format_prompt("Why is sea water salty?"),
    format_prompt("What is the capital of France?"),
    format_prompt("Explain what machine learning is in one sentence."),
]

# Load Alpaca training prompts
print("Loading Alpaca dataset...")
alpaca_dataset = load_dataset("tatsu-lab/alpaca", split="train")
training_prompts = []
for example in alpaca_dataset:
    instruction = example['instruction']
    input_text = example.get('input', '')
    if input_text:
        training_prompts.append(format_prompt(f"{instruction}\n\nInput: {input_text}"))
    else:
        training_prompts.append(format_prompt(instruction))

random.seed(42)
random.shuffle(training_prompts)
print(f"Training prompts: {len(training_prompts):,}")

# -----------------------------------------------------------------------------
# Step 5: Set up optimizer (includes value head!)
# -----------------------------------------------------------------------------
print_header("Step 2: Setting up Training")

# Optimizer now includes BOTH LoRA params AND value head params
trainable_params_list = [p for p in policy_model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params_list, lr=learning_rate)
print(f"Optimizer: AdamW, lr={learning_rate}")
print(f"Trainable parameters: {sum(p.numel() for p in trainable_params_list):,}")

# Metrics tracking
metrics_history = {
    "step": [],
    "reward_mean": [],
    "reward_std": [],
    "kl": [],
    "policy_loss": [],
    "value_loss": [],  # NEW!
    "response_length": [],
    "ratio_mean": [],
    "clip_fraction": [],
    "advantage_mean": [],  # NEW!
    "advantage_std": [],   # NEW!
}

# CSV logging for plotting later
log_path = "rlhf/logs/ppo_gae_log.csv"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
log_file = open(log_path, "w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow([
    "step", "reward_mean", "reward_std", "kl", "policy_loss",
    "value_loss", "response_length", "ratio_mean", "clip_fraction",
    "advantage_mean", "advantage_std"
])
print(f"Logging metrics to {log_path}")

print(f"""
Training config:
  - Steps: {num_steps}
  - Batch size: {batch_size}
  - PPO epochs: {ppo_epochs}
  - KL coefficient: {kl_coef}
  - Clip epsilon: {clip_epsilon}
  - GAE gamma: {gamma}
  - GAE lambda: {lam}
  - Value coef: {value_coef}
""")

# -----------------------------------------------------------------------------
# Step 6: Training Loop with GAE
# -----------------------------------------------------------------------------
print_header("Step 3: Starting PPO Training with GAE")

for step in range(num_steps):
    # =========================================================================
    # 6.1: Sample batch of prompts
    # =========================================================================
    start_idx = (step * batch_size) % len(training_prompts)
    batch_prompts = training_prompts[start_idx:start_idx + batch_size]
    if len(batch_prompts) < batch_size:
        batch_prompts = batch_prompts + training_prompts[:batch_size - len(batch_prompts)]

    # Tokenize and pad
    prompt_tokens = [tokenizer.encode(p) for p in batch_prompts]
    max_prompt_len = max(len(t) for t in prompt_tokens)
    padded_prompts = []
    for tokens in prompt_tokens:
        padding = [eos_token_id] * (max_prompt_len - len(tokens))
        padded_prompts.append(padding + tokens)
    prompt_ids = torch.tensor(padded_prompts, device=device)

    # =========================================================================
    # 6.2: Generate responses WITH values (NEW!)
    # =========================================================================
    policy_model.eval()
    with torch.no_grad():
        generated_ids, old_log_probs, old_values = policy_model.generate_with_values(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

    # =========================================================================
    # 6.3: Compute rewards
    # =========================================================================
    scalar_rewards = reward_fn.compute(
        generated_ids, max_prompt_len, max_new_tokens, eos_token_id
    )

    # Compute response lengths for logging
    response_ids = generated_ids[:, max_prompt_len:]
    response_lengths = []
    for i in range(batch_size):
        eos_positions = (response_ids[i] == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            response_lengths.append(eos_positions[0].item())
        else:
            response_lengths.append(max_new_tokens)
    avg_response_length = sum(response_lengths) / len(response_lengths)

    # =========================================================================
    # 6.4: Compute GAE advantages (NEW! - replaces simple reward - baseline)
    # =========================================================================
    # Convert scalar rewards to per-token rewards
    per_token_rewards = make_per_token_rewards(scalar_rewards, max_new_tokens)

    # Compute GAE
    advantages, returns = compute_gae(
        values=old_values,
        rewards=per_token_rewards,
        gamma=gamma,
        lam=lam
    )

    # Normalize advantages (important for stability!)
    adv_mean = advantages.mean()
    adv_std = advantages.std()
    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    # =========================================================================
    # 6.5: PPO updates (with value loss!)
    # =========================================================================
    policy_model.train()

    for epoch in range(ppo_epochs):
        # ----- Policy Loss -----
        policy_loss, stats = compute_policy_loss(
            policy_model.gpt,  # Pass the underlying GPT model
            generated_ids,
            old_log_probs,
            advantages,  # Now (batch, response_len) instead of (batch,)!
            max_prompt_len,
            clip_epsilon=clip_epsilon
        )

        # ----- Value Loss (NEW!) -----
        # Get current value predictions
        hidden_states = policy_model.get_hidden_states(generated_ids[:, :-1])
        current_values = policy_model.value_head(hidden_states)
        # Keep only response positions (predicting values for response tokens)
        response_values = current_values[:, max_prompt_len - 1:]

        # Compute value loss
        value_loss = compute_value_loss(
            values=response_values,
            returns=returns.detach(),  # Detach to stop gradients from flowing through returns
            old_values=old_values,
            clip_value=0.2  # Optional: clip value predictions like PPO clips policy
        )

        # ----- KL Penalty -----
        kl = compute_kl_penalty(
            policy_model.gpt,
            ref_model,
            generated_ids,
            max_prompt_len
        )

        # ----- Total Loss -----
        loss = policy_loss + value_coef * value_loss + kl_coef * kl

        # ----- Backward Pass -----
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params_list, max_grad_norm)
        optimizer.step()

    # =========================================================================
    # 6.6: Logging
    # =========================================================================
    check_training_health(stats, step)

    # Store metrics
    metrics_history["step"].append(step)
    metrics_history["reward_mean"].append(scalar_rewards.mean().item())
    metrics_history["reward_std"].append(scalar_rewards.std().item())
    metrics_history["kl"].append(kl.item())
    metrics_history["policy_loss"].append(policy_loss.item())
    metrics_history["value_loss"].append(value_loss.item())
    metrics_history["response_length"].append(avg_response_length)
    metrics_history["ratio_mean"].append(stats["ratio_mean"])
    metrics_history["clip_fraction"].append(stats["clip_fraction"])
    metrics_history["advantage_mean"].append(adv_mean.item())
    metrics_history["advantage_std"].append(adv_std.item())

    # Write to CSV
    log_writer.writerow([
        step,
        f"{scalar_rewards.mean().item():.4f}",
        f"{scalar_rewards.std().item():.4f}",
        f"{kl.item():.4f}",
        f"{policy_loss.item():.4f}",
        f"{value_loss.item():.4f}",
        f"{avg_response_length:.1f}",
        f"{stats['ratio_mean']:.4f}",
        f"{stats['clip_fraction']:.4f}",
        f"{adv_mean.item():.4f}",
        f"{adv_std.item():.4f}",
    ])
    log_file.flush()

    # Print progress
    if step % log_interval == 0:
        print(f"Step {step:4d} | "
              f"R: {scalar_rewards.mean().item():+.3f} | "
              f"Len: {avg_response_length:5.1f} | "
              f"KL: {kl.item():.4f} | "
              f"VLoss: {value_loss.item():.4f} | "
              f"Clip: {stats['clip_fraction']:.1%}")

    # Show sample generations
    if step % sample_interval == 0 and step > 0:
        print_header(f"Sample generation at step {step}:", char="-", width=40)
        response_text = tokenizer.decode(response_ids[0].tolist())
        if eos_token_id in response_ids[0].tolist():
            eos_idx = response_ids[0].tolist().index(eos_token_id)
            response_text = tokenizer.decode(response_ids[0, :eos_idx].tolist())

        instruction = batch_prompts[0].split("Instruction:")[1].split("###")[0].strip()
        print(f"Instruction: {instruction[:60]}...")
        print(f"Response: {response_text[:200]}...")
        print(f"Length: {response_lengths[0]} tokens, Reward: {scalar_rewards[0].item():.3f}")

        # Show advantage distribution for this sample (NEW!)
        print(f"Advantages (first 5 tokens): {[f'{a:.2f}' for a in advantages[0, :5].tolist()]}")
        print("-" * 40)

# -----------------------------------------------------------------------------
# Training Complete
# -----------------------------------------------------------------------------
log_file.close()
print(f"Metrics saved to {log_path}")

print_header("Training Complete!")

print(f"""
Training Summary:
- Total steps: {num_steps}
- Final reward: {metrics_history['reward_mean'][-1]:.3f}
- Final KL: {metrics_history['kl'][-1]:.4f}
- Final value loss: {metrics_history['value_loss'][-1]:.4f}
- Final response length: {metrics_history['response_length'][-1]:.1f}

Reward progression:
- Start: {metrics_history['reward_mean'][0]:.3f}
- End:   {metrics_history['reward_mean'][-1]:.3f}
- Change: {metrics_history['reward_mean'][-1] - metrics_history['reward_mean'][0]:+.3f}

Value Loss progression:
- Start: {metrics_history['value_loss'][0]:.4f}
- End:   {metrics_history['value_loss'][-1]:.4f}
""")

# Final sample generations
print_header("Final Sample Generations")

policy_model.eval()
with torch.no_grad():
    prompt_tokens = [tokenizer.encode(p) for p in test_prompts]
    max_prompt_len = max(len(t) for t in prompt_tokens)
    padded_prompts = []
    for tokens in prompt_tokens:
        padding = [eos_token_id] * (max_prompt_len - len(tokens))
        padded_prompts.append(padding + tokens)
    prompt_ids = torch.tensor(padded_prompts, device=device)

    final_generated, _, _ = policy_model.generate_with_values(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )

for i in range(len(test_prompts)):
    response_ids = final_generated[i, max_prompt_len:].tolist()
    eos_idx = len(response_ids)
    for j, tok in enumerate(response_ids):
        if tok == eos_token_id:
            eos_idx = j
            break

    response_text = tokenizer.decode(response_ids[:eos_idx])
    instruction = test_prompts[i].split("Instruction:")[1].split("###")[0].strip()

    print(f"\n--- Prompt {i+1} ---")
    print(f"Instruction: {instruction[:60]}...")
    print(f"Response ({eos_idx} tokens): {response_text}")
