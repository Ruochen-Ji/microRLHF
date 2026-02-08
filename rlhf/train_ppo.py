"""
train_ppo.py - PPO training script for baby RLHF

Step 2: Environment Setup
- Load SFT'd GPT-2 (with LoRA)
- Create frozen reference model
- Load Alpaca prompts
- Test generation

Step 3: PPO Training Loop
- Sample prompts, generate responses
- Compute rewards and advantages
- PPO update with KL penalty
- Monitor for reward hacking
"""
import os
import sys
import random
import torch
import tiktoken
from datasets import load_dataset

# Add parent directory to path so we can import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT
from lora import apply_lora_to_model
from rl_utils import (
    generate,
    compute_kl_penalty,
    compute_policy_loss,
    check_training_health,
)
from naive_reward import BrevityReward  # Try different reward functions here!

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
max_new_tokens = 200# Shorter for faster iteration during RL
temperature = 1.0

# GPT-2 special tokens
eos_token_id = 50256  # <|endoftext|>

# PPO Training config
num_steps =  50          # Total training steps
batch_size = 4            # Prompts per step (small for memory)
ppo_epochs = 4            # PPO updates per batch
learning_rate = 3e-5      # 1e-5 → 3e-5 (faster learning)
kl_coef = 0.02            # 0.1 → 0.02 (looser leash, allow more drift)
clip_epsilon = 0.3        # 0.2 → 0.3 (wider clip range, bigger updates)
max_grad_norm = 1.0       # Gradient clipping

# Logging config
log_interval = 10         # Print metrics every N steps
sample_interval = 50      # Show sample generations every N steps

# Reward function - swap these to try different failure modes!
# Options: LengthReward(), BrevityReward()
reward_fn = BrevityReward(max_good_length=30, eos_bonus=0.5)
print(f"\nReward function: {reward_fn.__class__.__name__}")

# -----------------------------------------------------------------------------
# Step 2a: Load SFT'd GPT-2 with LoRA as the policy model
# -----------------------------------------------------------------------------
print(f"\nLoading SFT checkpoint from {sft_checkpoint_path}...")

# Load checkpoint
checkpoint = torch.load(sft_checkpoint_path, map_location=device)
lora_config = checkpoint['lora_config']
print(f"LoRA config: rank={lora_config['rank']}, alpha={lora_config['alpha']}")

# Create model from checkpoint config
from model import GPTConfig
gptconf = GPTConfig(**checkpoint['model_args'])
policy_model = GPT(gptconf)

# Apply LoRA structure
policy_model = apply_lora_to_model(
    policy_model,
    rank=lora_config['rank'],
    alpha=lora_config['alpha'],
    dropout=0.0,  # No dropout during inference/RL
    target_modules=lora_config['target_modules']
)

# Load state dict, stripping torch.compile() prefix
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
policy_model.load_state_dict(state_dict)
print("Loaded model weights from checkpoint")

policy_model.to(device)
policy_model.eval()

# Print block_size for debugging
print(f"Model block_size: {gptconf.block_size}")

# Count parameters
total_params = sum(p.numel() for p in policy_model.parameters())
trainable_params = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
print(f"Policy model loaded: {total_params:,} total params, {trainable_params:,} trainable (LoRA)")

# -----------------------------------------------------------------------------
# Step 2b: Create frozen reference model
# -----------------------------------------------------------------------------
print("\nCreating frozen reference model...")

# For reference model, we load a fresh copy (same process as policy)
# This ensures complete separation between the two models
ref_model = GPT(gptconf)
ref_model = apply_lora_to_model(
    ref_model,
    rank=lora_config['rank'],
    alpha=lora_config['alpha'],
    dropout=0.0,
    target_modules=lora_config['target_modules']
)
# Need to reload state_dict since we modified it in place above
state_dict = checkpoint['model']
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
ref_model.load_state_dict(state_dict)

# Freeze ALL parameters - reference model should never be updated
for param in ref_model.parameters():
    param.requires_grad = False

ref_model.to(device)
ref_model.eval()
print("Reference model created and frozen")

# -----------------------------------------------------------------------------
# Step 2c: Load tokenizer
# -----------------------------------------------------------------------------
print("\nLoading tokenizer...")
tokenizer = tiktoken.get_encoding("gpt2")
print(f"Tokenizer loaded. Vocab size: {tokenizer.n_vocab}")

# -----------------------------------------------------------------------------
# Step 2d: Prepare some test prompts (Alpaca format)
# -----------------------------------------------------------------------------
# Alpaca instruction format
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

# Test prompts
test_prompts = [
    format_prompt("Why is sea water salty?"),
    format_prompt("What is the capital of France?"),
    format_prompt("Explain what machine learning is in one sentence."),
    format_prompt("Write a haiku about programming."),
    format_prompt("List three benefits of exercise."),
]

print(f"\nPrepared {len(test_prompts)} test prompts")
print_header("Example prompt:", char="-", width=40)
print(test_prompts[0][:200] + "...")

# -----------------------------------------------------------------------------
# Step 2e: Test the generate() function
# -----------------------------------------------------------------------------
print_header("Testing generate() function")

# Tokenize prompts
prompt_tokens = [tokenizer.encode(p) for p in test_prompts]

# Pad to same length (simple padding - use the longest prompt)
max_prompt_len = max(len(t) for t in prompt_tokens)
padded_prompts = []
for tokens in prompt_tokens:
    # Pad from the left with EOS token (GPT-2 convention)
    # The reason is padding from the right will cause model to hallucinate unrelated contents. 
    padding = [eos_token_id] * (max_prompt_len - len(tokens))
    padded_prompts.append(padding + tokens)

prompt_ids = torch.tensor(padded_prompts, device=device)
print(f"Prompt tensor shape: {prompt_ids.shape}")  # (batch_size, prompt_length)

# Generate responses
print(f"\nGenerating {max_new_tokens} tokens per prompt...")
with torch.no_grad():
    generated_ids, log_probs = generate(
        policy_model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )

print(f"Generated tensor shape: {generated_ids.shape}")  # (batch_size, prompt_len + max_new_tokens)
print(f"Log probs shape: {log_probs.shape}")  # (batch_size, max_new_tokens)

# Compute rewards
rewards = reward_fn.compute(generated_ids, max_prompt_len, max_new_tokens, eos_token_id)
print(f"Rewards shape: {rewards.shape}")  # (batch_size,)
print(f"Rewards: {rewards.tolist()}")

# Decode and display results
print_header("Generated Responses")

for i in range(len(test_prompts)):
    # Extract just the response part
    response_ids = generated_ids[i, max_prompt_len:].tolist()
    response_text = tokenizer.decode(response_ids)

    # Find EOS position for display
    eos_pos = None
    for j, tok in enumerate(response_ids):
        if tok == eos_token_id:
            eos_pos = j
            break

    print(f"\n--- Prompt {i+1} ---")
    print(f"Instruction: {test_prompts[i].split('Instruction:')[1].split('###')[0].strip()[:50]}...")
    print(f"Response: {response_text}")
    print(f"Reward: {rewards[i].item():.3f}")
    print(f"EOS at position: {eos_pos if eos_pos else 'Not found'}")
    print(f"Log prob sum: {log_probs[i].sum().item():.2f}")

print_header("Step 2 Complete! Environment is set up.")

# -----------------------------------------------------------------------------
# Step 3: PPO Training Loop
# -----------------------------------------------------------------------------
print_header("Step 3: Starting PPO Training")

# 3a. Set up optimizer (only LoRA parameters)
# We use AdamW with a small learning rate for stable RL training
optimizer = torch.optim.AdamW(
    [p for p in policy_model.parameters() if p.requires_grad],
    lr=learning_rate
)
print(f"Optimizer: AdamW, lr={learning_rate}")

# 3b. Load diverse training prompts from Alpaca dataset
print("\nLoading Alpaca dataset for diverse training prompts...")
alpaca_dataset = load_dataset("tatsu-lab/alpaca", split="train")
print(f"Loaded {len(alpaca_dataset):,} examples from Alpaca")

# Format prompts (instruction only, no response - model will generate that)
training_prompts = []
for example in alpaca_dataset:
    instruction = example['instruction']
    input_text = example.get('input', '')
    if input_text:
        # Has additional context
        training_prompts.append(format_prompt(f"{instruction}\n\nInput: {input_text}"))
    else:
        training_prompts.append(format_prompt(instruction))

# Shuffle for variety
random.seed(42)
random.shuffle(training_prompts)
print(f"Training prompts: {len(training_prompts):,} unique prompts")

# 3c. Initialize baseline for advantage estimation
# We use a simple running mean of rewards as the baseline
# advantage = reward - baseline  (how much better than average?)
baseline = 0.0

# Running averages for tracking (incremental mean formula)
running_reward_avg = 0.0
running_length_avg = 0.0

# 3d. Tracking metrics for monitoring
metrics_history = {
    "step": [],
    "reward_mean": [],
    "reward_std": [],
    "kl": [],
    "policy_loss": [],
    "response_length": [],
    "ratio_mean": [],
    "clip_fraction": [],
}

print(f"\nTraining config:")
print(f"  - Steps: {num_steps}")
print(f"  - Batch size: {batch_size}")
print(f"  - PPO epochs: {ppo_epochs}")
print(f"  - KL coefficient: {kl_coef}")
print(f"  - Clip epsilon: {clip_epsilon}")
print(f"  - Max new tokens: {max_new_tokens}")

print_header("Beginning training...", char="-")

for step in range(num_steps):
    # =========================================================================
    # Step 3.1: Sample a batch of prompts
    # =========================================================================
    # Simple cycling through prompts
    start_idx = (step * batch_size) % len(training_prompts)
    batch_prompts = training_prompts[start_idx:start_idx + batch_size]

    # Handle wrap-around
    if len(batch_prompts) < batch_size:
        batch_prompts = batch_prompts + training_prompts[:batch_size - len(batch_prompts)]

    # Tokenize and pad
    prompt_tokens = [tokenizer.encode(p) for p in batch_prompts]
    max_prompt_len = max(len(t) for t in prompt_tokens)
    padded_prompts = []
    for tokens in prompt_tokens:
        padding = [eos_token_id] * (max_prompt_len - len(tokens))
        padded_prompts.append(padding + tokens)
    # (batch_size, max_prompt_length)
    prompt_ids = torch.tensor(padded_prompts, device=device)

    # =========================================================================
    # Step 3.2: Generate responses with the current policy
    # =========================================================================
    # This collects the "old" log probs needed for importance sampling
    policy_model.eval()
    with torch.no_grad():
        generated_ids, old_log_probs = generate(
            policy_model,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

    # =========================================================================
    # Step 3.3: Compute rewards
    # =========================================================================
    # (batch_size,)
    rewards = reward_fn.compute(generated_ids, max_prompt_len, max_new_tokens, eos_token_id)

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
    # Step 3.4: Compute advantages
    # =========================================================================
    # Simple advantage: how much better than the running average?
    # advantage > 0: this response was better than average, reinforce it
    # advantage < 0: this response was worse than average, discourage it
    advantages = rewards - baseline

    # Update baseline with exponential moving average
    # This adapts to the current reward scale
    baseline = 0.9 * baseline + 0.1 * rewards.mean().item()

    # =========================================================================
    # Step 3.5: PPO update (multiple epochs on the same batch)
    # =========================================================================
    # Why multiple epochs? We want to squeeze more learning from each batch
    # of expensive generations. The clipping prevents the policy from
    # changing too much even with multiple updates.

    policy_model.train()

    for epoch in range(ppo_epochs):
        # Compute policy loss with PPO clipping
        policy_loss, stats = compute_policy_loss(
            policy_model,
            generated_ids,
            old_log_probs,
            advantages,
            max_prompt_len,
            clip_epsilon=clip_epsilon
        )

        # Compute KL penalty to stay close to reference model
        kl = compute_kl_penalty(
            policy_model,
            ref_model,
            generated_ids,
            max_prompt_len
        )

        # Total loss = policy loss + KL penalty
        # The KL term prevents the policy from drifting too far
        loss = policy_loss + kl_coef * kl

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in policy_model.parameters() if p.requires_grad],
            max_grad_norm
        )

        # Update weights
        optimizer.step()

    # =========================================================================
    # Step 3.6: Logging and monitoring
    # =========================================================================

    # Check training health
    check_training_health(stats, step)

    # Store metrics
    metrics_history["step"].append(step)
    metrics_history["reward_mean"].append(rewards.mean().item())
    metrics_history["reward_std"].append(rewards.std().item())
    metrics_history["kl"].append(kl.item())
    metrics_history["policy_loss"].append(policy_loss.item())
    metrics_history["response_length"].append(avg_response_length)
    metrics_history["ratio_mean"].append(stats["ratio_mean"])
    metrics_history["clip_fraction"].append(stats["clip_fraction"])

    # Update running averages: new_avg = (old_avg * n + new_value) / (n + 1)
    current_reward = rewards.mean().item()
    running_reward_avg = (running_reward_avg * step + current_reward) / (step + 1)
    running_length_avg = (running_length_avg * step + avg_response_length) / (step + 1)

    # Print progress
    if step % log_interval == 0:
        print(f"Step {step:4d} | "
              f"Reward: {current_reward:+.3f} (avg: {running_reward_avg:+.3f}) | "
              f"Length: {avg_response_length:5.1f} (avg: {running_length_avg:5.1f}) | "
              f"KL: {kl.item():.4f} | "
              f"Clip: {stats['clip_fraction']:.1%}")

    # Show sample generations
    if step % sample_interval == 0 and step > 0:
        print_header(f"Sample generation at step {step}:", char="-", width=40)

        # Decode first response in batch
        response_text = tokenizer.decode(response_ids[0].tolist())
        # Truncate at EOS for display
        if eos_token_id in response_ids[0].tolist():
            eos_idx = response_ids[0].tolist().index(eos_token_id)
            response_text = tokenizer.decode(response_ids[0, :eos_idx].tolist())

        instruction = batch_prompts[0].split("Instruction:")[1].split("###")[0].strip()
        print(f"Instruction: {instruction[:60]}...")
        print(f"Response: {response_text[:200]}...")
        print(f"Length: {response_lengths[0]} tokens, Reward: {rewards[0].item():.3f}")
        print("-" * 40)

# -----------------------------------------------------------------------------
# Step 3 Complete: Training Summary
# -----------------------------------------------------------------------------
print_header("Step 3 Complete! PPO Training Finished.")

# Print final metrics summary
print(f"""
Training Summary:
- Total steps: {num_steps}
- Final reward: {metrics_history['reward_mean'][-1]:.3f}
- Final KL: {metrics_history['kl'][-1]:.4f}
- Final response length: {metrics_history['response_length'][-1]:.1f}

Reward progression:
- Start: {metrics_history['reward_mean'][0]:.3f}
- End:   {metrics_history['reward_mean'][-1]:.3f}
- Change: {metrics_history['reward_mean'][-1] - metrics_history['reward_mean'][0]:+.3f}

Length progression:
- Start: {metrics_history['response_length'][0]:.1f}
- End:   {metrics_history['response_length'][-1]:.1f}
- Change: {metrics_history['response_length'][-1] - metrics_history['response_length'][0]:+.1f}

Overall averages:
- Reward: {running_reward_avg:+.3f}
- Length: {running_length_avg:.1f}
""")

# Final sample generations to observe any reward hacking
print_header("Final Sample Generations (look for reward hacking!)")

policy_model.eval()
with torch.no_grad():
    # Use original test prompts
    prompt_tokens = [tokenizer.encode(p) for p in test_prompts[:3]]
    max_prompt_len = max(len(t) for t in prompt_tokens)
    padded_prompts = []
    for tokens in prompt_tokens:
        padding = [eos_token_id] * (max_prompt_len - len(tokens))
        padded_prompts.append(padding + tokens)
    prompt_ids = torch.tensor(padded_prompts, device=device)

    final_generated, _ = generate(
        policy_model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )

for i in range(3):
    response_ids = final_generated[i, max_prompt_len:].tolist()

    # Find EOS
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

    # Check for repetition (simple heuristic)
    words = response_text.split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            print("⚠️  WARNING: High repetition detected! (unique ratio: {:.1%})".format(unique_ratio))

print_header("Look for signs of reward hacking:")
print("  - Repetitive text ('the the the...')")
print("  - Filler words to increase length")
print("  - Responses that never end")
print("  - Degraded response quality")
