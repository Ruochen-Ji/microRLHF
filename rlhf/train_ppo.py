"""
train_ppo.py - PPO training script for baby RLHF

Step 2: Environment Setup
- Load SFT'd GPT-2 (with LoRA)
- Create frozen reference model
- Load Alpaca prompts
- Test generation
"""
import os
import sys
import torch
import tiktoken

# Add parent directory to path so we can import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT
from lora import apply_lora_to_model
from rl_utils import generate, compute_reward

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# SFT checkpoint path
sft_checkpoint_path = "out-alpaca-lora/ckpt.pt"

# Generation config
max_new_tokens = 400
temperature = 1.0

# GPT-2 special tokens
eos_token_id = 50256  # <|endoftext|>

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
print("Example prompt:")
print("-" * 40)
print(test_prompts[0][:200] + "...")
print("-" * 40)

# -----------------------------------------------------------------------------
# Step 2e: Test the generate() function
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Testing generate() function")
print("=" * 60)

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
rewards = compute_reward(generated_ids, max_prompt_len, max_new_tokens, eos_token_id)
print(f"Rewards shape: {rewards.shape}")  # (batch_size,)
print(f"Rewards: {rewards.tolist()}")

# Decode and display results
print("\n" + "=" * 60)
print("Generated Responses")
print("=" * 60)

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

print("\n" + "=" * 60)
print("Step 2 Complete! Environment is set up.")
print("=" * 60)
print(f"""
Summary:
- Policy model: GPT + LoRA (rank={lora_config['rank']})
  - Total params: {sum(p.numel() for p in policy_model.parameters()):,}
  - Trainable (LoRA): {sum(p.numel() for p in policy_model.parameters() if p.requires_grad):,}
- Reference model: Frozen copy of SFT model
- Tokenizer: GPT-2 ({tokenizer.n_vocab} vocab)
- Generation: {max_new_tokens} tokens, temperature={temperature}
- Rewards working: {rewards.shape[0]} computed

Next: Step 3 - PPO Training Loop
""")
