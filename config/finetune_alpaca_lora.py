"""
LoRA Finetuning Configuration for Alpaca Dataset (Instruction-Tuning)

This config finetunes GPT-2 on the Alpaca instruction-following dataset
to create a simple Q&A / instruction-following model.

To run:
    1. First prepare the data:
       python data/alpaca/prepare.py
    
    2. Then run training:
       python train.py config/finetune_alpaca_lora.py

    3. Sample from your model:
       python sample.py --out_dir=out-alpaca-lora --start="Below is an instruction that describes a task. Write a response that appropriately completes the request.\\n\\n### Instruction:\\nWhat is the capital of France?\\n\\n### Response:\\n"

Note: GPT-2 is much smaller than the models Alpaca was designed for (LLaMA 7B).
Expect decent results on simple tasks, but limited capability on complex reasoning.
"""

import time

# Output directory
out_dir = 'out-alpaca-lora'

# Evaluation settings
eval_interval = 100  # Evaluate every 100 iterations
eval_iters = 50
wandb_log = False
wandb_project = 'alpaca'
wandb_run_name = 'gpt2-lora-' + str(time.time())

# Dataset
dataset = 'alpaca'

# Initialize from pretrained GPT-2
# Options: 'gpt2' (124M), 'gpt2-medium' (350M), 'gpt2-large' (774M), 'gpt2-xl' (1.5B)
# With your 32GB 5090, you can easily run gpt2-xl!
init_from = 'gpt2-medium'  # Good balance of speed and capability

# Checkpoint settings
always_save_checkpoint = False  # Only save when val loss improves

# =============================================================================
# LoRA Configuration
# =============================================================================
use_lora = True
lora_rank = 16  # Slightly higher rank for more complex instruction-following
lora_alpha = 32.0
lora_dropout = 0.05
lora_target_modules = ['c_attn', 'c_proj']  # Attention layers

# =============================================================================
# Training Configuration  
# =============================================================================

# Batch size settings
# Alpaca has 52K examples, so we want decent batch sizes
batch_size = 4
gradient_accumulation_steps = 16  # Effective batch size = 4 * 16 = 64

# Training iterations
# With ~25M tokens and batch_size*grad_accum*block_size = 64*1024 = 65K tokens/iter
# One epoch ≈ 400 iterations. Let's do ~2-3 epochs.
max_iters = 1000

# Learning rate
# Setting a higher learning rate because Lora can handle higher learning rate than full finetuning
learning_rate = 3e-4  # LoRA can handle higher LR

# Learning rate schedule
decay_lr = True
warmup_iters = 50
lr_decay_iters = 1000
min_lr = 3e-5

# Weight decay
weight_decay = 0.01

# =============================================================================
# Performance Settings
# =============================================================================

# Enable compilation for faster training (your 5090 will love this)
compile = True

# Context length - GPT-2 max is 1024
block_size = 1024

# Dropout for regularization
dropout = 0.1
