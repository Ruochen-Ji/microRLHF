"""
LoRA Finetuning Configuration for Shakespeare Dataset

This config demonstrates finetuning GPT-2 on Shakespeare using LoRA.
Compare this to the standard finetune_shakespeare.py to see the differences!

Key differences from full finetuning:
1. We set use_lora = True to enable LoRA
2. We can use a higher learning rate (LoRA is more stable)
3. Training is MUCH faster due to fewer parameters
4. Memory usage is lower (fewer gradients to store)

To run:
    python train.py config/finetune_shakespeare_lora.py

After training, you can sample with:
    python sample.py --out_dir=out-shakespeare-lora
"""

import time

# Output directory (separate from full finetuning)
out_dir = 'out-shakespeare-lora'

# Evaluation settings
eval_interval = 5
eval_iters = 40
wandb_log = False  # Set to True if you want to track with wandb
wandb_project = 'shakespeare'
wandb_run_name = 'lora-' + str(time.time())

# Dataset
dataset = 'shakespeare'

# Initialize from pretrained GPT-2
# Options: 'gpt2' (124M), 'gpt2-medium' (350M), 'gpt2-large' (774M), 'gpt2-xl' (1.5B)
# Start with 'gpt2' for faster experimentation, then try larger models
init_from = 'gpt2'

# Only save checkpoints if the validation loss improves
always_save_checkpoint = False

# =============================================================================
# LoRA Configuration
# =============================================================================
use_lora = True

# Rank: The dimensionality of the low-rank decomposition
# - Lower rank (4-8): Fewer parameters, faster training, may underfit
# - Higher rank (16-64): More expressive, slower, may overfit
# Recommendation: Start with 8, increase if underfitting
lora_rank = 8

# Alpha: Scaling factor for LoRA contribution
# - The LoRA output is scaled by alpha/rank
# - Higher alpha = stronger adaptation effect
# - Common practice: set alpha = 2 * rank
lora_alpha = 16.0

# Dropout: Regularization for LoRA layers
# - 0.0 for small datasets/short training
# - 0.05-0.1 for longer training or larger datasets
lora_dropout = 0.05

# Target modules: Which layers to apply LoRA to
# Options for GPT-2:
# - 'c_attn': Query/Key/Value projection (most important!)
# - 'c_proj': Attention output projection
# - 'c_fc': MLP first layer
# 
# The paper found Q and V projections most impactful.
# Since nanoGPT combines Q,K,V into c_attn, we target that.
# Adding c_proj gives a good balance of expressivity and efficiency.
lora_target_modules = ['c_attn', 'c_proj']

# =============================================================================
# Training Configuration
# =============================================================================

# Batch size settings
# With LoRA, we can often use larger batch sizes due to memory savings
batch_size = 4
gradient_accumulation_steps = 8

# Total training iterations
# LoRA often converges faster, so we can use fewer iterations
max_iters = 100

# Learning rate
# LoRA can handle higher learning rates than full finetuning!
# - Full finetuning typically uses 1e-5 to 3e-5
# - LoRA can use 1e-4 to 3e-4
learning_rate = 2e-4

# No learning rate decay for this short run
decay_lr = False

# Weight decay (regularization)
# Lower than full finetuning since we have fewer parameters
weight_decay = 0.01

# =============================================================================
# Performance Settings
# =============================================================================

# Disable compilation for easier debugging during learning
# Set to True for faster training once you understand the code
compile = False

# Block size (context length)
# GPT-2 was trained with 1024, but we can use smaller for faster training
block_size = 256

# Dropout for the base model (separate from LoRA dropout)
dropout = 0.1
