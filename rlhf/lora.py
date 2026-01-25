"""
LoRA (Low-Rank Adaptation)

Memory-efficient fine-tuning by training low-rank decomposition matrices.

Key concepts:
- Freeze base model weights
- Add trainable low-rank matrices A and B
- W' = W + BA where B is (d, r) and A is (r, k), r << min(d, k)

Note: A LoRA implementation already exists at the project root (lora.py).
This module may wrap or extend that implementation for RLHF use cases.

Phase 2 Implementation (if needed):
- LoRA layer wrapper
- Apply LoRA to specific modules (attention, MLP)
- Merge/unmerge functionality
"""

# TODO: Phase 2 - Wrap existing lora.py or extend if needed

# The existing lora.py in the project root already implements LoRA.
# This file can serve as a wrapper or extension for RLHF-specific needs.

# from lora import LoRALayer  # Import from root


class LoRAConfig:
    """Configuration for LoRA adaptation."""

    def __init__(
        self,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules: list = None,
    ):
        """
        Args:
            r: Rank of the low-rank decomposition
            alpha: Scaling factor (alpha/r is the actual scaling)
            dropout: Dropout probability for LoRA layers
            target_modules: Which modules to apply LoRA to
        """
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ['c_attn', 'c_proj']
