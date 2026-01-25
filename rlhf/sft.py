"""
Supervised Fine-Tuning (SFT)

Turns base GPT-2 into an instruction-following model.

Key concepts:
- Train on instruction-response pairs (Alpaca format)
- Model learns the FORMAT of helpful responses
- Does not necessarily learn WHAT humans prefer

Phase 2 Implementation:
- Instruction dataset loading (Alpaca format)
- SFT training loop
- Checkpoint saving with metadata
"""

# TODO: Phase 2 Implementation


class SFTTrainer:
    """Trainer for supervised fine-tuning on instruction datasets."""

    def __init__(self, model, config):
        """
        Args:
            model: GPT model to fine-tune
            config: SFT configuration
        """
        raise NotImplementedError("Phase 2: SFT Trainer")

    def train(self, dataset):
        """Run SFT training loop."""
        raise NotImplementedError("Phase 2: SFT training loop")
