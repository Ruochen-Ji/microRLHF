"""
NanoGPT Core Module

Contains the base GPT-2 model implementation and training utilities.
This module wraps the core nanoGPT code for use by the RLHF pipeline.

Core components:
- GPT model architecture
- Training loop
- Sampling/generation utilities
"""

from model import GPT, GPTConfig

__all__ = ['GPT', 'GPTConfig']
