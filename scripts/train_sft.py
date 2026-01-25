"""
SFT Training Script

Supervised fine-tuning on instruction datasets.

Usage:
    python scripts/train_sft.py --config configs/sft_config.yaml

    # With LoRA
    python scripts/train_sft.py --config configs/sft_config.yaml --use-lora

Phase 2 Implementation
"""

import argparse
import yaml

# TODO: Phase 2 Implementation


def load_config(config_path):
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_sft(config):
    """Run SFT training."""
    raise NotImplementedError("Phase 2: SFT training")


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning")
    parser.add_argument(
        "--config",
        default="configs/sft_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Enable LoRA fine-tuning",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.use_lora:
        config['model']['use_lora'] = True

    train_sft(config)


if __name__ == "__main__":
    main()
