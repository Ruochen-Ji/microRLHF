"""
Reward Model Training Script

Trains a reward model on preference data using Bradley-Terry loss.

Usage:
    python scripts/train_reward_model.py --config configs/reward_config.yaml

Phase 4 Implementation
"""

import argparse
import yaml

# TODO: Phase 4 Implementation


def load_config(config_path):
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_reward_model(config):
    """Train reward model on preference data."""
    raise NotImplementedError("Phase 4: Reward model training")


def main():
    parser = argparse.ArgumentParser(description="Train reward model")
    parser.add_argument(
        "--config",
        default="configs/reward_config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train_reward_model(config)


if __name__ == "__main__":
    main()
