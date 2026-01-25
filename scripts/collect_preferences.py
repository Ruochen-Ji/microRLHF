"""
Preference Data Collection Script

Generates synthetic preference data or launches annotation UI.

Usage:
    # Launch annotation UI
    python scripts/collect_preferences.py --mode ui

    # Generate synthetic preferences (using stronger model as judge)
    python scripts/collect_preferences.py --mode synthetic --model gpt-4

Phase 3 Implementation
"""

import argparse

# TODO: Phase 3 Implementation


def launch_annotation_ui():
    """Launch Gradio UI for human preference annotation."""
    raise NotImplementedError("Phase 3: Annotation UI")


def generate_synthetic_preferences(model_name, num_samples):
    """Generate synthetic preferences using a stronger model as judge."""
    raise NotImplementedError("Phase 3: Synthetic preferences")


def main():
    parser = argparse.ArgumentParser(description="Collect preference data")
    parser.add_argument(
        "--mode",
        choices=["ui", "synthetic"],
        default="ui",
        help="Collection mode: ui for human annotation, synthetic for model-as-judge",
    )
    parser.add_argument(
        "--model",
        default="gpt-4",
        help="Model to use for synthetic preferences",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of preference samples to generate",
    )
    parser.add_argument(
        "--output",
        default="data/preferences/train.json",
        help="Output path for preference data",
    )
    args = parser.parse_args()

    if args.mode == "ui":
        launch_annotation_ui()
    else:
        generate_synthetic_preferences(args.model, args.num_samples)


if __name__ == "__main__":
    main()
