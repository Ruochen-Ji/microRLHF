"""
Generic training metrics plotter.

Reads any CSV with a "step" column and plots all other numeric columns.
Works for reward model, PPO, SFT, or any training run.

Usage:
    python rlhf/analysis/plot_training.py <csv_path> [output_path]

Examples:
    python rlhf/analysis/plot_training.py rlhf/reward_model_log.csv
    python rlhf/analysis/plot_training.py rlhf/ppo_log.csv rlhf/analysis/ppo_progress.png
"""

import sys
import csv
import os
import math
import matplotlib.pyplot as plt

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def load_csv(path):
    """Load CSV and return {column_name: [(step, value), ...]} for all numeric columns."""
    series = {}

    with open(path) as f:
        reader = csv.DictReader(f)
        columns = [c for c in reader.fieldnames if c != "step"]

        for col in columns:
            series[col] = []

        for row in reader:
            step = int(row["step"])
            for col in columns:
                val = row[col].strip()
                if val:
                    try:
                        series[col].append((step, float(val)))
                    except ValueError:
                        pass

    # Drop columns with no data
    series = {k: v for k, v in series.items() if len(v) > 0}
    return series


def plot(series, output_path):
    """Plot each metric in its own subplot."""
    n = len(series)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows))

    # Flatten axes for easy indexing
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (name, data) in enumerate(series.items()):
        ax = axes[i]
        steps = [d[0] for d in data]
        values = [d[1] for d in data]
        color = COLORS[i % len(COLORS)]

        ax.plot(steps, values, color=color, linewidth=1.2, alpha=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.2)

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved {n} plots to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_training.py <csv_path> [output_path]")
        sys.exit(1)

    csv_path = sys.argv[1]
    # Default output: same directory as CSV, with .png extension
    default_output = os.path.splitext(csv_path)[0] + ".png"
    output_path = sys.argv[2] if len(sys.argv) > 2 else default_output

    series = load_csv(csv_path)
    print(f"Found {len(series)} metrics: {list(series.keys())}")
    plot(series, output_path)
