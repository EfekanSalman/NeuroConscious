"""Action frequency bar chart visualization."""

import os
import matplotlib.pyplot as plt
from config.constants import OUTPUT_DIR, ACTION_FREQUENCY_PLOT


def plot_action_counts(counts: dict[str, int]) -> None:
    """Generate and save a bar chart of action frequencies.

    Args:
        counts: Mapping of action names to occurrence counts.
    """
    actions = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(8, 5))
    plt.bar(actions, values, color=["#2ecc71", "#3498db", "#95a5a6"])
    plt.title("Action Frequency Distribution")
    plt.xlabel("Actions")
    plt.ylabel("Times Chosen")
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, ACTION_FREQUENCY_PLOT)
    plt.savefig(path)
    plt.close()
    print(f"📊 Saved {path}")
