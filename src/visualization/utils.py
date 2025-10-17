import matplotlib
#matplotlib.use("Agg")  # use headless backend before importing pyplot

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_curve(epoch_losses, title="Training Loss", xlabel="Epoch", ylabel="Loss", save_path="loss_curve.png"):

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

    epochs = np.arange(1, len(epoch_losses) + 1)
    ax.plot(epochs, epoch_losses, color="#1f77b4", linewidth=2.2, alpha=0.9, label=title)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss curve to {save_path}")
