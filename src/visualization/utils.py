import matplotlib.pyplot as plt
import numpy as np
import os
import torch

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

        
def visualize_flow_evolution(model, dataloader, fig_path, n_steps=10, device='cuda'):
    model.eval()

    u, f = next(iter(dataloader))
    u    = u.flatten(start_dim=1).float().to(device)
    f    = f.flatten(start_dim=1).float().to(device)
    x_0  = torch.randn((f.flatten(start_dim=1).shape)).to(device)

    n_samples = min(4, f.shape[0])       
    dim = f.shape[1]
    H = W = int(dim**0.5)   
    t_vals = torch.linspace(0, 1, n_steps + 2, device=device)

    fig, axes = plt.subplots(n_samples, len(t_vals), figsize=(2.5*len(t_vals), 2.5*n_samples))
    if n_samples == 1:
        axes = axes.unsqueeze(0)

    for i in range(n_samples):
        x_t = x_0[i].unsqueeze(0) 

        for j, t in enumerate(t_vals):
            if j == 0:
                # source function
                x_plot = f[i]
                title = "condition (f)"
            elif j == len(t_vals) - 1:
                # ground truth
                x_plot = u[i]
                title = "t=1 (u) gt"
            elif j == len(t_vals) -2:
                # t=1
                dt = t_vals[-1] - t_vals[-2]          
                t_last = t_vals[-2]
                v_t = model(x_t, f[i] , t_last.repeat(x_t.size(0), 1))
                x_t = x_t + dt * v_t  
                x_plot = x_t.squeeze(0)
                title = "t=1 (u) pred"
            else:
                # one euler step: x_{t+dt} = x_t + dt * v(x_t, t)
                dt = t_vals[j] - t_vals[j-1]
                t_mid = (t_vals[j] + t_vals[j-1]) / 2
                v_t = model(x_t, f[i], t_mid.repeat(x_t.size(0), 1))
                x_t = x_t + dt * v_t
                x_plot = x_t.squeeze(0)
                title = f"t={t.item():.2f}"

            ax = axes[i, j]
            ax.imshow(x_plot.view(H, W).cpu(), cmap='viridis')
            ax.axis('off')
            if i == 0:
                ax.set_title(title, fontsize=9)

    plt.tight_layout()
    plt.show()
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
