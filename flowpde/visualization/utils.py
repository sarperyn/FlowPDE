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


@torch.no_grad()
def visualize_flow_evolution(
    model, 
    dataloader, 
    fig_path, 
    n_steps=10, 
    n_samples=4, 
    device='cuda',
    cmap='viridis',
    show_ground_truth=True,
    integration_method='euler',
    figsize_per_col=2.5,
    title_fontsize=9,
    dpi=200
):
    """
    Visualize the flow matching evolution from noise (t=0) to solution (t=1).
    
    Args:
        model: Flow matching model (expects signature: model(x, condition, t))
        dataloader: DataLoader yielding batches of (u, f) or dicts with keys 'u', 'f'
        fig_path: Path to save the visualization figure
        n_steps: Number of intermediate time steps to visualize (excluding t=0 and t=1)
        n_samples: Number of samples to visualize (rows in the figure)
        device: Device for computation
        cmap: Matplotlib colormap for imshow
        show_ground_truth: If True, show ground truth u at t=1 in last column
        integration_method: 'euler' or 'midpoint' for ODE integration
        figsize_per_col: Width/height per column in inches
        title_fontsize: Font size for column titles
        dpi: DPI for saved figure
    
    Returns:
        None (saves figure to fig_path)
    """
    model.eval()

    # Get batch from dataloader
    batch = next(iter(dataloader))
    
    # Handle both tuple (u, f) and dict {'u': ..., 'f': ...} formats
    if isinstance(batch, dict):
        u = batch['u']
        f = batch['f']
    else:
        u, f = batch
    
    # Flatten and move to device
    u = u.flatten(start_dim=1).float().to(device)
    f = f.flatten(start_dim=1).float().to(device)
    
    # Sample initial noise
    x_0 = torch.randn_like(f).to(device)
    
    # Determine number of samples and spatial dimensions
    n_samples = min(n_samples, f.shape[0])
    dim = f.shape[1]
    H = W = int(dim ** 0.5)
    
    # Create time steps (include t=0, intermediate steps, and t=1)
    n_cols = n_steps + 2 + (1 if show_ground_truth else 0)
    t_vals = torch.linspace(0, 1, n_steps + 2, device=device)
    
    # Create figure
    fig, axes = plt.subplots(
        n_samples, 
        n_cols, 
        figsize=(figsize_per_col * n_cols, figsize_per_col * n_samples)
    )
    
    # Handle single sample case
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Integrate each sample
    for i in range(n_samples):
        x_t = x_0[i].unsqueeze(0)  # Shape: (1, dim)
        cond = f[i].unsqueeze(0)    # Shape: (1, dim)
        
        col_idx = 0
        for j, t in enumerate(t_vals):
            if j == 0:
                # Initial noise x_0
                x_plot = x_t.squeeze(0)
                title = "t=0 (noise)"
            elif j == len(t_vals) - 1:
                # Final prediction at t=1
                dt = t_vals[-1] - t_vals[-2]
                t_prev = t_vals[-2]
                
                if integration_method == 'midpoint':
                    t_mid = (t_prev + t) / 2
                    v_t = model(x_t, cond, t_mid.repeat(x_t.size(0), 1))
                else:  # euler
                    v_t = model(x_t, cond, t_prev.repeat(x_t.size(0), 1))
                
                x_t = x_t + dt * v_t
                x_plot = x_t.squeeze(0)
                title = "t=1 (pred)"
            else:
                # Intermediate steps
                dt = t_vals[j] - t_vals[j-1]
                
                if integration_method == 'midpoint':
                    t_mid = (t_vals[j-1] + t) / 2
                    v_t = model(x_t, cond, t_mid.repeat(x_t.size(0), 1))
                else:  # euler
                    v_t = model(x_t, cond, t_vals[j-1].repeat(x_t.size(0), 1))
                
                x_t = x_t + dt * v_t
                x_plot = x_t.squeeze(0)
                title = f"t={t.item():.2f}"
            
            # Plot this time step
            ax = axes[i, col_idx]
            ax.imshow(x_plot.view(H, W).cpu().numpy(), cmap=cmap)
            ax.axis('off')
            if i == 0:
                ax.set_title(title, fontsize=title_fontsize)
            
            col_idx += 1
        
        # Optionally show ground truth in last column
        if show_ground_truth:
            ax = axes[i, col_idx]
            ax.imshow(u[i].view(H, W).cpu().numpy(), cmap=cmap)
            ax.axis('off')
            if i == 0:
                ax.set_title("t=1 (GT)", fontsize=title_fontsize)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(fig_path) or ".", exist_ok=True)
    plt.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved flow evolution visualization to {fig_path}")
