import torch
import os


def save_model(*args):
        
        os.makedirs(args.save_dir, exits_ok=True) # Create saving directory if not exists
        ckpt_path = os.path.join(args.save_dir, f"model_{args.epoch+1}.pt") # Define checkpoint path

        # Save model and its arguments
        torch.save({ 
        "model_state": args.model.state_dict(),
        "optimizer_state": args.optimizer.state_dict(),
        "scheduler_state": args.lr_scheduler.state_dict(),
        "train_loss": args.epoch_loss,    
        }, ckpt_path)


def print_stats(**kwargs):
    parts = []
    for k, v in kwargs.items():
        # Auto-format based on value type
        if isinstance(v, float):
            # Use scientific notation if it's small or large
            if abs(v) < 1e-3 or abs(v) > 1e3:
                v_str = f"{v:.2e}"
            else:
                v_str = f"{v:.6f}".rstrip("0").rstrip(".")
        elif isinstance(v, int):
            v_str = f"{v}"
        else:
            v_str = str(v)
        parts.append(f"{k}: {v_str}")

    print(" | ".join(parts))
