"""
Burgers 1D Rectified Flow Training
"""

import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flowpde.datasets.exponax import BurgersGenerator, BurgersConfig
from flowpde.datasets.wrappers import FlowDatasetWrapper
from flowpde.models.resnet import ResNet
from flowpde.flows.flow_matching import FlowMatching
from flowpde.trainers.flow_trainer import FlowTrainer
from flowpde.core.base_conditioner import ConcatConditioner, NullConditioner


def parse_args():
    parser = argparse.ArgumentParser(description="Train Rectified Flow on Burgers 1D")
    
    # Dataset
    parser.add_argument("--num_points", type=int, default=160)
    parser.add_argument("--diffusivity_min", type=float, default=1e-4)
    parser.add_argument("--diffusivity_max", type=float, default=1e-2)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--domain_extent", type=float, default=1.0)
    parser.add_argument("--num_train_samples", type=int, default=1000)
    
    # Model
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--conditioner", type=str, default="concat", choices=["concat", "null"],
                        help="Conditioning strategy: 'concat' (default) or 'null' (unconditional)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="results/burgers_1d_rectified_flow")
    parser.add_argument("--print_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=50)
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    print(f"Burgers 1D Rectified Flow | Device: {args.device}")
    
    # Dataset
    print("Generating dataset with Exponax...")
    generator = BurgersGenerator(
        num_spatial_dims=1,
        num_points=args.num_points,
        domain_extent=args.domain_extent,
        dt=args.dt,
        num_steps=args.num_steps,
        diffusivity_min=args.diffusivity_min,
        diffusivity_max=args.diffusivity_max,
        torch_device=args.device,
    )
    train_dataset = generator.generate(num_samples=args.num_train_samples, seed=args.seed, problem='forward')
    print(f"Train samples: {len(train_dataset)}")
    
    train_loader = DataLoader(
        FlowDatasetWrapper(train_dataset),
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory=False,
    )
    
    # Model
    print("Creating model...")
    conditioner = ConcatConditioner(dim=1) if args.conditioner == "concat" else NullConditioner()
    model = ResNet(
        spatial_dim=1,
        spatial_size=args.num_points,
        base_channels=args.base_channels,
        blocks_per_stage=[2, 2, 2],
        solution_channels=1,
        condition_channels=1,
        downsample=False,
        return_spatial=False,
        conditioner=conditioner,
    )
    print(f"Conditioner: {args.conditioner}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Rectified Flow: linear path + logit_normal time sampler
    flow = FlowMatching(
        model=model,
        path='linear',
        time_sampler='logit_normal',
    )
    print("Using Rectified Flow (logit_normal time sampling)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Trainer
    trainer = FlowTrainer(
        flow=flow,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        gradient_clip=args.gradient_clip if args.gradient_clip > 0 else None,
    )
    
    # Output dir
    output_dir = Path(project_root) / args.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    print(f"Training for {args.epochs} epochs...")
    trainer.train(
        data_loader=train_loader,
        epochs=args.epochs,
        print_stats_interval=args.print_interval,
        save_interval=args.save_interval,
        save_dir=str(checkpoint_dir),
    )
    
    print(f"Done. Best loss: {trainer.best_loss:.6f}")
    print(f"Checkpoint: {checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
