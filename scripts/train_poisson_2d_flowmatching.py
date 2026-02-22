#!/usr/bin/env python3
"""
Poisson 2D Flow Matching Training Pipeline
==========================================

End-to-end script that:
1. Creates a Poisson 2D dataset (64x64 grid) using Exponax
2. Trains a UNet model using Flow Matching

Usage:
    python scripts/train_poisson_2d_flowmatching.py

    # With custom parameters
    python scripts/train_poisson_2d_flowmatching.py --epochs 200 --batch_size 32 --lr 1e-4
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flowpde.datasets.exponax import PoissonGenerator
from flowpde.datasets.wrappers import FlowDatasetWrapper
from flowpde.models.unet import UNet
from flowpde.flows.flow_matching import FlowMatching
from flowpde.trainers.flow_trainer import FlowTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Flow Matching on Poisson 2D")
    
    # Dataset parameters
    parser.add_argument("--num_points", type=int, default=64, help="Grid resolution (64 = 64x64)")
    parser.add_argument("--domain_extent", type=float, default=10.0, help="Physical domain size")
    parser.add_argument("--num_train_samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--num_test_samples", type=int, default=200, help="Number of test samples")
    
    # Model parameters
    parser.add_argument("--base_channels", type=int, default=64, help="Base channels for UNet")
    parser.add_argument("--use_attention", action="store_true", default=True, help="Use attention in UNet")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping (0 to disable)")
    
    # Flow Matching parameters
    parser.add_argument("--path", type=str, default="linear", choices=["linear", "ot_conditional"], help="Path type")
    parser.add_argument("--time_sampler", type=str, default="uniform", choices=["uniform", "logit_normal"], help="Time sampler")
    parser.add_argument("--sigma", type=float, default=0.0, help="Noise level for OT path")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="results/poisson_2d_flowmatching", help="Output directory")
    parser.add_argument("--print_interval", type=int, default=10, help="Print stats every N epochs")
    parser.add_argument("--save_interval", type=int, default=50, help="Save model every N epochs")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print("=" * 60)
    print("Poisson 2D Flow Matching Training Pipeline")
    print("=" * 60)
    print(f"\nDevice: {args.device}")
    print(f"Random Seed: {args.seed}")
    

    # Create Poisson 2D Dataset
    print("\n" + "-" * 60)
    print("Creating Poisson 2D Dataset (Exponax)")
    print("-" * 60)
    
    print(f"  Grid resolution: {args.num_points}x{args.num_points}")
    print(f"  Domain extent: {args.domain_extent}")
    print(f"  Train samples: {args.num_train_samples}")
    print(f"  Test samples: {args.num_test_samples}")
    
    generator = PoissonGenerator(
        num_spatial_dims=2,
        num_points=args.num_points,
        domain_extent=args.domain_extent,
    )
    
    train_dataset = generator.generate(
        num_samples=args.num_train_samples,
        seed=args.seed,
        problem='forward',
    )
    test_dataset = generator.generate(
        num_samples=args.num_test_samples,
        seed=args.seed + 773,
        problem='forward',
    )
    
    print(f"\n  Train dataset size: {len(train_dataset)}")
    print(f"  Test dataset size: {len(test_dataset)}")
    
    # Examine a sample
    sample = train_dataset[0]
    print(f"\n  Sample structure:")
    print(f"    Input (source) shape: {sample['input'].shape}")
    print(f"    Target (solution) shape: {sample['target'].shape}")
    
    # Wrap datasets for FlowMatching ({'input','target'} → {'f','u'})
    train_dataset_wrapped = FlowDatasetWrapper(train_dataset)
    test_dataset_wrapped = FlowDatasetWrapper(test_dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset_wrapped, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,  # JAX/NumPy data, keep simple
        pin_memory=False,
    )
    
    test_loader = DataLoader(
        test_dataset_wrapped,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    
    print(f"\n  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create UNet Model
    print("\n" + "-" * 60)
    print("Creating UNet Model")
    print("-" * 60)
    
    model = UNet(
        spatial_dim=2,
        spatial_size=args.num_points,
        base_channels=args.base_channels,
        solution_channels=1,
        condition_channels=1,
        use_attention=args.use_attention,
        norm_type="group",
        activation="swish",
        return_spatial=False,  # Return flattened output for flow matching
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: UNet")
    print(f"  Spatial size: {args.num_points}x{args.num_points}")
    print(f"  Base channels: {args.base_channels}")
    print(f"  Use attention: {args.use_attention}")
    print(f"  Trainable parameters: {num_params:,}")
    
    # Setup Flow Matching
    print("\n" + "-" * 60)
    print("Setting up Flow Matching")
    print("-" * 60)
    
    flow = FlowMatching(
        model=model,
        path=args.path,
        time_sampler=args.time_sampler,
        sigma=args.sigma,
    )
    
    print(f"  Path: {args.path}")
    print(f"  Time sampler: {args.time_sampler}")
    print(f"  Sigma: {args.sigma}")
    

    # Setup Optimizer and Scheduler
    print("\n" + "-" * 60)
    print("Setting up Optimizer and Scheduler")
    print("-" * 60)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,  # Min LR is 1% of initial
    )
    
    print(f"  Optimizer: AdamW")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Scheduler: CosineAnnealingLR")
    
    # Create Trainer
    print("\n" + "-" * 60)
    print("Creating Trainer")
    print("-" * 60)
    
    trainer = FlowTrainer(
        flow=flow,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        gradient_clip=args.gradient_clip if args.gradient_clip > 0 else None,
        use_amp=False, 
    )
    
    print(f"  Device: {args.device}")
    print(f"  Gradient clipping: {args.gradient_clip if args.gradient_clip > 0 else 'disabled'}")
    
    # Train
    print("\n" + "-" * 60)
    print("Training")
    print("-" * 60)
    
    # Create output directory
    output_dir = Path(project_root) / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Output directory: {output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Print interval: {args.print_interval}")
    print(f"  Save interval: {args.save_interval}")
    
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60 + "\n")
    
    trainer.train(
        data_loader=train_loader,
        epochs=args.epochs,
        print_stats_interval=args.print_interval,
        save_interval=args.save_interval,
        save_dir=str(checkpoint_dir),
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nBest loss: {trainer.best_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"\nTo run inference, load the model from:")
    print(f"  {checkpoint_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
