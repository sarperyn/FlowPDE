#!/usr/bin/env python3
"""
Darcy 2D Flow Matching Training Pipeline
=========================================

End-to-end script that:
1. Creates a Darcy-flow dataset (64x64 grid) using a custom JAX FD+CG solver
2. Trains a UNet model using Flow Matching

The Darcy benchmark solves the variable-coefficient elliptic PDE:

$$
-\nabla\cdot(\kappa(x)\nabla u(x)) = f(x) \quad \text{on } \Omega=[0,1]^2,\qquad
u=0 \quad \text{on } \partial\Omega.
$$

where κ is a log-normal random field drawn from a GRF (matching the
original FNO Darcy benchmark setup: Kovachki et al., 2021).

Forward problem (default): $(\kappa, f) \rightarrow u$
    condition_channels = 2   [$\mathrm{cat}([\kappa, f], \mathrm{dim}=0)$]
    solution_channels  = 1   [$u$]

Inverse problem: $u \rightarrow \kappa$
    condition_channels = 1   [observed $u$ (possibly noisy/masked)]
    solution_channels  = 1   [$\kappa$]

Usage:
    python scripts/train_darcy_2d_flowmatching.py

    # With custom parameters
    python scripts/train_darcy_2d_flowmatching.py --epochs 200 --batch_size 16

    # Inverse problem
    python scripts/train_darcy_2d_flowmatching.py --problem inverse --obs_noise_std 0.01
"""

import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flowpde.datasets.exponax import DarcyGenerator, DarcyConfig
from flowpde.datasets.wrappers import FlowDatasetWrapper
from flowpde.models.unet import UNet
from flowpde.flows.flow_matching import FlowMatching
from flowpde.trainers.flow_trainer import FlowTrainer
from flowpde.core.base_conditioner import ConcatConditioner, NullConditioner


def parse_args():
    parser = argparse.ArgumentParser(description="Train Flow Matching on Darcy 2D")

    # Dataset parameters
    parser.add_argument("--num_points", type=int, default=64,
                        help="Grid resolution (64 = 64x64)")
    parser.add_argument("--domain_extent", type=float, default=1.0,
                        help="Physical domain size (standard Darcy: 1.0)")
    parser.add_argument("--num_train_samples", type=int, default=1000,
                        help="Number of training samples")
    parser.add_argument("--num_test_samples", type=int, default=200,
                        help="Number of test samples")

    # Problem type
    parser.add_argument("--problem", type=str, default="forward",
                        choices=["forward", "inverse"],
                        help="'forward': (κ,f)→u | 'inverse': u→κ")

    # κ field parameters (GRF)
    parser.add_argument("--kappa_alpha", type=float, default=2.0,
                        help="Spectral decay exponent for κ GRF (higher = smoother)")
    parser.add_argument("--kappa_tau", type=float, default=3.0,
                        help="Inverse correlation length for κ GRF (higher = more oscillatory)")
    parser.add_argument("--kappa_scale", type=float, default=1.0,
                        help="Log-contrast scale for κ (higher = larger permeability contrast)")

    # Solver parameters
    parser.add_argument("--cg_steps", type=int, default=500,
                        help="Fixed CG iterations for the FD linear system")

    # Inverse-problem noise/masking
    parser.add_argument("--obs_noise_std", type=float, default=0.0,
                        help="Additive Gaussian noise on observed u (inverse only)")
    parser.add_argument("--obs_mask_fraction", type=float, default=1.0,
                        help="Fraction of observed u pixels (1.0 = fully observed)")

    # Model parameters
    parser.add_argument("--base_channels", type=int, default=64,
                        help="Base channels for UNet")
    parser.add_argument("--use_attention", action="store_true", default=True,
                        help="Use self-attention at UNet bottleneck")
    parser.add_argument("--conditioner", type=str, default="concat",
                        choices=["concat", "null"],
                        help="Conditioning strategy: 'concat' (default) or 'null' (unconditional)")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay")
    parser.add_argument("--gradient_clip", type=float, default=1.0,
                        help="Gradient clipping (0 to disable)")

    # Flow Matching parameters
    parser.add_argument("--path", type=str, default="linear",
                        choices=["linear", "ot_conditional"],
                        help="Interpolation path type")
    parser.add_argument("--time_sampler", type=str, default="uniform",
                        choices=["uniform", "logit_normal"],
                        help="Time sampling distribution")
    parser.add_argument("--sigma", type=float, default=0.0,
                        help="Noise level for OT path")

    # Output parameters
    parser.add_argument("--output_dir", type=str,
                        default="results/darcy_2d_flowmatching",
                        help="Output directory")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="Print stats every N epochs")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="Save model every N epochs")

    # Device
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("=" * 60)
    print("Darcy 2D Flow Matching Training Pipeline")
    print("=" * 60)
    print(f"\nDevice: {args.device}")
    print(f"Random Seed: {args.seed}")
    print(f"Problem: {args.problem}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("Creating Darcy 2D Dataset")
    print("-" * 60)

    print(f"  Grid resolution: {args.num_points}x{args.num_points}")
    print(f"  Domain extent:   {args.domain_extent}")
    print(f"  Train samples:   {args.num_train_samples}")
    print(f"  Test samples:    {args.num_test_samples}")
    print(f"  κ GRF: α={args.kappa_alpha}, τ={args.kappa_tau}, scale={args.kappa_scale}")
    print(f"  CG steps:        {args.cg_steps}")
    if args.problem == "inverse":
        print(f"  Obs noise std:   {args.obs_noise_std}")
        print(f"  Obs mask frac:   {args.obs_mask_fraction}")

    generator = DarcyGenerator(
        num_spatial_dims=2,
        num_points=args.num_points,
        domain_extent=args.domain_extent,
        kappa_alpha=args.kappa_alpha,
        kappa_tau=args.kappa_tau,
        kappa_scale=args.kappa_scale,
        cg_steps=args.cg_steps,
        obs_noise_std=args.obs_noise_std,
        obs_mask_fraction=args.obs_mask_fraction,
    )

    train_dataset = generator.generate(
        num_samples=args.num_train_samples,
        seed=args.seed,
        problem=args.problem,
    )
    test_dataset = generator.generate(
        num_samples=args.num_test_samples,
        seed=args.seed + 773,
        problem=args.problem,
    )

    print(f"\n  Train dataset size: {len(train_dataset)}")
    print(f"  Test dataset size:  {len(test_dataset)}")

    # Examine a sample to confirm shapes
    sample = train_dataset[0]
    print(f"\n  Sample structure:")
    print(f"    Input  shape: {sample['input'].shape}")
    print(f"    Target shape: {sample['target'].shape}")

    # condition_channels depends on the problem type:
    #   forward:  input = cat([κ, f], dim=0)  → 2 channels
    #   inverse:  input = u                   → 1 channel
    condition_channels = sample['input'].shape[0]
    solution_channels  = sample['target'].shape[0]
    print(f"\n  condition_channels: {condition_channels}")
    print(f"  solution_channels:  {solution_channels}")

    # Wrap datasets for FlowMatching ({'input','target'} → {'f','u'})
    train_dataset_wrapped = FlowDatasetWrapper(train_dataset)
    test_dataset_wrapped  = FlowDatasetWrapper(test_dataset)

    train_loader = DataLoader(
        train_dataset_wrapped,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
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
    print(f"  Test batches:  {len(test_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("Creating UNet Model")
    print("-" * 60)

    conditioner = ConcatConditioner(dim=1) if args.conditioner == "concat" else NullConditioner()

    model = UNet(
        spatial_dim=2,
        spatial_size=args.num_points,
        base_channels=args.base_channels,
        solution_channels=solution_channels,
        condition_channels=condition_channels,
        use_attention=args.use_attention,
        norm_type="group",
        activation="swish",
        return_spatial=False,
        conditioner=conditioner,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model:             UNet")
    print(f"  Spatial size:      {args.num_points}x{args.num_points}")
    print(f"  Base channels:     {args.base_channels}")
    print(f"  Use attention:     {args.use_attention}")
    print(f"  Conditioner:       {args.conditioner}")
    print(f"  condition_channels:{condition_channels}")
    print(f"  solution_channels: {solution_channels}")
    print(f"  Trainable params:  {num_params:,}")

    # ── Flow Matching ─────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("Setting up Flow Matching")
    print("-" * 60)

    flow = FlowMatching(
        model=model,
        path=args.path,
        time_sampler=args.time_sampler,
        sigma=args.sigma,
    )

    print(f"  Path:         {args.path}")
    print(f"  Time sampler: {args.time_sampler}")
    print(f"  Sigma:        {args.sigma}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
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
        eta_min=args.lr * 0.01,
    )

    print(f"  Optimizer: AdamW")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay:  {args.weight_decay}")
    print(f"  Scheduler:     CosineAnnealingLR")

    # ── Trainer ───────────────────────────────────────────────────────────────
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

    print(f"  Device:            {args.device}")
    print(f"  Gradient clipping: {args.gradient_clip if args.gradient_clip > 0 else 'disabled'}")

    # ── Output directory ──────────────────────────────────────────────────────
    output_dir    = Path(project_root) / args.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("Training")
    print("-" * 60)
    print(f"  Output directory: {output_dir}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Print interval:   {args.print_interval}")
    print(f"  Save interval:    {args.save_interval}")

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
