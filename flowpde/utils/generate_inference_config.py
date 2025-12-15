
"""
Utility script to create inference configuration files from training configuration files.

This script reads a training YAML config and generates an inference YAML config
with appropriate settings for running inference on trained models.

Usage:
    python utils/create_inference_config.py --training_config configs/training/mlp_flow_poisson.yaml
    python utils/create_inference_config.py --training_config configs/training/unet_flow_poisson.yaml --checkpoint_path checkpoints/my_model/model_999.pt
"""

import os
import sys
import yaml
import argparse
from typing import Dict, Optional, Any

# Import shared utilities
from flowpde.utils.utils import load_config, find_latest_checkpoint


def save_yaml(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary as YAML file with header comments."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        # Write header comments
        f.write("# Inference configuration for trained flow matching model\n")
        f.write(f"# Usage: python tests/generic_test_inference.py --config {filepath}\n\n")
        # Write YAML content
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=2)
    print(f"✓ Inference config saved to: {filepath}")


def create_inference_config(
    training_config: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    n_steps: int = 50,
    integration_method: str = "rk4",
    vis_n_steps: int = 10,
    n_samples: int = 4,
    batch_size: int = 32,
    metrics: list = None,
    sample_batch_size: int = 4,
    save_samples: bool = True,
    dpi: int = 200,
    data_pattern: Optional[str] = None,
    auto_find_checkpoint: bool = True,
    project_dir: str = "."
) -> Dict[str, Any]:
    """
    Create an inference configuration from a training configuration.
    
    Args:
        training_config: Loaded training config dictionary
        checkpoint_path: Path to model checkpoint (relative to project root)
        output_dir: Directory for inference results
        n_steps: Number of ODE integration steps
        integration_method: 'euler', 'midpoint', or 'rk4'
        vis_n_steps: Number of steps shown in visualization
        n_samples: Number of samples to visualize
        batch_size: Batch size for evaluation
        metrics: List of metrics to compute
        sample_batch_size: Batch size for sample generation
        save_samples: Whether to save generated samples
        dpi: Figure DPI for visualizations
        data_pattern: Custom data pattern for inference data
        auto_find_checkpoint: Whether to automatically find latest checkpoint
        project_dir: Project root directory
    
    Returns:
        Dictionary containing inference configuration
    """
    if metrics is None:
        metrics = ['mse', 'relative_l2']
    
    # Create base inference config by copying relevant fields from training config
    inference_config = {}
    
    # Copy essential fields from training config
    inference_config['name'] = training_config.get('name', 'experiment')
    inference_config['seed'] = training_config.get('seed', 42)
    inference_config['data_dir'] = training_config.get('data_dir', 'data/static/poisson')
    inference_config['spatial_dim'] = training_config.get('spatial_dim', 32)
    
    # Copy model configuration (required for loading model)
    inference_config['model_config'] = training_config.get('model_config', {})
    
    # Copy training config class (needed to identify trainer type)
    inference_config['training_config'] = {
        'class': training_config.get('training_config', {}).get('class', 'src.trainers.flow_matching.FlowMatchingTrainer')
    }
    
    # Copy dataset configuration
    inference_config['dataset_config'] = training_config.get('dataset_config', {})
    
    # Auto-generate or find checkpoint path if not provided
    if checkpoint_path is None:
        if auto_find_checkpoint:
            # Try to find the latest checkpoint automatically using shared utility
            found_checkpoint = find_latest_checkpoint(project_dir, training_config)
            if found_checkpoint:
                checkpoint_path = found_checkpoint
                print(f"✓ Found latest checkpoint: {checkpoint_path}")
            else:
                # Fall back to placeholder path
                name = training_config.get('name', 'experiment')
                spatial_dim = training_config.get('spatial_dim', '')
                checkpoint_path = f"results/training/{name}_{spatial_dim}/checkpoints/model_latest.pt"
                print(f"⚠ No checkpoint found - using placeholder: {checkpoint_path}")
        else:
            # Default checkpoint location based on training config
            name = training_config.get('name', 'experiment')
            spatial_dim = training_config.get('spatial_dim', '')
            checkpoint_path = f"results/training/{name}_{spatial_dim}/checkpoints/model_latest.pt"
    
    # Set output directory
    if output_dir is None:
        output_dir = training_config.get('training_config', {}).get('output_dir', 'results/training')
        output_dir = output_dir.replace('training', 'inference')
    
    # Create inference configuration section
    inference_config['inference_config'] = {
        'checkpoint_path': checkpoint_path,
    }
    
    # Add data pattern if specified
    if data_pattern:
        inference_config['inference_config']['data_pattern'] = data_pattern
    
    # Add output directory
    inference_config['inference_config']['output_dir'] = output_dir
    
    # Integration parameters (for flow matching)
    inference_config['inference_config']['n_steps'] = n_steps
    inference_config['inference_config']['integration_method'] = integration_method
    
    # Visualization parameters
    inference_config['inference_config']['vis_n_steps'] = vis_n_steps
    inference_config['inference_config']['n_samples'] = n_samples
    inference_config['inference_config']['dpi'] = dpi
    
    # Evaluation parameters
    inference_config['inference_config']['batch_size'] = batch_size
    inference_config['inference_config']['metrics'] = metrics
    
    # Sampling parameters
    inference_config['inference_config']['sample_batch_size'] = sample_batch_size
    inference_config['inference_config']['save_samples'] = save_samples
    
    return inference_config


def main():
    parser = argparse.ArgumentParser(
        description='Create inference configuration from training configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        """
    )
    
    parser.add_argument(
        '--training_config',
        type=str,
        required=True,
        help='Path to training configuration YAML file'
    )
    
    parser.add_argument(
        '--output_config',
        type=str,
        default=None,
        help='Path for output inference config (default: configs/inference/{name}_inference.yaml)'
    )
    
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help='Path to model checkpoint relative to project root'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory for inference results (default: derived from training config)'
    )
    
    parser.add_argument(
        '--n_steps',
        type=int,
        default=50,
        help='Number of ODE integration steps (default: 50)'
    )
    
    parser.add_argument(
        '--integration_method',
        type=str,
        default='rk4',
        choices=['euler', 'midpoint', 'rk4'],
        help='Integration method for ODE solver (default: rk4)'
    )
    
    parser.add_argument(
        '--vis_n_steps',
        type=int,
        default=10,
        help='Number of steps shown in visualization (default: 10)'
    )
    
    parser.add_argument(
        '--n_samples',
        type=int,
        default=4,
        help='Number of samples to visualize (default: 4)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation (default: 32)'
    )
    
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=['mse', 'relative_l2'],
        help='Metrics to compute (default: mse relative_l2)'
    )
    
    parser.add_argument(
        '--sample_batch_size',
        type=int,
        default=4,
        help='Batch size for sample generation (default: 4)'
    )
    
    parser.add_argument(
        '--no_save_samples',
        action='store_true',
        help='Do not save generated samples'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=200,
        help='DPI for saved figures (default: 200)'
    )
    
    parser.add_argument(
        '--data_pattern',
        type=str,
        default=None,
        help='Custom data pattern for inference data (e.g., "data/static/poisson/*32*")'
    )
    
    args = parser.parse_args()
    
    # Get project root directory
    project_dir = os.getcwd()
    
    # Load training configuration
    if not os.path.exists(args.training_config):
        print(f"Error: Training config file not found: {args.training_config}")
        sys.exit(1)
    
    print(f"Loading training config from: {args.training_config}")
    training_config = load_config(args.training_config)
    
    # Create inference configuration
    inference_config = create_inference_config(
        training_config=training_config,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        n_steps=args.n_steps,
        integration_method=args.integration_method,
        vis_n_steps=args.vis_n_steps,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        metrics=args.metrics,
        sample_batch_size=args.sample_batch_size,
        save_samples=not args.no_save_samples,
        dpi=args.dpi,
        data_pattern=args.data_pattern,
        auto_find_checkpoint=True,
        project_dir=project_dir
    )
    
    # Determine output path
    if args.output_config is None:
        # Auto-generate output filename
        name = training_config.get('name', 'experiment')
        spatial_dim = training_config.get('spatial_dim', '')
        output_filename = f"{name}_{spatial_dim}_inference.yaml" if spatial_dim else f"{name}_inference.yaml"
        output_path = os.path.join('configs', 'inference', output_filename)
    else:
        output_path = args.output_config
    
    # Save inference configuration
    save_yaml(inference_config, output_path)
    
    print("\n" + "="*70)
    print("Inference Configuration Created Successfully")
    print("="*70)
    print(f"\nTo run inference:")
    print(f"  python tests/generic_test_inference.py --config {output_path}")
    print("\nRemember to:")
    print("  1. Update checkpoint_path in the config if needed")
    print("  2. Ensure the checkpoint file exists")
    print("  3. Verify data_dir points to the correct dataset")
    print("="*70)


if __name__ == '__main__':
    main()
