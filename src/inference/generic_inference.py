import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glob import glob
from utils.utils import find_latest_checkpoint, load_config, load_class
from utils.args_utils import get_args, override_config
from src.visualization.utils import visualize_flow_evolution
from src.visualization.inference import (
    sample_flow_matching,
    compute_prediction_error,
)

"""
GENERIC INFERENCE SCRIPT
For a given .yaml config and checkpoint, it can run inference and evaluation
"""


def run_flow_matching_inference(model, dataloader, device, config, output_dir):
    """Run inference for flow matching models."""
    print("\n" + "="*70)
    print("Running Flow Matching Inference")
    print("="*70)
    
    inference_config = config.get("inference_config", {})
    n_steps = inference_config.get("n_steps", 50)
    integration_method = inference_config.get("integration_method", "midpoint")
    
    # 1. Visualize flow evolution
    print("\n1. Generating flow evolution visualization...")
    vis_path = os.path.join(output_dir, f"flow_evolution_{integration_method}.png")
    visualize_flow_evolution(
        model=model,
        dataloader=dataloader,
        fig_path=vis_path,
        n_steps=inference_config.get("vis_n_steps", 10),
        n_samples=inference_config.get("n_samples", 4),
        device=device,
        integration_method=integration_method,
        show_ground_truth=True,
        dpi=inference_config.get("dpi", 200)
    )
    print(f"Saved to: {vis_path}")
    
    # 2. Compute prediction errors
    print("\n2. Computing prediction errors...")
    for metric in inference_config.get("metrics", ["mse", "relative_l2"]):
        errors = compute_prediction_error(
            model=model,
            dataloader=dataloader,
            device=device,
            n_steps=n_steps,
            integration_method=integration_method,
            metric=metric
        )
        print(f"{metric.upper()}: {errors[f'mean_{metric}']:.6f}")
    
    # 3. Generate sample batch
    print("\n3. Generating sample predictions...")
    batch = next(iter(dataloader))
    if isinstance(batch, dict):
        f_batch = batch['f'][:inference_config.get("sample_batch_size", 4)]
    else:
        _, f_batch = batch
        f_batch = f_batch[:inference_config.get("sample_batch_size", 4)]
    
    samples = sample_flow_matching(
        model=model,
        condition=f_batch,
        n_steps=n_steps,
        device=device,
        integration_method=integration_method
    )
    print(f"Generated {samples.shape[0]} samples")
    print(f"Shape: {samples.shape}")
    
    # 4. Save sample outputs if requested
    if inference_config.get("save_samples", False):
        samples_path = os.path.join(output_dir, "samples.pt")
        torch.save({
            'samples': samples.cpu(),
            'conditions': f_batch.cpu()
        }, samples_path)
        print(f"   ✓ Saved samples to: {samples_path}")
    
    print("\n" + "="*70)
    print(" Flow Matching Inference Complete")
    print("="*70)


if __name__ == "__main__":

    # Project root directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Get config
    args = get_args()

    # Upload config file
    config_path = os.path.abspath(os.path.join(project_dir, args.config.lstrip("/")))
    assert os.path.exists(config_path), f"Config file not found at {config_path}"

    # Parse config
    config = load_config(config_path)
    print("Config loaded successfully.")

    # Override config with command line arguments
    config = override_config(config, args)
    print("Config overridden with command line arguments.")

    # Set random seed
    seed = config.get("seed")
    torch.manual_seed(seed)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset - load validation or test data
    print("\nLoading dataset...")
    DatasetClass = load_class(config["dataset_config"]["class"])
    data_pattern = config.get("inference_config", {}).get("data_pattern", None)
    
    if data_pattern is None:
        # Use validation data by default
        data_pattern = os.path.join(project_dir, config.get("data_dir", "data"), f"*{config.get('spatial_dim', '')}*train*")
    else:
        data_pattern = os.path.join(project_dir, data_pattern)
    
    matches = glob(data_pattern)
    if not matches:
        raise FileNotFoundError(f"No data files found matching: {data_pattern}")
    
    dataset = DatasetClass(matches[0])
    batch_size = config.get("inference_config", {}).get("batch_size", 32)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Using file: {os.path.basename(matches[0])}")

    # Load model
    print("\nLoading model...")
    model_cfg = config["model_config"]
    ModelClass = load_class(model_cfg["class"])
    print("Model config:", model_cfg)
    model = ModelClass(**model_cfg.get("init_args", {}))

    # Load checkpoint
    checkpoint_path = config.get("inference_config", {}).get("checkpoint_path", None)
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(project_dir, config)

    if checkpoint_path is None:
        raise FileNotFoundError("No checkpoint found. Specify 'checkpoint_path' in inference_config.")
    
    checkpoint_path = os.path.abspath(os.path.join(project_dir, checkpoint_path.lstrip("/")))
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    print(f"Checkpoint loaded (epoch: {checkpoint.get('epoch', 'unknown')})")

    # Create output directory
    output_dir = config.get("inference_config", {}).get("output_dir", "results/inference")
    output_dir = os.path.join(project_dir, output_dir, f"{config.get('name','exp')}_{config.get('spatial_dim','')}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Determine trainer type and run appropriate inference
    trainer_class = config["training_config"]["class"]
    
    if "flow_matching" in trainer_class.lower() or "flowmatching" in trainer_class.lower():
        run_flow_matching_inference(model, dataloader, device, config, output_dir)
    # elif "nf" in trainer_class.lower() or "normalizing" in trainer_class.lower():
    #     run_normalizing_flow_inference(model, dataloader, device, config, output_dir)
    else:
        print(f"\nWarning: Unknown trainer type '{trainer_class}'")
        print("Attempting flow matching inference as default...")
        try:
            run_flow_matching_inference(model, dataloader, device, config, output_dir)
        except Exception as e:
            print(f"Error running inference: {e}")
            print("Please specify inference_config in your yaml file.")
