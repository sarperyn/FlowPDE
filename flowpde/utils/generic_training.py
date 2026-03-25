"""
DEPRECATED — generic_training.py
==================================
This script was an early proof-of-concept that drove training entirely from
a YAML config file using dynamic class loading (``load_class``).  It has been
superseded by the per-benchmark training scripts in ``scripts/``:

    scripts/train_poisson_2d_flowmatching.py
    scripts/train_burgers_1d_flowmatching.py
    scripts/train_burgers_1d_rectified_flow.py
    scripts/train_darcy_2d_flowmatching.py

This file is kept for reference and will be removed in a future version.
Do NOT add new functionality here.
"""

import torch
import os
import sys
from glob import glob

# Add project root for script-level imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from flowpde.utils import load_config, load_class, get_args, override_config

"""
GENERIC TEST SCRIPT
For a given .yaml file it can run experiments
"""


if __name__ == "__main__":

    # Project root directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Get config
    args = get_args()

    # Upload config file
    config_path = os.path.abspath(os.path.join(project_dir, args.config.lstrip("/"))) #both /config/poisson.yaml and config/poisson.yaml work
    assert os.path.exists(config_path), f"Config file not found at {config_path}"

    # Parse config and use it to set up everything
    config = load_config(config_path)
    #print(config)
    print("Config loaded successfully.")

    # Override config with command line arguments
    config = override_config(config, args)
    print("Config overridden with command line arguments.")

    # Create output directories
    output_dir = os.path.join(project_dir, config.get("training_config", {}).get("output_dir"), f"{config.get('name')}_{config.get('spatial_dim','')}")
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)
    print(f"Output directory: {output_dir}")

    # Set random seed for reproducibility
    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    DatasetClass = load_class(config["dataset_config"]["class"])
    data_pattern = os.path.join(project_dir, config.get("data_dir", "data"), f"*{config.get('spatial_dim', '')}*train*")
    matches = glob(data_pattern)
    if not matches:
        raise FileNotFoundError(f"No data files found matching: {data_pattern}")
    dataset = DatasetClass(matches[0])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["training_config"].get("batch_size", 1), shuffle=True)
    print("Dataset loaded.")

    # Model
    model_cfg = config["model_config"]
    ModelClass = load_class(model_cfg["class"])
    print("Model config:", model_cfg)
    model = ModelClass(**model_cfg.get("init_args", {}))
    print("Model initialized.")

    # Optimizer
    optimizer_cfg = config.get("optimizer_config", {})
    OptimizerClass = load_class(optimizer_cfg.get("class_path"))
    print("Optimizer init args:", optimizer_cfg.get("init_args", {}))
    optimizer = OptimizerClass(model.parameters(), **optimizer_cfg.get("init_args", {}))
    print("Optimizer set.")

    # Scheduler
    SchedulerClass = load_class(config["scheduler_config"]["class"])
    scheduler = SchedulerClass(optimizer, T_0=10)
    print("Scheduler ready.")

    # Trainer
    TrainerClass = load_class(config["training_config"]["class"])
    trainer = TrainerClass(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        device=device
    )
    print("Trainer created.")
    # Start traning
    print("Starting training...")
    trainer.train(
        data_loader=dataloader,
        epochs=config['training_config'].get('epoch', config['training_config'].get('epochs', 1)),
        print_stats_interval=config['training_config'].get('print_stats_interval', 10),
        save_interval=config['training_config'].get('save_interval', 1000),
        save_dir=os.path.join(project_dir, "results", "training", f"{config.get('name','exp')}_{config.get('spatial_dim','')}", "checkpoints") ,
    )

