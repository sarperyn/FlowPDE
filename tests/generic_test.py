import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glob import glob
from utils.utils import load_config, load_class

"""
GENERIC TEST SCRIPT
For a given .yaml file it can run experiments
"""


if __name__ == "__main__":

    # Project root directory
    project_dir = os.getcwd()

    # Upload config file
    config_path = os.path.join(project_dir, "configs", "poisson.yaml")
    assert os.path.exists(config_path), f"Config file not found at {config_path}"

    # Parse config and use it to set up everything
    config = load_config(config_path)
    print(config)
    print("Config loaded successfully.")

    # Set random seed for reproducibility
    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    DatasetClass = load_class(config["dataset_config"]["class"])
    dataset = DatasetClass(glob(os.path.join(project_dir, config["data_dir"],f"*{config["spatial_dim"]}*"))[0])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["trainer_config"]["batch_size"], shuffle=True)
    print("Dataset loaded.")

    # Model
    ModelClass = load_class(config["model_config"]["class"])
    model = ModelClass(input_dim=config["spatial_dim"] ** 2, hidden_dim=256)
    print("Model initialized.")

    # Optimizer
    optimizer_cfg = config["optimizer_config"]
    OptimizerClass = load_class(optimizer_cfg["class_path"])
    optimizer = OptimizerClass(model.parameters(), **optimizer_cfg["init_args"])
    print("Optimizer set.")

    # Scheduler
    SchedulerClass = load_class(config["scheduler_config"]["class"])
    scheduler = SchedulerClass(optimizer, T_0=10)
    print("Scheduler ready.")

    # Trainer
    TrainerClass = load_class(config["trainer_config"]["class"])
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
        epochs=config['trainer_config']['epochs'],
        print_stats_interval=config['trainer_config']['print_stats_interval'],
        save_interval=config['trainer_config']['save_interval'],
        save_dir=os.path.join(project_dir, "checkpoints", f"{config['name']}_{config['spatial_dim']}"),
        visualize=False
    )

