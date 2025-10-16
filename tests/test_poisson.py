import torch
import os
import sys
import yaml
#sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from glob import glob
from torch.utils.data import Dataset, DataLoader

from src.models.mlp import MLP
from src.trainers.flow_matching import FlowMatchingTrainer
from src.datasets.poisson import PoissonDataset
from utils.utils import load_config, load_class




if __name__ == "__main__":

    # Project root directory
    project_dir = os.getcwd()

    # Upload config file
    config_path = os.path.join(project_dir, "configs", "poisson.yaml")
    assert os.path.exists(config_path), f"Config file not found at {config_path}"

    #Parse config and use it to set up everything
    config = load_config(config_path)
    print(config)
    print("Config loaded successfully.")

    # Set random seed for reproducibility
    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Dataset ---
    DatasetClass = load_class(config["dataset_config"]["class"])
    dataset = DatasetClass(glob(os.path.join(project_dir, config["data_dir"],f"*{config["spatial_dim"]}*"))[0])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["trainer_config"]["batch_size"], shuffle=True)
    print("Dataset loaded.")

    # --- 2. Model ---
    ModelClass = load_class(config["model_config"]["class"])
    model = ModelClass(input_dim=config["spatial_dim"] ** 2, hidden_dim=256)
    print("Model initialized.")

    # --- 3. Optimizer ---
    optimizer_cfg = config["optimizer_config"]
    OptimizerClass = load_class(optimizer_cfg["class_path"])
    optimizer = OptimizerClass(model.parameters(), **optimizer_cfg["init_args"])
    print("Optimizer set.")

    # --- 4. Scheduler (optional) ---
    SchedulerClass = load_class(config["scheduler_config"]["class"])
    scheduler = SchedulerClass(optimizer, T_0=10)
    print("Scheduler ready.")

    # --- 5. Trainer ---
    TrainerClass = load_class(config["trainer_config"]["class"])
    trainer = TrainerClass(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        device=device
    )
    print("Trainer created.")

    print("🚀 Starting training...")
    trainer.train(
        data_loader=dataloader,
        epochs=config['trainer_config']['epochs'],
        print_stats_interval=config['trainer_config']['print_stats_interval'],
        save_interval=config['trainer_config']['save_interval'],
        save_dir=os.path.join(project_dir, "checkpoints", f"{config['name']}_{config['spatial_dim']}"),
        visualize=False
    )

