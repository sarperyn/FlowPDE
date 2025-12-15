import argparse

def get_args():
    """
    Parse command line arguments for configuring the experiment.

    This function combines command-line configuration with YAML-based configuration.
    Users can provide a base configuration file via `--config` and override specific 
    hyperparameters (like learning rate, batch size, epochs, etc.) directly from the CLI.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, 
                        help="Path to the YAML configuration file.")
    
    parser.add_argument('--device', type=str, default=None, 
                        help="Device to use for computation. Set to 'cuda:0' for GPU or 'cpu' for CPU.")

    parser.add_argument('--name', type=str, default=None, 
                        help="Identifier for the experiment, used to create subdirectories for results.")

    ## TODO: implement wandb logging later maybe end of project
    # parser.add_argument('--wandb', action='store_true', 
    #                     help="If set, enables logging with Weights & Biases (WandB) for experiment tracking.") # this is implemented yet

    parser.add_argument('--seed', type=int, default=None, 
                        help="Random seed for reproducibility of results.")

    parser.add_argument('--lr', type=float, default=None, 
                        help="Learning rate for the optimizer.")

    parser.add_argument('--bs', type=int, default=None, 
                        help="Batch size for training, determining the number of samples per gradient update.")

    parser.add_argument('--epoch', type=int, default=None, 
                        help="Number of epochs for training, the number of full dataset iterations.")

    parser.add_argument('--save_dir', type=str, default='results/', 
                        help="Directory where the results will be saved. Change for your experiment directory.")

    args = parser.parse_args()
    return args


def override_config(config, args):
    """
    Safely override YAML config with CLI args that match existing keys.
    Only updates keys that already exist in the config.
    """
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        if value is None or key == "config":
            continue
        _update_if_exists(config, key, value)
    return config


def _update_if_exists(d, key, value):
    """
    Recursively update an existing key in a nested dict.
    Does NOT create new keys. Returns True if updated, else False.
    """
    for k, v in d.items():
        if k == key:
            d[k] = value
            print(f"[INFO] Override: {key} → {value}")
            return True
        elif isinstance(v, dict) and _update_if_exists(v, key, value):
            return True
    return False  # not found, do nothing