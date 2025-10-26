# Create Inference Config Utility

## Overview

`create_inference_config.py` is a utility script that automatically generates inference configuration files from training configuration files. This eliminates the need to manually create inference configs and ensures consistency between training and inference settings.

## Features

- ✅ Automatically extracts model, dataset, and experiment settings from training configs
- ✅ **Automatically finds the latest checkpoint** in `results/training/{name}_{spatial_dim}/checkpoints/`
- ✅ Generates properly formatted inference YAML files
- ✅ Supports customization of all inference parameters
- ✅ Auto-generates checkpoint paths based on experiment name and spatial dimensions
- ✅ Provides sensible defaults for ODE integration and visualization parameters
- ✅ Supports Flow Matching models (easily extensible to other model types)

## Checkpoint Path Structure

The script expects checkpoints to be stored in the following structure:
```
results/training/{experiment_name}_{spatial_dim}/checkpoints/model_{epoch}.pt
```

For example:
- `results/training/poisson_dirichlet_8/checkpoints/model_19999.pt`
- `results/training/unet_poisson_dirichlet_32/checkpoints/model_499.pt`

The script will **automatically find and use the latest checkpoint** (highest epoch number) if no checkpoint path is explicitly provided.

## Quick Start

### Basic Usage

Create an inference config from a training config:

```bash
python utils/create_inference_config.py --training_config configs/training/mlp_flow_poisson.yaml
```

This will create: `configs/inference/mlp_poisson_dirichlet_8_inference.yaml`

### With Checkpoint Path

Specify the exact checkpoint to use:

```bash
python utils/create_inference_config.py \
    --training_config configs/training/mlp_flow_poisson.yaml \
    --checkpoint_path checkpoints/mlp_poisson_dirichlet_8/model_19999.pt
```

### Custom Parameters

Customize inference parameters:

```bash
python utils/create_inference_config.py \
    --training_config configs/training/unet_flow_poisson.yaml \
    --n_steps 100 \
    --integration_method midpoint \
    --batch_size 16 \
    --vis_n_steps 20
```

### Custom Output Location

Specify where to save the inference config:

```bash
python utils/create_inference_config.py \
    --training_config configs/training/mlp_flow_poisson.yaml \
    --output_config configs/inference/my_custom_config.yaml
```

## Command Line Arguments

### Required Arguments

- `--training_config`: Path to training configuration YAML file

### Optional Arguments

#### Output Configuration
- `--output_config`: Path for output inference config (default: auto-generated based on experiment name)
- `--output_dir`: Directory for inference results (default: `results/inference`)

#### Model & Checkpoint
- `--checkpoint_path`: Path to model checkpoint relative to project root (default: auto-generated)

#### ODE Integration Parameters
- `--n_steps`: Number of ODE integration steps (default: 50)
- `--integration_method`: Integration method - choices: `euler`, `midpoint`, `rk4` (default: rk4)

#### Visualization Parameters
- `--vis_n_steps`: Number of steps shown in visualization (default: 10)
- `--n_samples`: Number of samples to visualize (default: 4)
- `--dpi`: DPI for saved figures (default: 200)

#### Evaluation Parameters
- `--batch_size`: Batch size for evaluation (default: 32)
- `--metrics`: Metrics to compute (default: mse relative_l2)
  - Available: `mse`, `mae`, `relative_l2`

#### Sampling Parameters
- `--sample_batch_size`: Batch size for sample generation (default: 4)
- `--no_save_samples`: Flag to disable saving generated samples

#### Data Parameters
- `--data_pattern`: Custom data pattern for inference data (e.g., `"data/static/poisson/*32*test*"`)

## Examples

### Example 1: Basic Inference Config

```bash
python utils/create_inference_config.py \
    --training_config configs/training/mlp_flow_poisson.yaml
```

**Generated config:**
- Name: `mlp_poisson_dirichlet_8_inference.yaml`
- Checkpoint: Auto-detects latest in `results/training/mlp_poisson_dirichlet_8/checkpoints/`
- Integration: RK4 with 50 steps

### Example 2: High-Accuracy Inference

```bash
python utils/create_inference_config.py \
    --training_config configs/training/unet_flow_poisson.yaml \
    --checkpoint_path results/training/poisson_dirichlet_32/checkpoints/model_499.pt \
    --n_steps 200 \
    --integration_method rk4 \
    --batch_size 8
```

**Use case:** When you need very accurate results and have compute budget.

### Example 3: Fast Inference for Testing

```bash
python utils/create_inference_config.py \
    --training_config configs/training/mlp_flow_poisson.yaml \
    --checkpoint_path results/training/poisson_dirichlet_8/checkpoints/model_19999.pt \
    --n_steps 20 \
    --integration_method euler \
    --batch_size 64
```

**Use case:** Quick testing or when speed is more important than accuracy.

### Example 4: Custom Metrics and Visualization

```bash
python utils/create_inference_config.py \
    --training_config configs/training/unet_flow_poisson.yaml \
    --metrics mse mae relative_l2 \
    --vis_n_steps 20 \
    --n_samples 8 \
    --dpi 300
```

**Use case:** Comprehensive evaluation with high-quality visualizations.

### Example 5: Test Set Evaluation

```bash
python utils/create_inference_config.py \
    --training_config configs/training/mlp_flow_poisson.yaml \
    --checkpoint_path results/training/poisson_dirichlet_8/checkpoints/model_19999.pt \
    --data_pattern "data/static/poisson/*8*test*" \
    --no_save_samples
```

**Use case:** Evaluate on test set without saving samples.

## Integration Methods Comparison

| Method | Accuracy | Speed | Recommended Use |
|--------|----------|-------|-----------------|
| `euler` | Low | Fast | Quick testing |
| `midpoint` | Medium | Medium | Good balance |
| `rk4` | High | Slow | Final evaluation, paper results |

## Generated Config Structure

The script generates an inference config with the following structure:

```yaml
name: experiment_name
seed: 42
data_dir: data/static/poisson
spatial_dim: 32

model_config:
  class: src.models.mlp.MLP
  init_args:
    input_dim: 1024
    hidden_dim: 256

training_config:
  class: src.trainers.flow_matching.FlowMatchingTrainer

dataset_config:
  class: src.datasets.poisson.PoissonDataset

inference_config:
  checkpoint_path: checkpoints/experiment_32/model_999.pt
  output_dir: results/inference
  
  # ODE Integration
  n_steps: 50
  integration_method: rk4
  
  # Visualization
  vis_n_steps: 10
  n_samples: 4
  dpi: 200
  
  # Evaluation
  batch_size: 32
  metrics:
    - mse
    - relative_l2
  
  # Sampling
  sample_batch_size: 4
  save_samples: true
```

## Running Inference

After creating the inference config, run inference using:

```bash
python tests/generic_test_inference.py --config configs/inference/your_inference_config.yaml
```

## Tips & Best Practices

1. **Start with defaults**: The default parameters (RK4, 50 steps) provide good accuracy
2. **Check checkpoint exists**: Verify the checkpoint path points to an actual file
3. **Match spatial dimensions**: Ensure the checkpoint was trained with the same `spatial_dim`
4. **Use RK4 for papers**: For publication-quality results, use RK4 with 50-100 steps
5. **Test with Euler first**: Use Euler for quick sanity checks during development
6. **Batch size**: Adjust based on GPU memory (larger = faster, but needs more memory)

## Troubleshooting

### "Training config file not found"
- Check the path to your training config
- Use absolute or relative path from project root

### "Checkpoint not found" (when running inference)
- Update `checkpoint_path` in the generated inference config
- Verify the checkpoint file exists at the specified location

### Generated config has wrong parameters
- Use command line arguments to override defaults
- Edit the generated YAML file directly if needed

## Extending the Script

To add support for new model types or trainer types:

1. Open `utils/create_inference_config.py`
2. Modify the `create_inference_config()` function
3. Add conditional logic based on `training_config['training_config']['class']`

Example:

```python
# In create_inference_config()
trainer_class = training_config.get('training_config', {}).get('class', '')

if 'normalizing_flow' in trainer_class.lower():
    # Add NF-specific inference parameters
    inference_config['inference_config']['n_samples'] = 1000
    inference_config['inference_config']['compute_log_prob'] = True
```

## See Also

- `tests/generic_test_inference.py` - Main inference script
- `docs/INFERENCE_GUIDE.md` - Comprehensive inference guide
- `examples/flow_matching_inference_example.py` - Inference examples
