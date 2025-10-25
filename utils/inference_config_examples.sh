#!/bin/bash
# Quick reference script for create_inference_config.py
# This file contains common usage patterns

echo "=========================================="
echo "Create Inference Config - Quick Reference"
echo "=========================================="
echo ""

echo "1. BASIC USAGE (auto-generated checkpoint path):"
echo "   python utils/create_inference_config.py --training_config configs/training/mlp_flow_poisson.yaml"
echo ""

echo "2. WITH SPECIFIC CHECKPOINT:"
echo "   python utils/create_inference_config.py \\"
echo "       --training_config configs/training/mlp_flow_poisson.yaml \\"
echo "       --checkpoint_path results/training/poisson_dirichlet_8/checkpoints/model_19999.pt"
echo ""

echo "3. CUSTOM PARAMETERS (high accuracy):"
echo "   python utils/create_inference_config.py \\"
echo "       --training_config configs/training/unet_flow_poisson.yaml \\"
echo "       --checkpoint_path results/training/poisson_dirichlet_32/checkpoints/model_499.pt \\"
echo "       --n_steps 100 \\"
echo "       --integration_method rk4 \\"
echo "       --batch_size 16"
echo ""

echo "4. FAST INFERENCE (for testing):"
echo "   python utils/create_inference_config.py \\"
echo "       --training_config configs/training/mlp_flow_poisson.yaml \\"
echo "       --n_steps 20 \\"
echo "       --integration_method euler \\"
echo "       --batch_size 64"
echo ""

echo "5. CUSTOM OUTPUT LOCATION:"
echo "   python utils/create_inference_config.py \\"
echo "       --training_config configs/training/mlp_flow_poisson.yaml \\"
echo "       --output_config configs/inference/my_experiment.yaml"
echo ""

echo "6. WITH TEST DATA:"
echo "   python utils/create_inference_config.py \\"
echo "       --training_config configs/training/mlp_flow_poisson.yaml \\"
echo "       --data_pattern 'data/static/poisson/*8*test*' \\"
echo "       --checkpoint_path results/training/poisson_dirichlet_8/checkpoints/model_19999.pt"
echo ""

echo "=========================================="
echo "After creating config, run inference with:"
echo "  python tests/generic_test_inference.py --config <generated_config.yaml>"
echo "=========================================="
