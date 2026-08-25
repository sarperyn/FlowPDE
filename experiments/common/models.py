"""Model and objective builders for experiments."""

from __future__ import annotations

from typing import Any, Dict

from flowpde.core.base_conditioner import ConcatConditioner, NullConditioner
from flowpde.flows import NeuralODEFlow
from flowpde.models import ConvNet, MLP, ResNet, UNet
from flowpde.objectives import FlowMatchingObjective, MaximumLikelihoodObjective


def build_conditioner(name: str):
    """Build a supported input conditioner."""
    if name == "null":
        return NullConditioner()
    if name == "concat":
        return ConcatConditioner(dim=1)
    raise ValueError(f"Unknown conditioner: {name}")


def build_unet_objective(
    model_config: Dict[str, Any],
    objective_config: Dict[str, Any],
    conditioner_name: str,
) -> FlowMatchingObjective:
    """Build UNet + NeuralODEFlow + FlowMatchingObjective."""
    conditioner = build_conditioner(conditioner_name)
    model = UNet(
        spatial_dim=model_config["spatial_dim"],
        spatial_size=model_config["spatial_size"],
        base_channels=model_config["base_channels"],
        solution_channels=model_config.get("solution_channels", 1),
        condition_channels=model_config.get("condition_channels", 2),
        max_channels=model_config.get("max_channels", 512),
        use_attention=model_config.get("use_attention", True),
        norm_type=model_config.get("norm_type", "group"),
        activation=model_config.get("activation", "silu"),
        return_spatial=False,
        conditioner=conditioner,
    )
    flow = build_neural_ode_flow(model, objective_config)
    return FlowMatchingObjective(
        flow=flow,
        path=objective_config.get("path", "linear"),
        time_sampler=objective_config.get("time_sampler", "uniform"),
        coupling=objective_config.get("coupling", "independent"),
        source=objective_config.get("source", "gaussian"),
        sigma=objective_config.get("sigma", 0.0),
        target_key="target",
        condition_key="input",
    )


def count_parameters(model) -> int:
    """Count trainable model parameters."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_model(
    backbone: str,
    model_config: Dict[str, Any],
    conditioner_name: str,
):
    """Build a supported experiment backbone."""
    conditioner = build_conditioner(conditioner_name)
    common = {
        "solution_channels": model_config.get("solution_channels", 1),
        "condition_channels": model_config.get("condition_channels", 1),
        "return_spatial": False,
        "conditioner": conditioner,
    }

    if backbone == "unet":
        return UNet(
            spatial_dim=model_config["spatial_dim"],
            spatial_size=model_config["spatial_size"],
            base_channels=model_config.get("base_channels", 32),
            max_channels=model_config.get("max_channels", 256),
            use_attention=model_config.get("use_attention", True),
            norm_type=model_config.get("norm_type", "group"),
            activation=model_config.get("activation", "silu"),
            **common,
        )
    if backbone == "convnet":
        return ConvNet(
            spatial_dim=model_config["spatial_dim"],
            spatial_size=model_config["spatial_size"],
            hidden_channels=model_config.get("hidden_channels", 64),
            num_blocks=model_config.get("num_blocks", 8),
            kernel_size=model_config.get("kernel_size", 3),
            norm_type=model_config.get("norm_type", "group"),
            activation=model_config.get("activation", "silu"),
            dropout=model_config.get("dropout", 0.0),
            **common,
        )
    if backbone == "resnet":
        return ResNet(
            spatial_dim=model_config["spatial_dim"],
            spatial_size=model_config["spatial_size"],
            base_channels=model_config.get("base_channels", 64),
            blocks_per_stage=model_config.get("blocks_per_stage", [2, 2, 2]),
            kernel_size=model_config.get("kernel_size", 3),
            norm_type=model_config.get("norm_type", "group"),
            activation=model_config.get("activation", "silu"),
            downsample=model_config.get("downsample", False),
            **common,
        )
    if backbone == "mlp":
        spatial_dim = int(model_config["spatial_dim"])
        spatial_size = int(model_config["spatial_size"])
        solution_channels = int(model_config.get("solution_channels", 1))
        condition_channels = int(model_config.get("condition_channels", 1))
        input_dim = solution_channels * (spatial_size ** spatial_dim)
        condition_dim = condition_channels * (spatial_size ** spatial_dim)
        return MLP(
            input_dim=input_dim,
            condition_dim=condition_dim,
            hidden_dim=model_config.get("hidden_dim", 256),
            num_layers=model_config.get("num_layers", 4),
            activation=model_config.get("activation", "silu"),
            dropout=model_config.get("dropout", 0.0),
            conditioner=conditioner,
        )
    raise ValueError(f"Unknown backbone: {backbone}")


def build_neural_ode_flow(model, objective_config: Dict[str, Any]) -> NeuralODEFlow:
    """Build a NeuralODEFlow with shared likelihood/sampling settings."""
    return NeuralODEFlow(
        model=model,
        base_distribution=objective_config.get("base_distribution", "gaussian"),
        trace_estimator=objective_config.get("trace_estimator", "hutchinson"),
        n_trace_samples=objective_config.get("n_trace_samples", 1),
        ode_method=objective_config.get("ode_method", "dopri5"),
        ode_n_steps=objective_config.get("ode_n_steps"),
        use_adjoint=objective_config.get("use_adjoint", False),
        ode_rtol=objective_config.get("ode_rtol", 1e-5),
        ode_atol=objective_config.get("ode_atol", 1e-7),
        target_key="target",
        condition_key="input",
    )


def build_objective(
    model_config: Dict[str, Any],
    objective_config: Dict[str, Any],
    conditioner_name: str,
    backbone: str = "unet",
) -> FlowMatchingObjective | MaximumLikelihoodObjective:
    """Build backbone + NeuralODEFlow + configured objective."""
    model = build_model(
        backbone=backbone,
        model_config=model_config,
        conditioner_name=conditioner_name,
    )
    flow = build_neural_ode_flow(model, objective_config)

    objective_name = objective_config.get("name", "flow_matching")
    if objective_name in {"flow_matching", "fm"}:
        return FlowMatchingObjective(
            flow=flow,
            path=objective_config.get("path", "linear"),
            time_sampler=objective_config.get("time_sampler", "uniform"),
            coupling=objective_config.get("coupling", "independent"),
            source=objective_config.get("source", "gaussian"),
            sigma=objective_config.get("sigma", 0.0),
            target_key="target",
            condition_key="input",
        )
    if objective_name in {"maximum_likelihood", "mle"}:
        return MaximumLikelihoodObjective(
            flow=flow,
            regularization=objective_config.get("regularization", 0.0),
            normalize_by_dim=objective_config.get("normalize_by_dim", True),
            target_key="target",
            condition_key="input",
        )
    raise ValueError(f"Unknown objective: {objective_name}")


def build_flow_matching_objective(
    model_config: Dict[str, Any],
    objective_config: Dict[str, Any],
    conditioner_name: str,
    backbone: str = "unet",
) -> FlowMatchingObjective:
    """Build backbone + NeuralODEFlow + FlowMatchingObjective."""
    model = build_model(
        backbone=backbone,
        model_config=model_config,
        conditioner_name=conditioner_name,
    )
    flow = build_neural_ode_flow(model, objective_config)
    return FlowMatchingObjective(
        flow=flow,
        path=objective_config.get("path", "linear"),
        time_sampler=objective_config.get("time_sampler", "uniform"),
        coupling=objective_config.get("coupling", "independent"),
        source=objective_config.get("source", "gaussian"),
        sigma=objective_config.get("sigma", 0.0),
        target_key="target",
        condition_key="input",
    )
