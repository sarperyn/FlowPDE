# API Reference

Reference documentation for the FlowPDE modules.

## The central separation: flow vs. objective

This is the most important structural fact about the library.

- **[`NeuralODEFlow`](flows/neural_ode.md)** is the continuous-time *dynamics*: it owns
  the model, sampling by ODE integration, and exact log-likelihood.
- **[Objectives](objectives/flow_matching.md)** are *how you train that flow*.

Both objectives wrap the same flow. Training logic does not belong on the flow, and
dynamics do not belong on an objective.

```python
flow = NeuralODEFlow(model, target_key="target", condition_key="input")
objective = FlowMatchingObjective(flow, path="linear", time_sampler="uniform")
trainer = Trainer(objective, optimizer, ...)
```

## Modules

| Module | Description |
|--------|-------------|
| [**Core**](core/base_flow.md) | Abstract bases (`BaseFlow`, `BaseSolver`, `BaseConditioner`) and the config system |
| [**Flows**](flows/neural_ode.md) | `NeuralODEFlow` and its pluggable components |
| [**Objectives**](objectives/flow_matching.md) | Flow matching and maximum likelihood |
| [**Models**](models/mlp.md) | Velocity-field backbones: MLP, UNet, ConvNet, ResNet |
| [**Solvers**](solvers/ode_solvers.md) | ODE integration for inference |
| [**Trainers**](trainers/trainer.md) | `Trainer`, EMA, `FlowEvaluator`, reflow |
| [**Datasets**](datasets/exponax.md) | Exponax PDE data generation and normalization |
| [**Utils**](utils/metrics.md) | Metrics and visualization |
