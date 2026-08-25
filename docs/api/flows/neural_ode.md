# NeuralODEFlow

`NeuralODEFlow` is the continuous-time dynamics object: it owns the velocity-field
model, generates samples by integrating the ODE, and computes exact log-likelihood
via the Jacobian trace.

It does **not** contain training logic — that lives in [objectives](../objectives/flow_matching.md).

::: flowpde.flows.neural_ode
