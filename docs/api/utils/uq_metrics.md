# Uncertainty Quantification Metrics

A conditional flow produces a *distribution* $p(u \mid f)$, not a point prediction.
Relative $L^2$ of the ensemble mean says nothing about whether that distribution is any
good — these metrics ask whether it is **calibrated**.

Draw an ensemble with `FlowEvaluator(..., ensemble_size=K)` or by calling
[`flow.sample()`](../flows/neural_ode.md) $K$ times, then score it here.

!!! tip "Score in physical units"
    Denormalize both samples and ground truth before scoring, exactly as with the
    deterministic [metrics](metrics.md).

::: flowpde.utils.uq_metrics
