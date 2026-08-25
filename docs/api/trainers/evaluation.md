# FlowEvaluator

Integrates the ODE, denormalizes predictions, and scores them against ground truth in
physical units. Evaluation noise uses a fixed seed, so epoch-to-epoch differences
reflect the model rather than the draw.

Set `ensemble_size > 1` to score the ensemble mean and report spread — the UQ hook.

::: flowpde.trainers.evaluation
