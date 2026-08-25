# EMA

Exponential moving average of model weights, updated once per optimizer step with a
warmup ramp. Validation and checkpointing run under the averaged weights — these are
the weights you evaluate and ship.

::: flowpde.trainers.ema
