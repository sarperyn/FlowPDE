# Trainer

Training loop for any object exposing `compute_loss(batch)` and `.model`, with EMA,
checkpointing, LR scheduling, and validation-based model selection.

::: flowpde.trainers.trainer
