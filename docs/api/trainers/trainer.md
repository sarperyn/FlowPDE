# Trainer

Training loop for any object exposing `compute_loss(batch)` and `.model`, with EMA,
checkpointing, LR scheduling, and validation-based model selection.

Training checkpoint format 2 stores the target dimension, component and solver
settings, raw/EMA weights, optimizer/scheduler state, AMP scaler, selection history,
and PyTorch RNG state. `load_checkpoint` resumes epoch numbering and model
selection. Its `epochs` argument specifies additional epochs after a resume.
Older checkpoints remain loadable; metadata absent from an older file must still
be supplied by the caller. Worker-specific data-loader RNG state is not captured.

::: flowpde.trainers.trainer
