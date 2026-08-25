# Experiment 01: Darcy Conditioning Ablation

This experiment compares two conditioning strategies for a UNet trained with
flow matching on the 2D Darcy forward problem.

Variants:

- `null`: `NullConditioner`, ignores `[kappa, source]`
- `concat`: `ConcatConditioner`, feeds `[x_t, kappa, source]` to the UNet

The primary metric is sampled relative L2 error in physical units.

