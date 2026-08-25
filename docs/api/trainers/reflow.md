# Reflow

The procedure that straightens trajectories: generate `(z, ODE(z))` pairs with the
current model, retrain on those pairs, repeat.

!!! danger "Correctness requirement"
    Reflow training must use the *same* `z` that produced each generated target. The
    pairing is deterministic and induced by the model; resampling noise independently
    decouples the pairs and silently reduces reflow to training on the model's own
    samples.

::: flowpde.trainers.reflow
