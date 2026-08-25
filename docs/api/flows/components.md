# Flow Components

Modular components that `FlowMatchingObjective` composes, so flow-matching variants
are *configuration* rather than subclasses.

Every getter (`get_path`, `get_time_sampler`, `get_coupling`, `get_source`) accepts
either a registry name or an already-constructed instance.

## Paths

The interpolation between noise and data.

!!! warning "Invariant"
    A path's `velocity()` must be the exact time derivative of its `interpolate()`.
    If they disagree, training silently regresses a target that does not match the
    interpolation the model is shown.

::: flowpde.flows.components.paths

## Time Samplers

::: flowpde.flows.components.time_samplers

## Couplings

::: flowpde.flows.components.couplings

## Sources

Where trajectories start. `BatchSource` is what makes reflow correct — it consumes
the exact noise `z` that produced each generated target instead of resampling.

::: flowpde.flows.components.sources
