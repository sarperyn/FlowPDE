# Flow Matching

Flow matching is the primary training objective in FlowPDE. This page covers the theory
and how it maps onto the code.

## Background

Given a source distribution \(p_0\) (Gaussian noise) and a target distribution \(p_1\)
(PDE solutions), flow matching learns a time-dependent velocity field \(v_\theta\) that
transports samples from \(p_0\) to \(p_1\):

\[
\frac{dx_t}{dt} = v_\theta(x_t, f, t), \quad t \in [0, 1]
\]

where \(f\) is the conditioning variable — the PDE source term, coefficient field, or
initial condition.

## Training Objective

The loss regresses the model's velocity against the ground-truth conditional velocity:

\[
\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, x_1} \left[ \| v_\theta(x_t, f, t) - v_t(x_t \mid x_0, x_1) \|^2 \right]
\]

where \(x_t\) interpolates between noise \(x_0\) and data \(x_1\) along a chosen
**path**.

!!! warning "The loss has an irreducible floor"
    Many \((x_0, x_1)\) pairs give the same \(x_t\), so the model regresses a conditional
    *average* and the loss never approaches zero. Do not use `train_loss` for model
    selection — select on `FlowEvaluator` output.

## Path Interpolants

A path is a pair of functions, `interpolate()` and `velocity()`.

### Linear Path (default)

\[
x_t = (1 - t) x_0 + t x_1, \qquad v_t = x_1 - x_0
\]

### OT-Conditional Path

Adds a noise bridge with variance vanishing at both endpoints:

\[
x_t = t x_1 + (1 - t) x_0 + \sigma \sqrt{t(1-t)}\, \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)
\]

!!! danger "Invariant"
    A path's `velocity()` must be the exact time derivative of its `interpolate()`. If
    they disagree, training silently regresses a target that does not match the
    interpolation the model is shown. `tests/test_components.py` checks this by finite
    differences — keep that test passing when adding a path.

## Time Sampling

How \(t\) is sampled during training affects convergence:

| Sampler | Distribution | Use Case |
|---------|-------------|----------|
| `uniform` | \(t \sim U(0, 1)\) | General purpose |
| `logit_normal` | \(t = \sigma(z),\ z \sim \mathcal{N}(\mu, \sigma^2)\) | Weights the middle of the path |
| `beta` | \(t \sim \mathrm{Beta}(\alpha, \beta)\) | Flexible concentration |

`TruncatedSampler` is a **wrapper**, not a registry name — it clamps another sampler
away from exactly 0 and 1, where the velocity target can be numerically awkward. Pass
it as an instance:

```python
from flowpde.flows.components.time_samplers import TruncatedSampler, UniformSampler

objective = FlowMatchingObjective(
    flow,
    time_sampler=TruncatedSampler(UniformSampler(), low=1e-5, high=1 - 1e-5),
)
```

This is the "string or instance" rule the component getters follow throughout: anywhere
a name is accepted, a constructed object is too.

## Couplings

How noise and data are paired within a batch:

- **Independent** — random pairing (standard)
- **Mini-batch OT** — optimal transport within the mini-batch via
  `scipy.optimize.linear_sum_assignment`, reducing crossing trajectories

## Sources

The source distribution is a first-class component because reflow depends on it:

- **`GaussianSource`** — draws fresh \(x_0 \sim \mathcal{N}(0, I)\) (default)
- **`BatchSource`** — consumes the \(x_0\) already present in the batch, which is what
  makes reflow correct

## Rectified Flow and Reflow

These are two different things, and the distinction matters.

**The objective.** `create_flow_matching(flow, variant='rectified')` sets a linear path
with logit-normal time sampling. Note that 1-rectified flow is mathematically identical
to standard flow matching with a linear path and independent coupling — *the objective
alone straightens nothing*. (The logit-normal sampler is from Esser et al. 2024/SD3;
Liu et al. use uniform.)

**Reflow** is the procedure that does the straightening: generate \((z, \mathrm{ODE}(z))\)
pairs with the current model, retrain on those pairs, repeat.

```python
from flowpde.flows import BatchSource
from flowpde.trainers import generate_reflow_pairs, reflow

pairs = generate_reflow_pairs(objective, loader, n_steps=100)   # (z, ODE(z))
objective.source = BatchSource()                               # consume that exact z

# ...or run the whole loop:
reflow(objective, loader,
       optimizer_factory=lambda p: torch.optim.Adam(p, lr=1e-4),
       num_iterations=2, epochs_per_iteration=50)
```

!!! danger "Reflow pairs must be preserved"
    Reflow training must use the **same** \(z\) that produced each generated target. The
    pairing is deterministic and induced by the model; resampling noise independently
    decouples the pairs and silently reduces reflow to training on the model's own
    samples.

    `BatchSource` is `strict=True` by default so a missing key raises rather than
    silently resampling, and `compute_loss` skips the coupling when the source already
    defines the pairing — mini-batch OT would otherwise reorder the pairs and destroy
    them.

## Measuring Straightness

`estimate_straightness` implements the Liu et al. definition — **deviation from the
chord**, not the spread of velocity norms. A field that turns at constant speed must
not score as straight.

It takes a **batch**, not a DataLoader:

```python
batch = next(iter(loader))

s = objective.estimate_straightness(batch, mode="trajectory")   # the model's own ODE paths
s = objective.estimate_straightness(batch, mode="interpolant")  # the training interpolant

s["straightness"]              # 0 = perfectly straight
s["normalized_straightness"]   # dimensionless, comparable across datasets
```

## Sampling

At inference, integrate the learned ODE from noise to solution:

```python
samples = flow.sample(condition=f, n_steps=50, solver="dopri5")
```

Adaptive solvers (`dopri5`, `dopri8`) adjust step size automatically and ignore
`n_steps`; fixed-step solvers (`euler`, `rk4`) use it directly, which is what you vary
when reporting accuracy against solver-step count.

## A Note on Uncertainty

For the Poisson and Burgers **forward** maps, \(p(u \mid f)\) is a Dirac — the solution
is deterministic. Sampler spread there is model error, not physical uncertainty. Put
the UQ story on inverse problems, where the posterior is genuinely non-degenerate.
