"""
Flow matching objective for neural ODE flows.

This module provides a single, configurable objective that encompasses:
- Standard Flow Matching (Lipman et al., 2023)
- Rectified Flow (Liu et al., 2023)
- OT-Conditional Flow Matching (Tong et al., 2023)

All variants are achieved through configuration, not inheritance.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flowpde.core.base_conditioner import NullConditioner
from flowpde.flows import NeuralODEFlow
from flowpde.flows.components import (
    BatchSource,
    Coupling,
    GaussianSource,
    MiniBatchOTCoupling,
    PathInterpolant,
    SourceDistribution,
    TimeSampler,
    UniformSampler,
    get_coupling,
    get_path,
    get_source,
    get_time_sampler,
)


class FlowMatchingObjective(nn.Module):
    """
    Flow-matching objective for `NeuralODEFlow`.

    This objective trains a neural ODE flow by supervised velocity regression
    along interpolation paths instead of maximum likelihood. Multiple flow
    matching variants are configured through modular components:

    - **Path**: How to interpolate between noise and data
    - **Time Sampler**: Distribution for sampling training times
    - **Coupling**: How to pair noise and data samples
    - **Source**: Where trajectories start (noise, or precomputed pairs)

    Standard Configurations:

    1. **Flow Matching** (default):
       ```python
       FlowMatchingObjective(flow, path='linear', time_sampler='uniform')
       ```

    2. **Rectified Flow**:
       ```python
       FlowMatchingObjective(flow, path='linear', time_sampler='logit_normal')
       ```

    3. **OT-Conditional Flow Matching**:
       ```python
       FlowMatchingObjective(flow, path='ot_conditional', sigma=0.01)
       ```

    Args:
        flow: Neural ODE flow whose model predicts velocity
            v(x_t, condition, t)
        path: Interpolation path ('linear', 'ot_conditional') or PathInterpolant
        time_sampler: Time distribution ('uniform', 'logit_normal') or TimeSampler
        coupling: Coupling strategy ('independent', 'minibatch_ot') or Coupling
        source: Source distribution ('gaussian', 'batch') or SourceDistribution.
            Use 'batch' (`BatchSource`) to train on precomputed (x_0, x_1)
            pairs, which is what reflow requires.
        sigma: Noise level for OT-conditional path (default: 0.0)
        target_key: Default batch key for target tensors (default: 'u')
        condition_key: Default batch key for condition tensors (default: 'f')

    References:
        - Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
        - Liu et al., "Flow Straight and Fast: Rectified Flow", ICLR 2023
        - Tong et al., "Conditional Flow Matching", NeurIPS 2023
    """

    def __init__(
        self,
        flow: NeuralODEFlow,
        path: Union[str, PathInterpolant] = "linear",
        time_sampler: Union[str, TimeSampler] = "uniform",
        coupling: Union[str, Coupling] = "independent",
        source: Optional[Union[str, SourceDistribution]] = None,
        sigma: float = 0.0,
        target_key: Optional[str] = None,
        condition_key: Optional[str] = None,
    ):
        super().__init__()
        self.flow = flow
        self.target_key = target_key or flow.target_key
        self.condition_key = condition_key or flow.condition_key

        # Initialize components
        # Pass sigma to OT path if needed
        if isinstance(path, str) and path in ['ot_conditional', 'conditional_optimal_transport', 'ot']:
            self.path = get_path(path, sigma=sigma)
        else:
            self.path = get_path(path)

        self.time_sampler = get_time_sampler(time_sampler)
        self.coupling = get_coupling(coupling)
        self.source = flow.source if source is None else get_source(source)
        self.sigma = getattr(self.path, "sigma", 0.0)
        self._configure_prior()
        if self.sigma > 0 and getattr(self.path, "schedule", None) == "bridge":
            if not (isinstance(self.time_sampler, UniformSampler)
                    and 0 < self.time_sampler.low < self.time_sampler.high < 1):
                raise ValueError("bridge training requires UniformSampler with 0 < low < high < 1; use smooth_bridge for full-interval training")

        # Store string names for config
        self._path_name = path if isinstance(path, str) else path.__class__.__name__
        self._time_sampler_name = time_sampler if isinstance(time_sampler, str) else time_sampler.__class__.__name__
        self._coupling_name = coupling if isinstance(coupling, str) else coupling.__class__.__name__
        self._source_name = type(self.source).__name__

    @property
    def model(self):
        return self.flow.model

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, source):
        self._source = get_source(source)
        self._source_name = type(self._source).__name__
        if hasattr(self, "sigma"):
            self._configure_prior()

    def _configure_prior(self):
        prior = self.source.fallback if isinstance(self.source, BatchSource) else self.source
        if self.sigma > 0 and getattr(self.path, "schedule", None) == "constant":
            if not isinstance(prior, GaussianSource):
                raise ValueError("Constant tube noise currently requires a Gaussian source")
            prior = GaussianSource((prior.std ** 2 + self.sigma ** 2) ** 0.5)
        self.flow.source = prior

    def sample_base_distribution(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        batch: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        Draw `x_0` from the configured source distribution.

        Args:
            shape: Shape of `x_0`, matching the target.
            device: Device to place the result on.
            batch: Training batch, forwarded so sources such as
                `BatchSource` can read precomputed values from it.
                Omitted at inference, where sources fall back to noise.
        """
        return self.source(shape, device, batch, dtype=self.flow.model_dtype)

    @property
    def model_device(self) -> torch.device:
        return self.flow.model_device

    def _source_defines_pairing(self, batch: Optional[Dict[str, Tensor]]) -> bool:
        """Whether the source already determines which x_0 goes with which x_1.

        Reflow pairs are meaningful only as pairs: the model is retrained on
        exactly the trajectories it generated. Reordering them with a coupling
        would break that correspondence.
        """
        source = self.source
        key = getattr(source, "key", None)
        return key is not None and batch is not None and key in batch

    def _couple(self, x_0, x_1, condition):
        if isinstance(self.coupling, MiniBatchOTCoupling) and not isinstance(
            getattr(self.model, "conditioner", None), NullConditioner
        ):
            return self.coupling.couple_conditioned(x_0, x_1, condition)
        return self.coupling(x_0, x_1)

    def compute_loss(
        self,
        batch: Dict[str, Tensor],
        target_key: Optional[str] = None,
        condition_key: Optional[str] = None,
        **kwargs: Any
    ) -> Tensor:
        """
        Compute flow matching loss.

        The loss minimizes the MSE between predicted and target velocities:

        $$\\mathcal{L} = \\mathbb{E}_{t, x_0, x_1}[\\|v_\\theta(x_t, f, t) - v_t\\|^2]$$

        where:
        - $x_t$ is the interpolated point on the path
        - $v_t$ is the target velocity (derivative of path)
        - $f$ is the conditioning information

        Args:
            batch: Dictionary containing target and condition tensors.
            target_key: Batch key for target data. Defaults to this flow's
                configured target key ('u' by default).
            condition_key: Batch key for conditioning data. Defaults to this
                flow's configured condition key ('f' by default).

        Returns:
            MSE loss tensor (scalar)
        """
        # Extract and prepare data
        x_1, condition = self.flow._extract_target_condition(
            batch,
            target_key=target_key or self.target_key,
            condition_key=condition_key or self.condition_key,
        )
        self.flow.set_target_dim(x_1.shape[1])
        batch_size = x_1.shape[0]

        # Draw x_0 from the source. Passing the batch lets BatchSource
        # return the precomputed x_0 that reflow depends on.
        x_0 = self.sample_base_distribution(x_1.shape, self.model_device, batch)

        # Apply coupling strategy. A source that carries its own pairing
        # already fixes which x_0 goes with which x_1, so re-coupling here
        # would destroy it.
        if not self._source_defines_pairing(batch):
            x_0, x_1 = self._couple(x_0, x_1, condition)

        # Sample time
        t = self.time_sampler(batch_size, self.model_device).to(x_1)

        # Compute path interpolation and target velocity
        x_t, v_target = self.path(x_0, x_1, t)

        # Predict velocity with model
        v_pred = self.model(x_t, condition, t).flatten(start_dim=1)
        if v_pred.shape != v_target.shape:
            raise ValueError(f"Velocity shape {v_pred.shape} != target {v_target.shape}")

        # MSE loss
        loss = F.mse_loss(v_pred, v_target)

        return loss

    def sample(
        self,
        condition: Tensor,
        n_steps: Optional[int] = None,
        solver: Optional[str] = None,
        x_init: Optional[Tensor] = None,
        target_shape: Optional[Union[int, Tuple[int, ...]]] = None,
        return_trajectory: bool = False,
        no_grad: bool = True,
        **solver_kwargs: Any
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Generate samples by solving the flow ODE.

        Integrates the learned velocity field from t=0 (noise) to t=1 (data):

        $$\\frac{dx}{dt} = v_\\theta(x_t, f, t), \\quad x_0 \\sim \\mathcal{N}(0, I)$$

        Sampling delegates to `NeuralODEFlow.sample`, whose prior is configured
        by this objective. Constant tube noise broadens a Gaussian prior to
        match the training path at t=0 and produces smoothed target data.

        Args:
            condition: Conditioning tensor (B, *)
            n_steps: Number of integration steps.  Defaults to the flow's
                `ode_n_steps`; required for fixed-step solvers.
            solver: ODE solver ('euler', 'midpoint', 'rk4', 'dopri5').
                Defaults to the flow's `ode_method`.
            x_init: Optional initial noise (default: drawn from `source`)
            target_shape: Shape of generated targets excluding batch, or a
                flattened target dimension.  Only needed when the flow has
                not recorded its target dimension and no `x_init` is given.
            return_trajectory: If True, return full trajectory
            no_grad: Integrate under `torch.no_grad()` (default).  Pass False
                to differentiate through sampling.
            **solver_kwargs: Additional solver arguments (`rtol`, `atol`,
                `adjoint`, `method_options`).  Unknown names raise.

        Returns:
            Generated samples (B, dim)
            If return_trajectory: (samples, trajectory) where trajectory is (n_steps+1, B, dim)
        """
        return self.flow.sample(
            condition=condition,
            n_steps=n_steps,
            solver=solver,
            x_init=x_init,
            target_shape=target_shape,
            return_trajectory=return_trajectory,
            no_grad=no_grad,
            **solver_kwargs,
        )

    def estimate_straightness(
        self,
        batch: Dict[str, Tensor],
        n_time_points: int = 10,
        mode: str = "trajectory",
        n_steps: int = 50,
        solver: str = "euler",
        target_key: Optional[str] = None,
        condition_key: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Measure how straight the learned transport paths are.

        Straightness follows Liu et al. (2023): a flow is straight when its
        velocity along a trajectory equals the chord connecting the endpoints,

        $$S = \\int_0^1 \\mathbb{E}\\left[\\lVert (Z_1 - Z_0)
              - v_\\theta(Z_t, t) \\rVert^2\\right] dt,$$

        so $S = 0$ exactly when every trajectory is a straight line
        traversed at constant velocity — which is what makes few-step Euler
        sampling accurate, and what reflow is meant to improve.

        Two modes are available:

        - `'trajectory'` (default): integrate the learned ODE and measure
          deviation along the model's **own** trajectories.  This is the
          quantity that predicts few-step sampling quality.
        - `'interpolant'`: measure deviation along the training interpolant
          between sampled `(x_0, x_1)` pairs.  Cheaper (no ODE solve) and it
          reports how far the learned marginal velocity sits from the
          conditional target, but it does *not* describe the sampling paths.

        Args:
            batch: Batch with target and condition tensors.
            n_time_points: Number of time points at which velocity is probed.
            mode: `'trajectory'` or `'interpolant'`.
            n_steps: ODE steps used to build trajectories (`'trajectory'`).
            solver: ODE solver used to build trajectories (`'trajectory'`).
            target_key: Batch key for target data.
            condition_key: Batch key for conditioning data.

        Returns:
            Dictionary with:

            - `'straightness'`: the integral above (0 = perfectly straight).
            - `'normalized_straightness'`: divided by the mean squared chord
                length, making it dimensionless and comparable across datasets
                and normalization choices.
            - `'chord_norm'`: mean chord length, for reference.
        """
        if mode not in {"trajectory", "interpolant"}:
            raise ValueError(
                f"mode must be 'trajectory' or 'interpolant', got '{mode}'"
            )

        was_training = self.training
        self.eval()

        x_1, condition = self.flow._extract_target_condition(
            batch,
            target_key=target_key or self.target_key,
            condition_key=condition_key or self.condition_key,
        )
        batch_size = x_1.shape[0]

        try:
            with torch.no_grad():
                if mode == "trajectory":
                    x_0 = self.flow.sample_base_distribution(x_1.shape, self.model_device)
                    _, trajectory = self.sample(
                        condition=condition,
                        n_steps=n_steps,
                        solver=solver,
                        x_init=x_0,
                        return_trajectory=True,
                    )
                    # trajectory: (n_steps + 1, B, dim)
                    z_0, z_1 = trajectory[0], trajectory[-1]
                    chord = z_1 - z_0

                    # Probe the whole trajectory, endpoints included: the
                    # straightness integral in Liu et al. runs over the
                    # closed interval. (The 'interpolant' mode below uses
                    # [0.01, 0.99] instead, so the two modes are not
                    # numerically comparable -- they are different
                    # estimators, as the docstring says.)
                    indices = torch.linspace(
                        0, trajectory.shape[0] - 1, n_time_points
                    ).round().long()
                    time_grid = torch.linspace(
                        0.0, 1.0, trajectory.shape[0], device=self.model_device
                    )

                    deviations = []
                    for index in indices:
                        z_t = trajectory[index]
                        t = time_grid[index].expand(batch_size, 1)
                        v_pred = self.model(z_t, condition, t.to(z_t)).flatten(start_dim=1)
                        deviations.append((v_pred - chord).pow(2).sum(dim=1))
                else:
                    x_0 = self.sample_base_distribution(x_1.shape, self.model_device, batch)
                    if not self._source_defines_pairing(batch):
                        x_0, x_1 = self._couple(x_0, x_1, condition)
                    chord = x_1 - x_0

                    time_points = torch.linspace(
                        0.01, 0.99, n_time_points, device=self.model_device
                    )
                    deviations = []
                    for t_val in time_points:
                        t = t_val.expand(batch_size, 1)
                        x_t, target_velocity = self.path(x_0, x_1, t)
                        v_pred = self.model(x_t, condition, t).flatten(start_dim=1)
                        deviations.append((v_pred - target_velocity).pow(2).sum(dim=1))

                # Mean over time (the integral) and over the batch.
                straightness = torch.stack(deviations, dim=0).mean().item()
                chord_sq = chord.pow(2).sum(dim=1).mean()
                normalized = (straightness / chord_sq.clamp(min=1e-12)).item()
                chord_norm = chord.norm(dim=1).mean().item()
        finally:
            if was_training:
                self.train()

        return {
            "straightness": straightness,
            "normalized_straightness": normalized,
            "chord_norm": chord_norm,
        }

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization."""
        return {
            'objective': 'flow_matching',
            'flow': self.flow.get_config(),
            'path': self._path_name,
            'time_sampler': self._time_sampler_name,
            'coupling': self._coupling_name,
            'source': self.source.get_config(),
            'sigma': self.sigma,
            'components': {
                'path': self.path.get_config(),
                'time_sampler': self.time_sampler.get_config(),
                'coupling': self.coupling.get_config(),
                'source': self.source.get_config(),
            },
            'target_key': self.target_key,
            'condition_key': self.condition_key,
            'model_type': self.model.__class__.__name__,
        }

    def restore_config(self, config):
        self.flow.restore_config(config.get("flow", {}))
        components = config.get("components")
        if components:
            self.path = get_path(components["path"])
            self.time_sampler = get_time_sampler(components["time_sampler"])
            self.coupling = get_coupling(components["coupling"])
            self.source = get_source(components["source"])
            self.sigma = getattr(self.path, "sigma", 0.0)
            self._path_name = type(self.path).__name__
            self._time_sampler_name = type(self.time_sampler).__name__
            self._coupling_name = type(self.coupling).__name__
            self._source_name = type(self.source).__name__
        self.target_key = config["target_key"]
        self.condition_key = config["condition_key"]
        self._configure_prior()

    def __repr__(self) -> str:
        return (
            f"FlowMatchingObjective(\n"
            f"  path={self._path_name},\n"
            f"  time_sampler={self._time_sampler_name},\n"
            f"  coupling={self._coupling_name},\n"
            f"  source={self.source!r},\n"
            f"  sigma={self.sigma}\n"
            f")"
        )


def create_flow_matching(
    flow: NeuralODEFlow,
    variant: str = "standard",
    **kwargs: Any
) -> FlowMatchingObjective:
    """
    Create a flow matching objective with preset configurations.

    Args:
        flow: Neural ODE flow
        variant: Preset name:
            - 'standard': Standard flow matching (linear, uniform)
            - 'rectified': Rectified flow (linear, logit-normal)
            - 'ot_cfm': OT-Conditional FM (ot_conditional, uniform)
        **kwargs: Override any default parameters

    Returns:
        Configured flow matching objective
    """
    presets = {
        'standard': {
            'path': 'linear',
            'time_sampler': 'uniform',
            'coupling': 'independent',
            'sigma': 0.0,
        },
        'rectified': {
            'path': 'linear',
            'time_sampler': 'logit_normal',
            'coupling': 'independent',
            'sigma': 0.0,
        },
        'ot_cfm': {
            'path': 'ot_conditional',
            'time_sampler': 'uniform',
            'coupling': 'independent',
            'sigma': 0.0,
        },
        'ot_cfm_coupled': {
            'path': 'ot_conditional',
            'time_sampler': 'uniform',
            'coupling': 'minibatch_ot',
            'sigma': 0.0,
        },
    }

    if variant not in presets:
        raise ValueError(f"Unknown variant: '{variant}'. Available: {list(presets.keys())}")

    config = presets[variant].copy()
    config.update(kwargs)

    return FlowMatchingObjective(flow, **config)
