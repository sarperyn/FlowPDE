"""
Neural ODE flows with optional likelihood evaluation.

Continuous normalizing flows learn invertible transformations using neural ODEs.
Log-density evaluation uses exact or stochastic divergence estimates in the
instantaneous change-of-variables formula.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from flowpde.core.base_flow import BaseFlow
from flowpde.flows.components.sources import get_source
from flowpde.solvers.ode_solvers import evaluation_mode


class NeuralODELogProbVectorField(nn.Module):
    """
    Vector field wrapper that includes log probability computation.

    Augments state with log probability and computes trace of Jacobian
    for the instantaneous change of variables formula.
    """

    def __init__(
        self,
        model: nn.Module,
        condition: Tensor,
        trace_estimator: str = 'hutchinson',
        n_trace_samples: int = 1
    ):
        super().__init__()
        self.model = model
        self.condition = condition
        self.trace_estimator = trace_estimator
        if not isinstance(n_trace_samples, int) or n_trace_samples < 1:
            raise ValueError("n_trace_samples must be a positive integer")
        self.n_trace_samples = n_trace_samples
        self.register_buffer("_trace_noise", None, persistent=False)

    def forward(self, t: Tensor, state: Tensor) -> Tensor:
        r"""
        Compute augmented dynamics: $[dx/dt, d(\log p)/dt]$.

        Args:
            t: Current time (scalar)
            state: Augmented state $[x, \log p_x]$ with shapes:
                   x: (batch_size, dim)
                   $\log p_x$: (batch_size, 1)

        Returns:
            Augmented dynamics $[dx/dt, d(\log p_x)/dt]$
        """
        batch_size = state.shape[0]

        # Split augmented state; the log-density channel state[:, -1:] does not feed back into the dynamics
        x = state[:, :-1]  # (batch_size, dim)

        # Prepare time for model
        t_batch = t.expand(batch_size, 1)

        create_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            x = x.requires_grad_(True)

            # Compute velocity
            v = self.model(x, self.condition, t_batch).flatten(start_dim=1)
            if v.shape != x.shape:
                raise ValueError("Velocity shape must match the ODE state")

            # Compute trace of Jacobian
            trace = self._compute_trace(v, x, create_graph=create_graph)

        # Change of variables: $d(\log p)/dt = -\text{tr}(\partial v/\partial x)$
        dlogpx_dt = -trace.view(batch_size, 1)

        # Combine into augmented dynamics
        dstate_dt = torch.cat([v, dlogpx_dt], dim=1)

        return dstate_dt if create_graph else dstate_dt.detach()

    def _compute_trace(self, v: Tensor, x: Tensor, create_graph: bool = True) -> Tensor:
        r"""
        Compute trace of Jacobian $\partial v/\partial x$.

        Args:
            v: Velocity field (batch_size, dim)
            x: State (batch_size, dim)

        Returns:
            Trace (batch_size,)
        """
        if self.trace_estimator == 'exact':
            return self._exact_trace(v, x, create_graph)
        elif self.trace_estimator == 'hutchinson':
            return self._hutchinson_trace(v, x, create_graph)
        else:
            raise ValueError(f"Unknown trace estimator: {self.trace_estimator}")

    def _exact_trace(self, v: Tensor, x: Tensor, create_graph: bool = True) -> Tensor:
        """Compute exact trace by summing diagonal of Jacobian."""
        batch_size, dim = v.shape
        trace = torch.zeros(batch_size, device=v.device, dtype=v.dtype)

        if not v.requires_grad:
            return trace

        for i in range(dim):
            grad_outputs = torch.zeros_like(v)
            grad_outputs[:, i] = 1

            dvi_dx = torch.autograd.grad(
                v, x,
                grad_outputs=grad_outputs,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=True
            )[0]

            trace = trace + (dvi_dx[:, i] if dvi_dx is not None else v[:, i] * 0)

        return trace

    def _hutchinson_trace(self, v: Tensor, x: Tensor, create_graph: bool = True) -> Tensor:
        r"""
        Hutchinson trace estimator: $\mathbb{E}[\varepsilon^T (\partial v/\partial x) \varepsilon]$ where $\varepsilon \sim \mathcal{N}(0, I)$.

        Unbiased estimator that only requires one Jacobian-vector product.
        """
        batch_size, dim = v.shape

        if self._trace_noise is None:
            self._trace_noise = torch.randn(
                self.n_trace_samples, batch_size, dim,
                device=v.device, dtype=v.dtype,
            )
        epsilon = self._trace_noise
        if not v.requires_grad:
            return v.new_zeros(batch_size)

        traces = []
        for eps in epsilon:
            # Compute Jacobian-vector product
            jvp = torch.autograd.grad(
                v, x,
                grad_outputs=eps,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=True
            )[0]

            # Trace estimate: ε^T * jvp
            trace = v.sum(dim=1) * 0 if jvp is None else (eps * jvp).sum(dim=1)
            traces.append(trace)

        # Average over samples
        return torch.stack(traces).mean(dim=0)


class NeuralODEFlow(BaseFlow):
    r"""
    Conditional neural ODE flow with optional exact log probability.

    `NeuralODEFlow` represents the continuous-time flow/dynamics. Training
    objectives live in `flowpde.objectives`.

    $$\log p(x_1) = \log p(x_0) - \int_0^1 \text{tr}\left(\frac{\partial f}{\partial x}\right) dt$$

    Args:
        model: Neural network that computes velocity $v(x, \text{condition}, t)$
        base_distribution: Base distribution for sampling ('gaussian' or 'uniform')
        trace_estimator: Method for trace computation ('exact' or 'hutchinson')
        n_trace_samples: Number of samples for Hutchinson estimator
        target_key: Default batch key for target tensors (default: 'u')
        condition_key: Default batch key for condition tensors (default: 'f')

    References:
        - Grathwohl et al., "FFJORD: Free-form Continuous Dynamics for Scalable
          Reversible Generative Models", ICLR 2019
        - Chen et al., "Neural Ordinary Differential Equations", NeurIPS 2018
    """

    def __init__(
        self,
        model: nn.Module,
        base_distribution: str = 'gaussian',
        trace_estimator: str = 'hutchinson',
        n_trace_samples: int = 1,
        ode_method: str = 'dopri5',
        ode_n_steps: Optional[int] = None,
        use_adjoint: bool = False,
        ode_rtol: float = 1e-5,
        ode_atol: float = 1e-7,
        target_key: str = "u",
        condition_key: str = "f",
        source=None,
    ):
        super().__init__(
            model,
            target_key=target_key,
            condition_key=condition_key,
        )
        self.base_distribution = base_distribution
        self.source = get_source(source if source is not None else base_distribution)
        self.trace_estimator = trace_estimator
        self.n_trace_samples = n_trace_samples
        self.ode_method = ode_method
        self.ode_n_steps = ode_n_steps
        self.use_adjoint = use_adjoint
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol

        # The flattened target dimension, learned from the first training
        # batch. It is a buffer so it travels in state_dict(): a flow restored
        # from a checkpoint still knows what shape to sample, instead of
        # silently falling back to the condition's dimension.
        self.register_buffer("target_dim", torch.tensor(-1, dtype=torch.long))

        if trace_estimator not in ['exact', 'hutchinson']:
            raise ValueError(f"Unknown trace estimator: {trace_estimator}")

    def set_target_dim(self, dim: int) -> None:
        """Record the flattened target dimension (called by the objective)."""
        self.target_dim.fill_(int(dim))

    def _resolve_target_dim(
        self,
        target_shape: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> int:
        """Flattened size of the samples to draw."""
        if target_shape is not None:
            if isinstance(target_shape, int):
                return target_shape
            dim = 1
            for size in target_shape:
                dim *= size
            return dim

        dim = int(self.target_dim)
        if dim > 0:
            return dim

        raise RuntimeError(
            "This flow does not know its target dimension, so it cannot draw "
            "x_0. It is recorded from the first training batch and saved in "
            "the flow's state_dict, so a flow built fresh around loaded "
            "backbone weights has never seen it. Pass x_init=... or "
            "target_shape=... to sample()."
        )

    def _make_solver(self, method: Optional[str] = None, **solver_kwargs: Any):
        """
        Build the ODE solver used for sampling and transport.

        Settings fall back to the ones this flow was constructed with, so
        `ode_method`, `ode_rtol`, `ode_atol` and `use_adjoint` actually govern
        integration instead of being overridden by a call-site default.
        Unknown keyword arguments raise rather than being dropped.
        """
        from flowpde.solvers.ode_solvers import ODEFlowSolver

        solver_kwargs.setdefault("rtol", self.ode_rtol)
        solver_kwargs.setdefault("atol", self.ode_atol)
        solver_kwargs.setdefault("adjoint", self.use_adjoint)
        return ODEFlowSolver(
            model=self.model,
            method=method if method is not None else self.ode_method,
            **solver_kwargs,
        )

    def sample_base_distribution(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Sample from base distribution."""
        return self.source.sample(shape, device, dtype=self.model_dtype, generator=generator)

    def _integrate_logprob(
        self,
        x: Tensor,
        condition: Tensor,
        t_span: Tuple[float, float] = (1.0, 0.0),
        n_steps: Optional[int] = None,
        method: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Integrate the augmented (state, log-density) dynamics.

        This is the one path that cannot go through `ODEFlowSolver`: the
        augmented state carries log p alongside x, and the trace estimator
        needs higher-order derivatives.  torchdiffeq's adjoint mode is a poor
        fit for that graph, so direct autograd is always used here.

        Args:
            x: Initial state.
            condition: Conditioning tensor.
            t_span: Time interval; (1, 0) maps data to the base distribution.
            n_steps: Steps for fixed-step methods.
            method: Solver name; defaults to the flow's `ode_method`.

        Returns:
            Final state and the accumulated change in log density.
        """
        from torchdiffeq import odeint

        batch_size = x.shape[0]

        log_px = x.new_zeros(batch_size, 1)
        state = torch.cat([x, log_px], dim=1)

        vector_field = NeuralODELogProbVectorField(
            self.model, condition,
            trace_estimator=self.trace_estimator,
            n_trace_samples=self.n_trace_samples,
        )

        t_start, t_end = t_span
        t_eval = x.new_tensor([t_start, t_end])

        method = method if method is not None else self.ode_method
        n_steps = n_steps if n_steps is not None else self.ode_n_steps

        kwargs: Dict[str, Any] = {"method": method}
        if method in {"euler", "midpoint", "rk4", "explicit_adams",
                      "implicit_adams", "fixed_adams"}:
            if n_steps is None or int(n_steps) != n_steps or n_steps < 1:
                raise ValueError(
                    f"n_steps must be set when using fixed-step ODE method "
                    f"'{method}'."
                )
            kwargs["options"] = {"step_size": abs(t_end - t_start) / int(n_steps)}
        else:
            kwargs["rtol"] = self.ode_rtol
            kwargs["atol"] = self.ode_atol

        with evaluation_mode(self.model):
            trajectory = odeint(vector_field, state, t_eval, **kwargs)
        final_state = trajectory[-1]
        return final_state[:, :-1], final_state[:, -1]

    def sample(
        self,
        condition: Tensor,
        n_steps: Optional[int] = None,
        solver: Optional[str] = None,
        x_init: Optional[Tensor] = None,
        target_shape: Optional[Union[int, Tuple[int, ...]]] = None,
        return_trajectory: bool = False,
        no_grad: bool = True,
        generator: Optional[torch.Generator] = None,
        **solver_kwargs: Any
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Sample by integrating the learned velocity field from t=0 to t=1.

        Args:
            condition: Conditioning tensor (B, *).
            n_steps: Integration steps.  Defaults to the flow's
                `ode_n_steps`; required for fixed-step solvers.
            solver: ODE solver name.  Defaults to the flow's `ode_method`,
                so the constructor's choice is what actually runs.
            x_init: Optional initial noise.  When omitted, drawn from the
                flow's base distribution.
            target_shape: Shape of the samples excluding batch, or the
                flattened dimension.  Only needed when the flow has not
                recorded its target dimension and no `x_init` is given.
            return_trajectory: Also return the full integration trajectory.
            no_grad: Integrate under `torch.no_grad()` (default).
            **solver_kwargs: Forwarded to `ODEFlowSolver`
                (`rtol`, `atol`, `adjoint`, `method_options`).  Unknown names
                raise rather than being silently dropped.

        Returns:
            Samples (B, dim), or `(samples, trajectory)` when
            `return_trajectory` is set.
        """
        condition = condition.flatten(start_dim=1).to(device=self.model_device, dtype=self.model_dtype)
        batch_size = condition.shape[0]

        if x_init is None:
            dim = self._resolve_target_dim(target_shape)
            x_init = self.sample_base_distribution(
                (batch_size, dim), self.model_device, generator=generator
            )
        else:
            x_init = x_init.flatten(start_dim=1).to(self.model_device)

        ode_solver = self._make_solver(solver, **solver_kwargs)
        return ode_solver.sample(
            condition=condition,
            x_init=x_init,
            t_span=(0.0, 1.0),
            return_trajectory=return_trajectory,
            n_steps=n_steps if n_steps is not None else self.ode_n_steps,
            no_grad=no_grad,
        )

    def log_prob(
        self,
        x: Tensor,
        condition: Tensor,
        **kwargs: Any
    ) -> Tensor:
        """
        Compute log probability of data.

        Args:
            x: Data samples
            condition: Conditioning tensor

        Returns:
            Log probabilities (batch_size,)
        """
        x = x.flatten(start_dim=1).to(device=self.model_device, dtype=self.model_dtype)
        condition = condition.flatten(start_dim=1).to(device=self.model_device, dtype=self.model_dtype)
        unknown = set(kwargs) - {"n_steps", "solver", "method"}
        if unknown:
            raise TypeError(f"Unknown log_prob options: {sorted(unknown)}")

        # Transport data back to the base distribution, accumulating the
        # change in log density along the way.
        x_0, delta_logp = self._integrate_logprob(
            x, condition,
            t_span=(1.0, 0.0),
            n_steps=kwargs.get("n_steps"),
            method=kwargs.get("solver") or kwargs.get("method"),
        )

        log_p0 = self.base_log_prob(x_0)

        # Log probability at data
        log_px = log_p0 - delta_logp

        return log_px

    def base_log_prob(self, z: Tensor) -> Tensor:
        """
        Log density of the base distribution, matching
        `sample_base_distribution` exactly.
        """
        return self.source.log_prob(z)

    def forward_transform(
        self,
        x: Tensor,
        condition: Optional[Tensor] = None,
        n_steps: Optional[int] = None,
        solver: Optional[str] = None,
        no_grad: bool = True,
        **solver_kwargs: Any
    ) -> Tensor:
        """
        Forward transformation: data -> latent (backward ODE).

        Args:
            x: Data samples
            condition: Conditioning tensor

        Returns:
            Latent samples in base distribution
        """
        if condition is None:
            raise ValueError("NeuralODEFlow requires conditioning")

        x = x.flatten(start_dim=1).to(device=self.model_device, dtype=self.model_dtype)
        condition = condition.flatten(start_dim=1).to(device=self.model_device, dtype=self.model_dtype)

        ode_solver = self._make_solver(solver, **solver_kwargs)
        return ode_solver.sample(
            condition=condition,
            x_init=x,
            t_span=(1.0, 0.0),
            n_steps=n_steps if n_steps is not None else self.ode_n_steps,
            no_grad=no_grad,
        )

    def inverse_transform(
        self,
        z: Tensor,
        condition: Optional[Tensor] = None,
        **kwargs: Any
    ) -> Tensor:
        """
        Inverse transformation: latent -> data (forward ODE).

        Args:
            z: Latent samples from base distribution
            condition: Conditioning tensor

        Returns:
            Data samples
        """
        if condition is None:
            raise ValueError("NeuralODEFlow requires conditioning")
        return self.sample(condition=condition, x_init=z, **kwargs)

    def get_config(self) -> Dict:
        """Return configuration dictionary."""
        return {
            'flow_type': 'neural_ode',
            'base_distribution': self.base_distribution,
            'source': self.source.get_config(),
            'trace_estimator': self.trace_estimator,
            'n_trace_samples': self.n_trace_samples,
            'ode_method': self.ode_method,
            'ode_n_steps': self.ode_n_steps,
            'use_adjoint': self.use_adjoint,
            'ode_rtol': self.ode_rtol,
            'ode_atol': self.ode_atol,
            'target_key': self.target_key,
            'condition_key': self.condition_key,
            'model_type': self.model.__class__.__name__
        }

    def restore_config(self, config):
        for name in ("base_distribution", "trace_estimator", "n_trace_samples",
                     "ode_method", "ode_n_steps", "use_adjoint", "ode_rtol", "ode_atol",
                     "target_key", "condition_key"):
            if name in config:
                setattr(self, name, config[name])
        if "source" in config:
            self.source = get_source(config["source"])
