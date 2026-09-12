"""Reduced Astrobee dynamics with rematerialized derived pose steps."""

import jax
import jax.numpy as jnp
import jaxlie

from environments import astrobee_reduced as _base
from environments.astrobee_reduced import (
    CONTROL_DIM,
    OBS_DIM,
    Control,
    DynamicsParams,
    EnvParams,
    Observation,
    ReducedState,
    State,
    control_limits,
    cost,
    default_dynamics_params,
    default_env_params,
    evaluation_metrics,
    f,
    get_observation,
    get_reduced_state,
    lift_reduced_state,
    sample_initial_states,
    sample_reference_action,
    save_trajectory_gif,
)

ENV_NAME = "astrobee_reduced_derived_remat"


@jax.checkpoint
def f_q_joint(reduced: _base.ReducedState, dt: float):
    """Evaluate F = exp(-xi dt) Z exp(xi_ref dt) directly."""
    Z = _base._unflatten_pose(reduced[: _base.POSE_DIM])
    A_1 = jaxlie.SE3.exp(-reduced[7:13] * dt)
    A_2 = jaxlie.SE3.exp(reduced[13:19] * dt)
    return _base._flatten_pose(A_1 @ (Z @ A_2))


def f_joint(
    reduced: _base.ReducedState,
    u: _base.Control,
    u_ref: _base.Control,
    params: _base.DynamicsParams,
    dt: float,
) -> _base.ReducedState:
    return jnp.concat(
        [f_q_joint(reduced, dt), _base.f_xi_joint(reduced, u, u_ref, params, dt)]
    )


__all__ = [
    "CONTROL_DIM",
    "ENV_NAME",
    "OBS_DIM",
    "Control",
    "DynamicsParams",
    "EnvParams",
    "Observation",
    "ReducedState",
    "State",
    "control_limits",
    "cost",
    "default_dynamics_params",
    "default_env_params",
    "evaluation_metrics",
    "f",
    "f_joint",
    "get_observation",
    "get_reduced_state",
    "lift_reduced_state",
    "sample_initial_states",
    "sample_reference_action",
    "save_trajectory_gif",
]
