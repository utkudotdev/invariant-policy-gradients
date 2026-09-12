"""Reduced Astrobee dynamics with rematerialized natural pose steps."""

import jax
import jax.numpy as jnp

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

ENV_NAME = "astrobee_reduced_remat"


@jax.checkpoint
def f_q_joint(reduced: _base.ReducedState, dt: float):
    state, state_ref = _base.lift_reduced_state(reduced)
    q_next = _base.f_q(state, dt)
    q_ref_next = _base.f_q(state_ref, dt)
    return _base._flatten_pose(q_next.inverse() @ q_ref_next)


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
