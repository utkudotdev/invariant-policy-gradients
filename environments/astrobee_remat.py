"""Unreduced Astrobee dynamics with rematerialized joint steps."""

import jax

from environments.astrobee import (
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
from environments.astrobee import (
    f_joint as _f_joint,
)

ENV_NAME = "astrobee_remat"
f_joint = jax.checkpoint(_f_joint)

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
