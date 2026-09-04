"""Environment-agnostic types shared by the training loop and every environment.

An environment module (e.g. `environments.particle`) is expected to provide:

    State, Control, ReducedState, Observation                -- types
    DynamicsParams, EnvParams                                -- types
    ENV_NAME                                                 -- output/display name
    REDUCED_DIM, OBS_DIM, CONTROL_DIM                        -- sizing
    default_dynamics_params(), default_env_params()          -- experiment defaults
    f(state, u, dynamics_params, dt)                         -- dynamics
    f_joint(reduced, u, u_ref, dynamics_params, dt)          -- reduced (joint) dynamics
    get_reduced_state(state, state_ref)                      -- state pair -> reduced state
    get_observation(reduced)                                 -- reduced state -> observation
    lift_reduced_state(reduced)                              -- reduced state -> state pair
    sample_reference_action(key, env_params)                 -- u^d ~ rho
    sample_initial_states(key, batch, env_params)            -- (state0, ref_state0)
    cost(rollout, env_params)                                -- scalar tracking cost
    evaluation_metrics(rollout)                              -- named error series
    save_trajectory_gif(rollout, dt, path, ...)              -- visualization
"""

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Float

StateT = TypeVar("StateT")
ControlT = TypeVar("ControlT")

Policy = Callable[[jax.Array, jax.Array], jax.Array]
"""Maps (observation, reference action) to a control.

The observation is `get_observation(reduced_state)`, not the reduced state that
`f_joint` propagates: environments are free to hand the network a larger, more
network-friendly re-encoding of the state they actually carry.
"""


@jax.tree_util.register_dataclass
@dataclass
class Rollout(Generic[StateT, ControlT]):
    """A single trajectory: T + 1 states and the T controls that produced them."""

    s: Float[StateT, " steps+1"]
    s_ref: Float[StateT, " steps+1"]
    us: Float[ControlT, " steps"]
    us_ref: Float[ControlT, " steps"]


@dataclass(frozen=True)
class EvaluationMetric:
    """A per-step metric exposed by an environment for generic reporting."""

    name: str
    values: Float[jax.Array, " steps+1"]
    unit: str = ""


def batched_pytree_get_first(x):
    return jax.tree.map(lambda y: y[0], x)


def batched_pytree_prepend(x, y):
    return jax.tree.map(lambda a, b: jnp.concatenate([a[None], b], axis=0), x, y)
