"""Environment-agnostic types shared by the training loop and every environment.

An environment module (e.g. `environments.particle`) is expected to provide:

    State, Control, Observation, DynamicsParams, EnvParams   -- types
    OBS_DIM, CONTROL_DIM                                     -- policy sizing
    f(state, u, dynamics_params, dt)                         -- dynamics
    f_joint(obs, u, u_ref, dynamics_params, dt)              -- reduced (joint) dynamics
    get_observation(state, state_ref)                        -- state pair -> observation
    lift_observation(obs)                                    -- observation -> state pair
    sample_reference_action(key, env_params)                 -- u^d ~ rho
    sample_initial_states(key, batch, env_params)            -- (state0, ref_state0)
    cost(rollout, env_params)                                -- scalar tracking cost
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
"""Maps (observation, reference action) to a control."""


@jax.tree_util.register_dataclass
@dataclass
class Rollout(Generic[StateT, ControlT]):
    """A single trajectory: T + 1 states and the T controls that produced them."""

    s: Float[StateT, " steps+1"]
    s_ref: Float[StateT, " steps+1"]
    us: Float[ControlT, " steps"]
    us_ref: Float[ControlT, " steps"]


def batched_pytree_get_first(x):
    return jax.tree.map(lambda y: y[0], x)


def batched_pytree_prepend(x, y):
    return jax.tree.map(lambda a, b: jnp.concatenate([a[None], b], axis=0), x, y)
