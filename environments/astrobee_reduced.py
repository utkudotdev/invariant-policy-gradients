"""SE(3)-reduced Astrobee environment (paper eq. 55)."""

import jax
import jax.numpy as jnp
import jaxlie
from jaxtyping import Float

from environments.astrobee import (
    CONTROL_DIM,
    Control,
    DynamicsParams,
    EnvParams,
    State,
    Twist,
    _flatten_pose,
    _unflatten_pose,
    control_limits,
    cost,
    default_dynamics_params,
    default_env_params,
    evaluation_metrics,
    f,
    f_q,
    f_xi,
    sample_initial_states,
    sample_reference_action,
    save_trajectory_gif,
    state_twist,
    unpack_twist,
)

ENV_NAME = "astrobee_reduced"

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


# --- reduced state (SE(3) quotient) and observation -------------------------

ReducedState = Float[jax.Array, "19"]
"""Reduced state p(s) = (q^-1 q^d, xi, xi^d) (paper eq. 55).
    [0:7]   pose error q^-1 q^d as (unit quaternion wxyz, translation xyz)
    [7:13]  body twist xi         as a `Twist`, i.e. (v, omega)
    [13:19] reference twist xi^d  as a `Twist`, i.e. (v^d, omega^d)
"""

POSE_DIM = 7
TWIST_DIM = 6
REDUCED_DIM = POSE_DIM + 2 * TWIST_DIM

Observation = Float[jax.Array, "24"]
"""What the policy network is fed: `get_observation(reduced_state)`.

Same content as a `ReducedState`, but with the pose error re-expressed as
(R, p) with R the 3x3 rotation matrix, flattened:
    [0:12]  pose error as (R.reshape(9), p)
    [12:18] body twist xi
    [18:24] reference twist xi^d
"""

OBS_POSE_DIM = 12
OBS_DIM = OBS_POSE_DIM + 2 * TWIST_DIM


def get_reduced_state(state: State, state_ref: State) -> ReducedState:
    """Reduce a state pair to p(s) = (q^-1 q^d, xi, xi^d) (paper eq. 55)."""
    return jnp.concat(
        [
            get_q_reduced_state(state, state_ref),
            get_xi_reduced_state(state, state_ref),
        ]
    )


def get_q_reduced_state(state: State, state_ref: State) -> Float[jax.Array, "7"]:
    q_err = state.q.inverse() @ state_ref.q
    return _flatten_pose(q_err)


def get_xi_reduced_state(state: State, state_ref: State) -> Float[jax.Array, "12"]:
    return jnp.concat([state_twist(state), state_twist(state_ref)])


def get_observation(reduced: ReducedState) -> Observation:
    """Map the reduced state to the network's input representation.

    The only change is to the pose block: the rotation is handed to the policy
    as a 3x3 matrix rather than a quaternion, because the quaternion double
    cover (q and -q are the same rotation) makes SO(3) -> R^4 discontinuous, so
    a network reading it has to learn to identify two far-apart inputs. The
    matrix embedding is continuous and double-cover-free.
    """
    q_err = _unflatten_pose(reduced[:POSE_DIM])
    return jnp.concat(
        [
            q_err.rotation().as_matrix().reshape(9),
            q_err.translation(),
            reduced[POSE_DIM:REDUCED_DIM],
        ]
    )


def lift_reduced_state(reduced: ReducedState) -> tuple[State, State]:
    """A representative of the orbit p^-1(s~), taken on the section q = identity."""
    q, q_ref = lift_q_reduced_state(reduced)
    xi, xi_ref = lift_xi_reduced_state(reduced)
    v, omega = unpack_twist(xi)
    v_ref, omega_ref = unpack_twist(xi_ref)

    return (
        State(q=q, omega=omega, v=v),
        State(q=q_ref, omega=omega_ref, v=v_ref),
    )


def lift_q_reduced_state(reduced: ReducedState) -> tuple[jaxlie.SE3, jaxlie.SE3]:
    return jaxlie.SE3.identity(), _unflatten_pose(reduced[:POSE_DIM])


def lift_xi_reduced_state(reduced: ReducedState) -> tuple[Twist, Twist]:
    return (
        reduced[POSE_DIM : POSE_DIM + TWIST_DIM],
        reduced[POSE_DIM + TWIST_DIM : REDUCED_DIM],
    )


def f_joint(
    reduced: ReducedState,
    u: Control,
    u_ref: Control,
    params: DynamicsParams,
    dt: float,
) -> ReducedState:
    """Step the quotient MDP forward one step."""
    pose_next = f_q_joint(reduced, dt)
    xi_next = f_xi_joint(reduced, u, u_ref, params, dt)

    return jnp.concat([pose_next, xi_next])


# there is no reduction here that changes anything from an autodiff perspective,
# so we will just let autodiff handle it in all cases.
def f_xi_joint(
    reduced: ReducedState,
    u: Control,
    u_ref: Control,
    params: DynamicsParams,
    dt: float,
) -> Float[jax.Array, "12"]:
    xi, xi_ref = lift_xi_reduced_state(reduced)
    xi_next = f_xi(xi, u, params, dt)
    xi_ref_next = f_xi(xi_ref, u_ref, params, dt)
    return jnp.concat([xi_next, xi_ref_next])


# on the other hand, we will often replace this function
def f_q_joint(reduced: ReducedState, dt: float) -> Float[jax.Array, "7"]:
    """Step the reduced pose-error block using the natural expression."""
    state, state_ref = lift_reduced_state(reduced)
    q_next = f_q(state, dt)
    q_ref_next = f_q(state_ref, dt)
    return _flatten_pose(q_next.inverse() @ q_ref_next)
