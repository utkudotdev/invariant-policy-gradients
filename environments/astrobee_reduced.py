"""SE(3)-reduced Astrobee environment (paper eq. 55)."""

from functools import partial

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
# so we will just let autodiff handle it
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


# but for q, we *can* hopefully do better because we already know q = I, so we'll write a
# custom_vjp. arguably jax should be able to figure this out (and actually does a reasonable
# job with rematerialization enabled), but for this file we'll just write it ourselves.
@partial(jax.custom_vjp, nondiff_argnums=(1,))
def f_q_joint(reduced: ReducedState, dt: float) -> Float[jax.Array, "7"]:
    """Step the pose-error block of the quotient state.

    On the section q = I the two poses are q = I and q^d = Z, so this is

        F = exp(xi dt)^-1 Z exp(xi^d dt) = A_1 Z A_2

    with A_1 = exp(d_1), d_1 = -xi dt and A_2 = exp(d_2), d_2 = xi^d dt. That
    three-factor form is what the analytic VJP below differentiates.
    """
    state, state_ref = lift_reduced_state(reduced)
    q_next = f_q(state, dt)
    q_ref_next = f_q(state_ref, dt)
    q_err = q_next.inverse() @ q_ref_next
    return _flatten_pose(q_err)


# Everything below works in *ambient* coordinates: a pose is the 7 numbers
# (quaternion wxyz, translation) and a cotangent is another 7 numbers under the
# plain Euclidean pairing.
#
# The full analytical version of `f_q_joint` is of the form F = A_1 Z A_2
# where A_i = exp(...). So we just need two primitives:
# 1. `_compose_vjp`, the VJP of an SE(3) product
# 2. `_exp_vjp`, the VJP of A = exp(d) (in ambient space)
# There is also `_compose_left_vjp`, which is basically a smaller part of
# `_compose_vjp` that only computes the VJP of (A B) with respect to A.

Quaternion = Float[jax.Array, "4"]
"""Hamilton quaternion, scalar-first (wxyz), matching jaxlie."""


def _quat_mul(a: Quaternion, b: Quaternion) -> Quaternion:
    """Hamilton product, valid for non-unit inputs -- cotangents are not unit."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return jnp.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _quat_conj(q: Quaternion) -> Quaternion:
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])


def _pure(v: Float[jax.Array, "3"]) -> Quaternion:
    """The pure quaternion (0, v)."""
    return jnp.concatenate([jnp.zeros(1), v])


def _compose_vjp(
    A: jaxlie.SE3,
    B: jaxlie.SE3,
    q_bar: Quaternion,
    t_bar: Float[jax.Array, "3"],
) -> tuple[
    tuple[Quaternion, Float[jax.Array, "3"]], tuple[Quaternion, Float[jax.Array, "3"]]
]:
    """VJP of C = A B, in ambient coordinates. Returns (A-bar, B-bar).

    In components q_C = q_A q_B and t_C = q_A . t_B + t_A. The first is bilinear
    and the translation is affine in t_B, so those transpose to more quaternion
    products. The exception is q_A, which appears twice in the rotation of t_B:
    that node is quadratic, and its VJP is the -2 t_bar_C q_A t_B term.
    """
    q_A, q_B = A.rotation(), B.rotation()
    q_A_conj = _quat_conj(q_A.wxyz)

    q_bar_A = _quat_mul(q_bar, _quat_conj(q_B.wxyz)) - 2.0 * _quat_mul(
        _quat_mul(_pure(t_bar), q_A.wxyz), _pure(B.translation())
    )
    t_bar_A = t_bar
    q_bar_B = _quat_mul(q_A_conj, q_bar)
    t_bar_B = q_A.inverse().apply(t_bar)

    return (q_bar_A, t_bar_A), (q_bar_B, t_bar_B)


def _compose_left_vjp(
    A: jaxlie.SE3,
    B: jaxlie.SE3,
    q_bar: Quaternion,
    t_bar: Float[jax.Array, "3"],
) -> tuple[Quaternion, Float[jax.Array, "3"]]:
    """VJP of C = A B with respect to A only.

    Keeping this separate from `_compose_vjp` matters when B is the reference
    trajectory: its cotangent is deliberately not propagated during BPTT, so
    computing the right-factor pullback would be wasted work.
    """
    q_A = A.rotation().wxyz
    q_bar_A = _quat_mul(q_bar, _quat_conj(B.rotation().wxyz)) - 2.0 * _quat_mul(
        _quat_mul(_pure(t_bar), q_A), _pure(B.translation())
    )
    return q_bar_A, t_bar


def _exp_vjp(A: jaxlie.SE3, q_bar: Quaternion, t_bar: Float[jax.Array, "3"]) -> Twist:
    """Pull an ambient cotangent on A = exp(d) back to a cotangent on d in R^6.

    Factors as d_bar = J_r(d)^T M(A)^T A_bar, where M(A) maps a body-frame
    velocity to the ambient velocity of (q, t) -- from qdot = q (0, omega) / 2
    and tdot = q . v, so M^T is the (rotate by q^-1, half the vector part of
    q^* q_bar) pair below.

    J_r is the derivative of exp itself and cannot be avoided, but jaxlie hands
    it to us: `A.jlog()` is exactly J_r(log A)^-1 (verified numerically against
    the series form), so J_r^T m is a 6x6 solve against its transpose -- closed
    form, with the small-angle branches already handled.
    """
    q = A.rotation()
    rho_bar = q.inverse().apply(t_bar)
    phi_bar = 0.5 * _quat_mul(_quat_conj(q.wxyz), q_bar)[1:]

    return jnp.linalg.solve(A.jlog().T, jnp.concatenate([rho_bar, phi_bar]))


def f_q_joint_fwd(reduced: ReducedState, dt: float):
    # The residual is just the input. Storing the three factors A_1, Z, A_2
    # instead would save two exp() calls here at a cost of 21 floats per step
    # against 19 -- and the input is already live as the scan carry, so this
    # way the rule adds nothing to the memory the rollout was keeping anyway.
    return f_q_joint(reduced, dt), reduced


def f_q_joint_bwd(dt: float, res: ReducedState, g: Float[jax.Array, "7"]):
    reduced = res
    q_bar_F, t_bar_F = g[:4], g[4:]

    xi, xi_ref = lift_xi_reduced_state(reduced)
    A_1 = jaxlie.SE3.exp(-xi * dt)
    A_2 = jaxlie.SE3.exp(xi_ref * dt)
    Z = _unflatten_pose(reduced[:POSE_DIM])
    P = A_1 @ Z

    q_bar_P, t_bar_P = _compose_left_vjp(P, A_2, q_bar_F, t_bar_F)
    (q_bar_A1, t_bar_A1), (q_bar_Z, t_bar_Z) = _compose_vjp(A_1, Z, q_bar_P, t_bar_P)

    # d_1 = -xi dt, so the chain rule ends in a scaling. The xi_ref cotangent
    # is not needed by BPTT: the reference trajectory is exogenous.
    xi_bar = -dt * _exp_vjp(A_1, q_bar_A1, t_bar_A1)
    xi_ref_bar = jnp.full((TWIST_DIM,), jnp.inf)

    return (jnp.concatenate([q_bar_Z, t_bar_Z, xi_bar, xi_ref_bar]),)


f_q_joint.defvjp(f_q_joint_fwd, f_q_joint_bwd)
