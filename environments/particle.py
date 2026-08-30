"""Point-particle environment (arXiv:2409.11238, "Particle").

State is (r, v) in R^2 x R^2; the control is a force in R^2; the reduced
state is the state error (r - r^d, v - v^d), which the policy also sees
verbatim as its observation.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jaxtyping
import matplotlib.pyplot as plt
from jaxtyping import Float
from matplotlib import animation

from environments.base import Rollout


@jax.tree_util.register_dataclass
@dataclass
class State:
    r: Float[jax.Array, "2"]
    v: Float[jax.Array, "2"]


@jax.tree_util.register_dataclass
@dataclass
class DynamicsParams:
    m: float


Control = Float[jax.Array, "2"]

ReducedState = Float[jax.Array, "4"]
"""Reduced state (r - r^d, v - v^d) (paper eq. 44)."""

Observation = Float[jax.Array, "4"]
"""What the policy network is fed. Identical to the reduced state here."""

CONTROL_DIM = 2

REDUCED_DIM = 4

OBS_DIM = 4


@jax.tree_util.register_dataclass
@dataclass
class CostCoeffs:
    """Coefficients for the tracking cost (paper eq. 8/12)."""

    c_r: float  # linear position-error penalty
    a_r: float  # sharpness of the near-zero tracking bonus
    c_v: float  # velocity-error penalty
    c_u: float  # effort penalty (deviation from reference action)


@jax.tree_util.register_dataclass
@dataclass
class EnvParams:
    cost_coeffs: CostCoeffs
    sigma: float
    pos_std: float
    vel_std: float


def f(state: State, u: Control, params: DynamicsParams, dt: float) -> State:
    """Discretized point-particle dynamics.

    state = (r, v) with r, v in R^2. Applies control force u over one step:
        r_{t+1} = r_t + v_t dt
        v_{t+1} = v_t + (1/m) u_t dt
    """
    r, v = state.r, state.v
    r_next = r + v * dt
    v_next = v + (u / params.m) * dt
    return State(r=r_next, v=v_next)


def sample_reference_action(key: jaxtyping.Key, params: EnvParams) -> Control:
    """Draw a reference action u^d ~ N(0, sigma^2 I).

    The paper (arXiv:2409.11238, eq. 10-11c) models the reference action
    distribution rho for the Particle as an isotropic Gaussian N(0, Sigma).
    """
    return params.sigma * jax.random.normal(key, (CONTROL_DIM,))


def get_reduced_state(state: State, state_ref: State) -> ReducedState:
    return jnp.concatenate([state.r - state_ref.r, state.v - state_ref.v])


def get_observation(reduced: ReducedState) -> Observation:
    """Map the reduced state to the network's input representation.

    The reduced state is already a plain error vector in R^4, with nothing like
    the SE(3) pose of the Astrobee to re-parameterize, so this is the identity.
    It exists so every environment presents the same interface to the training
    loop.
    """
    return reduced


def lift_reduced_state(reduced: ReducedState) -> tuple[State, State]:
    return State(r=reduced[:2], v=reduced[2:]), State(
        r=jnp.zeros(2), v=jnp.zeros(2)
    )


# TODO: in theory i think we have linear output tangents so this could be done using custom_jvp and
# letting jax handle the transposition, but I don't know if this will do what we expect in practice.
@jax.custom_vjp
def f_joint(
    reduced: ReducedState,
    u: Control,
    u_ref: Control,
    params: DynamicsParams,
    dt: float,
) -> ReducedState:
    state, state_ref = lift_reduced_state(reduced)
    new_state = f(state, u, params, dt)
    new_state_ref = f(state_ref, u_ref, params, dt)
    return get_reduced_state(new_state, new_state_ref)


def f_joint_fwd(
    reduced: ReducedState,
    u: Control,
    u_ref: Control,
    params: DynamicsParams,
    dt: float,
):
    return f_joint(reduced, u, u_ref, params, dt), (params, dt)


def f_joint_bwd(res: tuple[DynamicsParams, float], g: ReducedState):
    params, dt = res
    return (
        g,
        jnp.array(
            [
                dt * g[0] + dt / params.m * g[2],
                dt * g[1] + dt / params.m * g[3],
            ]
        ),
        None,
        None,
        None,
    )


f_joint.defvjp(f_joint_fwd, f_joint_bwd)


def sample_initial_states(
    key: jaxtyping.Key, batch: int, params: EnvParams
) -> tuple[State, State]:
    """Reference starts at rest at the origin; primary starts perturbed.

    The perturbation gives the policy a nonzero initial tracking error to
    correct (the paper trains from a randomly sampled initial state).
    """
    kp, kv = jax.random.split(key)
    r0 = params.pos_std * jax.random.normal(kp, (batch, 2))
    v0 = params.vel_std * jax.random.normal(kv, (batch, 2))
    ref_r0 = jnp.zeros((batch, 2))
    ref_v0 = jnp.zeros((batch, 2))
    return State(r0, v0), State(ref_r0, ref_v0)


def cost(out: Rollout[State, Control], params: EnvParams) -> Float[jax.Array, ""]:
    coeffs = params.cost_coeffs

    r_err = jnp.linalg.norm(out.s.r[:-1] - out.s_ref.r[:-1], axis=1)
    v_err = jnp.linalg.norm(out.s.v[:-1] - out.s_ref.v[:-1], axis=1)
    u_err = jnp.linalg.norm(out.us - out.us_ref, axis=1)

    alpha = coeffs.c_r * r_err + jnp.tanh(coeffs.a_r * r_err) - 1.0
    return jnp.mean(alpha + coeffs.c_v * v_err + coeffs.c_u * u_err)


def tracking_error(out: Rollout[State, Control]) -> Float[jax.Array, " steps"]:
    """Per-step position tracking error, for reporting."""
    return jnp.linalg.norm(out.s.r - out.s_ref.r, axis=1)


def save_trajectory_gif(
    out: Rollout[State, Control],
    dt: float,
    path,
    title="trained policy tracking",
    stride=2,
):
    """Animate the reference and primary trajectories and save as a GIF."""
    rs, rs_ref = out.s.r, out.s_ref.r
    frames = range(0, rs.shape[0], stride)

    all_r = jnp.concatenate([rs, rs_ref], axis=0)
    lo = float(all_r.min()) - 0.5
    hi = float(all_r.max()) + 0.5

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    (ref_line,) = ax.plot([], [], "-", color="tab:orange", lw=1, label="reference")
    (pri_line,) = ax.plot([], [], "-", color="tab:blue", lw=1, label="primary")
    (ref_dot,) = ax.plot([], [], "o", color="tab:orange")
    (pri_dot,) = ax.plot([], [], "o", color="tab:blue")
    ax.legend(loc="upper left")

    def update(frame):
        ref_line.set_data(rs_ref[: frame + 1, 0], rs_ref[: frame + 1, 1])
        pri_line.set_data(rs[: frame + 1, 0], rs[: frame + 1, 1])
        ref_dot.set_data([rs_ref[frame, 0]], [rs_ref[frame, 1]])
        pri_dot.set_data([rs[frame, 0]], [rs[frame, 1]])
        return ref_line, pri_line, ref_dot, pri_dot

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=True)
    anim.save(path, writer=animation.PillowWriter(fps=int(1 / (dt * stride))))
    plt.close(fig)
    print(f"saved animation to {path}")
