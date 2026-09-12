"""Unreduced Astrobee environment (arXiv:2409.11238, Example 2).

This is the canonical home of the Astrobee state, parameters, dynamics,
sampling, objective, and reporting code. State is (q, xi) in SE(3) x R^6: the
pose q (a `jaxlie.SE3`) and the *body-frame* twist xi = (omega, v), packed in
arrays as (v, omega) to match jaxlie (see `Twist`). The control is the applied
wrench u = (mu, f) in R^6.

For joint rollouts, this module keeps both absolute states. The reduced
implementation is in :mod:`environments.astrobee_reduced`.

Paper dynamics (52):

    q_{t+1}     = q_t exp(xi_t^ dt)
    v_{t+1}     = v_t + (1/m) f_t dt
    omega_{t+1} = omega_t + J^{-1} (mu_t - omega_t x J omega_t) dt

Note (52b) is missing a Coriolis term: since xi is expressed in the body frame,
the correct update is

    v_{t+1} = v_t + ((1/m) f_t - omega_t x v_t) dt

which is what is implemented below.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jaxlie
import jaxtyping
import matplotlib.pyplot as plt
from jaxtyping import Float
from matplotlib import animation

from environments.base import EvaluationMetric, Rollout

ENV_NAME = "astrobee"

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


@jax.tree_util.register_dataclass
@dataclass
class State:
    q: jaxlie.SE3  # pose (body -> world)
    omega: Float[jax.Array, "3"]  # body-frame angular velocity
    v: Float[jax.Array, "3"]  # body-frame linear velocity


@jax.tree_util.register_dataclass
@dataclass
class DynamicsParams:
    m: float
    J: Float[jax.Array, "3 3"]  # body-frame inertia tensor


Control = Float[jax.Array, "6"]
"""Applied wrench u = (mu, f): torque in u[:3], force in u[3:]."""

CONTROL_DIM = 6

# Astrobee physical parameters (mass in kg, inertia in kg m^2).
ASTROBEE_MASS = 9.58
ASTROBEE_INERTIA = jnp.diag(jnp.array([0.153, 0.143, 0.162]))

# Actuation limits of the impeller/nozzle propulsion system (N and N m, per axis).
ASTROBEE_MAX_FORCE = 0.85
ASTROBEE_MAX_TORQUE = 0.1


def control_limits() -> Control:
    """Per-axis magnitude bounds on the wrench, ordered as (mu, f).
    Paper leaves this unbounded but depending on initialization we may run into numerical problems
    """
    return jnp.concatenate(
        [
            jnp.full((3,), ASTROBEE_MAX_TORQUE),
            jnp.full((3,), ASTROBEE_MAX_FORCE),
        ]
    )


def default_dynamics_params() -> DynamicsParams:
    return DynamicsParams(m=ASTROBEE_MASS, J=ASTROBEE_INERTIA)


@jax.tree_util.register_dataclass
@dataclass
class CostCoeffs:
    """Coefficients for the tracking cost (paper eq. 53)."""

    c_r: float  # linear position-error penalty
    a_r: float  # sharpness of the near-zero tracking bonus
    c_R: float  # attitude-error penalty
    c_xi: float  # twist-error penalty
    c_u: float  # effort penalty (deviation from the reference wrench)


@jax.tree_util.register_dataclass
@dataclass
class EnvParams:
    cost_coeffs: CostCoeffs
    sigma_torque: float  # std of the reference torque
    sigma_force: float  # std of the reference force
    pos_std: float
    att_std: float  # std of the initial rotation vector, in radians
    vel_std: float
    omega_std: float


def default_env_params() -> EnvParams:
    return EnvParams(
        cost_coeffs=CostCoeffs(c_r=1.0, a_r=5.0, c_R=1.0, c_xi=0.5, c_u=0.1),
        sigma_torque=0.03,
        sigma_force=0.3,
        pos_std=0.5,
        att_std=0.3,
        vel_std=0.1,
        omega_std=0.1,
    )


Twist = Float[jax.Array, "6"]
"""Body-frame twist packed as (v, omega): *translation first*.

The paper writes xi = (omega, v), but jaxlie's se(3) tangent is translation
first, and uniformly so -- `SE3.exp`, `SE3.log`, `SE3.adjoint` (which is the
block-upper-triangular [[R, skew(p) R], [0, R]]) and `SE3.jlog` all use it. We
adopt jaxlie's order so twists can be handed to those directly: the alternative
is a permutation similarity around every adjoint, which is an easy way to get a
silently wrong gradient. The ordering is only ever visible inside a packed
`Twist` -- `State` names `omega` and `v` as separate fields.
"""


def body_twist(v: Float[jax.Array, "3"], omega: Float[jax.Array, "3"]) -> Twist:
    """Pack (v, omega) into a `Twist`. Note the argument order."""
    return jnp.concatenate([v, omega])


def unpack_twist(twist: Twist) -> tuple[Float[jax.Array, "3"], Float[jax.Array, "3"]]:
    """Split a `Twist` back into (v, omega)."""
    return twist[:3], twist[3:]


def state_twist(state: State) -> Twist:
    """The state's body twist, packed. Preferred over calling `body_twist` with
    the fields by hand, which is where the argument order is easy to flip."""
    return body_twist(state.v, state.omega)


def f(state: State, u: Control, params: DynamicsParams, dt: float) -> State:
    """One step of the discretized Astrobee dynamics (paper eq. 52, corrected).

    The pose is integrated with the exact SE(3) exponential map; the twist with
    an explicit Euler step including both Coriolis terms.
    """
    q_next = f_q(state, dt)
    v_next, omega_next = unpack_twist(f_xi(state_twist(state), u, params, dt))

    return State(q=q_next, omega=omega_next, v=v_next)


# need these factored out for later
def f_q(state: State, dt: float) -> jaxlie.SE3:
    return state.q @ jaxlie.SE3.exp(state_twist(state) * dt)


def f_xi(twist: Twist, u: Control, params: DynamicsParams, dt: float) -> Twist:
    v, omega = unpack_twist(twist)
    mu, force = u[:3], u[3:]

    v_next = v + (force / params.m - jnp.cross(omega, v)) * dt
    omega_next = (
        omega + jnp.linalg.solve(params.J, mu - jnp.cross(omega, params.J @ omega)) * dt
    )

    return body_twist(v_next, omega_next)


def sample_reference_action(key: jaxtyping.Key, params: EnvParams) -> Control:
    """Draw a reference wrench u^d ~ N(0, Sigma) (paper Def. 5).

    Sigma is diagonal with separate scales for the torque and force blocks; the
    draw is clipped to the actuation limits so the reference trajectory it
    generates is itself dynamically feasible.
    """
    limits = control_limits()
    scale = jnp.concatenate(
        [
            jnp.full((3,), params.sigma_torque),
            jnp.full((3,), params.sigma_force),
        ]
    )
    return jnp.clip(scale * jax.random.normal(key, (CONTROL_DIM,)), -limits, limits)


# --- full joint state and observation ---------------------------------------

POSE_DIM = 7
TWIST_DIM = 6
STATE_DIM = POSE_DIM + TWIST_DIM
JOINT_STATE_DIM = 2 * STATE_DIM
OBS_POSE_DIM = 12
OBS_DIM = 2 * (OBS_POSE_DIM + TWIST_DIM)

JointState = Float[jax.Array, "26"]
"""Both absolute states packed as ``(q, xi, q_ref, xi_ref)``."""

# The training code uses this interface for both environment variants.
ReducedState = JointState
REDUCED_DIM = JOINT_STATE_DIM

Observation = Float[jax.Array, "36"]
"""Both absolute states as ``(R, p, v, omega, R_ref, p_ref, v_ref, omega_ref)``."""


def _flatten_pose(q: jaxlie.SE3) -> Float[jax.Array, "7"]:
    """Flatten a pose as (quaternion wxyz, translation xyz), jaxlie's ordering."""
    return q.parameters()


def _unflatten_pose(flat: Float[jax.Array, "7"]) -> jaxlie.SE3:
    """Reconstruct a pose from jaxlie's quaternion-and-translation ordering.

    The quaternion is taken as-is. It can drift slightly off unit norm through
    repeated products, but jaxlie's pose readouts are scale-invariant.
    """
    return jaxlie.SE3(flat)


def _pack_state(state: State) -> Float[jax.Array, "13"]:
    return jnp.concatenate([_flatten_pose(state.q), state_twist(state)])


def _unpack_state(flat: Float[jax.Array, "13"]) -> State:
    v, omega = unpack_twist(flat[POSE_DIM:STATE_DIM])
    return State(q=_unflatten_pose(flat[:POSE_DIM]), omega=omega, v=v)


def get_reduced_state(state: State, state_ref: State) -> ReducedState:
    """Pack two states without quotienting out their common SE(3) pose."""
    return jnp.concatenate([_pack_state(state), _pack_state(state_ref)])


def lift_reduced_state(joint: ReducedState) -> tuple[State, State]:
    """Unpack the full joint state; this loses no information."""
    return _unpack_state(joint[:STATE_DIM]), _unpack_state(joint[STATE_DIM:])


def _pose_observation(state: State) -> Float[jax.Array, "12"]:
    return jnp.concatenate(
        [state.q.rotation().as_matrix().reshape(9), state.q.translation()]
    )


def get_observation(joint: ReducedState) -> Observation:
    state, state_ref = lift_reduced_state(joint)
    return jnp.concatenate(
        [
            _pose_observation(state),
            state_twist(state),
            _pose_observation(state_ref),
            state_twist(state_ref),
        ]
    )


def f_joint(
    joint: ReducedState,
    u: Control,
    u_ref: Control,
    params: DynamicsParams,
    dt: float,
) -> ReducedState:
    """Advance both absolute states one step."""
    state, state_ref = lift_reduced_state(joint)
    return get_reduced_state(f(state, u, params, dt), f(state_ref, u_ref, params, dt))


# --- initial states --------------------------------------------------------


def sample_initial_states(
    key: jaxtyping.Key, batch: int, params: EnvParams
) -> tuple[State, State]:
    """Reference starts at rest at the identity pose; primary starts perturbed."""
    kp, ka, kv, kw = jax.random.split(key, 4)

    p0 = params.pos_std * jax.random.normal(kp, (batch, 3))
    phi0 = params.att_std * jax.random.normal(ka, (batch, 3))
    R0 = jax.vmap(jaxlie.SO3.exp)(phi0)
    q0 = jax.vmap(jaxlie.SE3.from_rotation_and_translation)(R0, p0)
    v0 = params.vel_std * jax.random.normal(kv, (batch, 3))
    omega0 = params.omega_std * jax.random.normal(kw, (batch, 3))

    zeros = jnp.zeros((batch, 3))
    q_ref = jax.vmap(jaxlie.SE3.from_rotation_and_translation)(
        jax.vmap(jaxlie.SO3.exp)(zeros), zeros
    )
    ref = State(q=q_ref, omega=zeros, v=zeros)

    return State(q=q0, omega=omega0, v=v0), ref


# --- cost ------------------------------------------------------------------


def _pose_error(state: State, state_ref: State):
    """(position error, attitude error) between a state and its reference."""
    p_err = jnp.linalg.norm(state.q.translation() - state_ref.q.translation())
    R_err = jnp.linalg.norm(
        (state.q.rotation().inverse() @ state_ref.q.rotation()).log()
    )
    return p_err, R_err


def cost(out: Rollout[State, Control], params: EnvParams) -> Float[jax.Array, ""]:
    """Mean running cost (paper eq. 53)."""
    coeffs = params.cost_coeffs

    steps = jax.tree.map(lambda x: x[:-1], out.s)
    steps_ref = jax.tree.map(lambda x: x[:-1], out.s_ref)
    p_err, R_err = jax.vmap(_pose_error)(steps, steps_ref)

    xi_err = jnp.linalg.norm(
        jnp.concatenate([steps.omega - steps_ref.omega, steps.v - steps_ref.v], axis=1),
        axis=1,
    )
    u_err = jnp.linalg.norm(out.us - out.us_ref, axis=1)

    alpha = coeffs.c_r * p_err + jnp.tanh(coeffs.a_r * p_err) - 1.0
    return jnp.mean(
        alpha + coeffs.c_R * R_err + coeffs.c_xi * xi_err + coeffs.c_u * u_err
    )


def tracking_error(out: Rollout[State, Control]) -> Float[jax.Array, " steps+1"]:
    """Per-step position tracking error, for reporting."""
    return jax.vmap(_pose_error)(out.s, out.s_ref)[0]


def attitude_error(out: Rollout[State, Control]) -> Float[jax.Array, " steps+1"]:
    """Per-step attitude tracking error ||log(R^T R^d)||, for reporting."""
    return jax.vmap(_pose_error)(out.s, out.s_ref)[1]


def evaluation_metrics(out: Rollout[State, Control]) -> tuple[EvaluationMetric, ...]:
    return (
        EvaluationMetric("position error", tracking_error(out), "m"),
        EvaluationMetric("attitude error", attitude_error(out), "rad"),
    )


# --- visualization ---------------------------------------------------------


def save_trajectory_gif(
    out: Rollout[State, Control],
    dt: float,
    path,
    title="trained policy tracking",
    stride=2,
    axis_len=0.3,
):
    """Animate the reference and primary poses in 3D and save as a GIF.

    Each body is drawn as its position trail plus a body-frame triad (red/green/
    blue for the x/y/z axes), solid for the primary and dashed for the reference.
    """
    ps = jax.vmap(lambda q: q.translation())(out.s.q)
    ps_ref = jax.vmap(lambda q: q.translation())(out.s_ref.q)
    Rs = jax.vmap(lambda q: q.rotation().as_matrix())(out.s.q)
    Rs_ref = jax.vmap(lambda q: q.rotation().as_matrix())(out.s_ref.q)
    frames = range(0, ps.shape[0], stride)

    all_p = jnp.concatenate([ps, ps_ref], axis=0)
    lo = jnp.min(all_p, axis=0) - axis_len
    hi = jnp.max(all_p, axis=0) + axis_len
    # Equal aspect: pad every axis out to the largest extent.
    center = (lo + hi) / 2
    half = float(jnp.max(hi - lo)) / 2 + 1e-3

    fig = plt.figure(figsize=(6.5, 6))
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(float(center[0]) - half, float(center[0]) + half)
    ax.set_ylim(float(center[1]) - half, float(center[1]) + half)
    ax.set_zlim(float(center[2]) - half, float(center[2]) + half)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)

    (ref_line,) = ax.plot([], [], [], "-", color="tab:orange", lw=1, label="reference")
    (pri_line,) = ax.plot([], [], [], "-", color="tab:blue", lw=1, label="primary")
    axis_colors = ["tab:red", "tab:green", "tab:blue"]
    pri_axes = [ax.plot([], [], [], "-", color=c, lw=2)[0] for c in axis_colors]
    ref_axes = [ax.plot([], [], [], "--", color=c, lw=2)[0] for c in axis_colors]
    ax.legend(loc="upper left")

    def set_triad(lines, p, R):
        for i, line in enumerate(lines):
            tip = p + axis_len * R[:, i]
            line.set_data_3d(
                [float(p[0]), float(tip[0])],
                [float(p[1]), float(tip[1])],
                [float(p[2]), float(tip[2])],
            )

    def update(frame):
        ref_line.set_data_3d(
            ps_ref[: frame + 1, 0], ps_ref[: frame + 1, 1], ps_ref[: frame + 1, 2]
        )
        pri_line.set_data_3d(ps[: frame + 1, 0], ps[: frame + 1, 1], ps[: frame + 1, 2])
        set_triad(pri_axes, ps[frame], Rs[frame])
        set_triad(ref_axes, ps_ref[frame], Rs_ref[frame])
        return [ref_line, pri_line, *pri_axes, *ref_axes]

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=int(1 / (dt * stride))))
    plt.close(fig)
    print(f"saved animation to {path}")
