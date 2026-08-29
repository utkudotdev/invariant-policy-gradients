"""Astrobee environment (arXiv:2409.11238, Example 2).

State is (q, xi) in SE(3) x R^6: the pose q (a `jaxlie.SE3`) and the *body-frame*
twist xi = (omega, v). The control is the applied wrench u = (mu, f) in R^6.

Two observation encodings are provided, in both cases with the reference action
u^d fed to the policy separately:

  * `FULL_OBSERVATION` -- the paper's baseline, which sees (x, x^d) whole.
  * `REDUCED_OBSERVATION` -- the SE(3) quotient of eq. (55), which sees only
    p(s) = (q^-1 q^d, xi, xi^d).

Paper dynamics (52):

    q_{t+1}     = q_t exp(xi_t^ dt)
    v_{t+1}     = v_t + (1/m) f_t dt
    omega_{t+1} = omega_t + J^{-1} (mu_t - omega_t x J omega_t) dt

Note (52b) is missing a Coriolis term: since xi is expressed in the body frame,
Newton's law reads m (vdot + omega x v) = f, so the correct update is

    v_{t+1} = v_t + ((1/m) f_t - omega_t x v_t) dt

which is what is implemented below, mirroring the omega x J omega term in (52c).
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jaxlie
import jaxtyping
import matplotlib.pyplot as plt
from jaxtyping import Float
from matplotlib import animation

from environments.base import Rollout


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


def wrench_limits() -> Control:
    """Per-axis magnitude bounds on the wrench, ordered as (mu, f).

    Worth respecting even though the paper's MDP leaves U = R^6 unbounded: the
    twist update (52c) is an explicit Euler step on the Euler equations, which
    diverges once ||omega|| grows large. Squashing the policy output through
    these limits keeps long differentiable rollouts finite.
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


def body_twist(omega: Float[jax.Array, "3"], v: Float[jax.Array, "3"]):
    """Pack (omega, v) into jaxlie's se(3) tangent ordering (translation first)."""
    return jnp.concatenate([v, omega])


def f(state: State, u: Control, params: DynamicsParams, dt: float) -> State:
    """One step of the discretized Astrobee dynamics (paper eq. 52, corrected).

    The pose is integrated with the exact SE(3) exponential map; the twist with
    an explicit Euler step including both Coriolis terms.
    """
    mu, force = u[:3], u[3:]
    omega, v = state.omega, state.v

    q_next = state.q @ jaxlie.SE3.exp(body_twist(omega, v) * dt)
    v_next = v + (force / params.m - jnp.cross(omega, v)) * dt
    omega_next = (
        omega + jnp.linalg.solve(params.J, mu - jnp.cross(omega, params.J @ omega)) * dt
    )

    return State(q=q_next, omega=omega_next, v=v_next)


def sample_reference_action(key: jaxtyping.Key, params: EnvParams) -> Control:
    """Draw a reference wrench u^d ~ N(0, Sigma) (paper Def. 5).

    Sigma is diagonal with separate scales for the torque and force blocks; the
    draw is clipped to the actuation limits so the reference trajectory it
    generates is itself dynamically feasible.
    """
    limits = wrench_limits()
    scale = jnp.concatenate(
        [
            jnp.full((3,), params.sigma_torque),
            jnp.full((3,), params.sigma_force),
        ]
    )
    return jnp.clip(scale * jax.random.normal(key, (CONTROL_DIM,)), -limits, limits)


# --- observations ----------------------------------------------------------

Observation = Float[jax.Array, " obs_dim"]
"""A flat encoding of the state pair; see `FULL_OBSERVATION` / `REDUCED_OBSERVATION`."""


def _flatten_pose(q: jaxlie.SE3) -> Float[jax.Array, "12"]:
    """Flatten a pose as (R, p).

    The rotation is exposed as a matrix rather than a quaternion so the policy
    sees a continuous, double-cover-free parameterization of SO(3).
    """
    return jnp.concatenate([q.rotation().as_matrix().reshape(9), q.translation()])


def _unflatten_pose(flat: Float[jax.Array, "12"]) -> jaxlie.SE3:
    return jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_matrix(flat[:9].reshape(3, 3)), flat[9:12]
    )


FULL_OBS_DIM = 36


def get_full_observation(state: State, state_ref: State) -> Observation:
    """Baseline observation: the actual and reference states, seen whole."""
    return jnp.concatenate(
        [
            _flatten_pose(state.q),
            state.omega,
            state.v,
            _flatten_pose(state_ref.q),
            state_ref.omega,
            state_ref.v,
        ]
    )


def lift_full_observation(obs: Observation) -> tuple[State, State]:
    """Inverse of `get_full_observation` (exact, since nothing is reduced away)."""
    return (
        State(q=_unflatten_pose(obs[:12]), omega=obs[12:15], v=obs[15:18]),
        State(q=_unflatten_pose(obs[18:30]), omega=obs[30:33], v=obs[33:36]),
    )


REDUCED_OBS_DIM = 24


def get_reduced_observation(state: State, state_ref: State) -> Observation:
    """Reduced observation p(s) = (q^-1 q^d, xi, xi^d) (paper eq. 55).

    The symmetry group is K = SE(3) acting on the left, Psi_k(q, xi) = (kq, xi),
    with H = {1} acting trivially on the wrench; Theorem 3 with lambda(s) = q
    gives this quotient map. The twists are *not* differenced (they are already
    body-frame, and K acts trivially on them), and because h_s = id the policy's
    action needs no u^d shift when lifted -- both unlike the Particle of eq. (44).
    """
    q_err = state.q.inverse() @ state_ref.q
    return jnp.concatenate(
        [
            _flatten_pose(q_err),
            state.omega,
            state.v,
            state_ref.omega,
            state_ref.v,
        ]
    )


def lift_reduced_observation(obs: Observation) -> tuple[State, State]:
    """A representative of the orbit p^-1(s~), taken on the section q = identity.

    This is the choice the paper makes in eq. (30a)/(46) to define the reduced
    reward, and it is exact for every quantity the cost depends on: the pose
    error, the twists, and the wrench are all K-invariant.
    """
    return (
        State(q=jaxlie.SE3.identity(), omega=obs[12:15], v=obs[15:18]),
        State(q=_unflatten_pose(obs[:12]), omega=obs[18:21], v=obs[21:24]),
    )


def make_f_joint(lift, encode):
    """Step the actual and reference states together in observation space.

    With the reduced encoding this is exactly the quotient transition: lifting to
    q = identity and re-encoding turns `q -> q exp(xi^ dt)` into
    `q_err -> exp(xi^ dt)^-1 q_err exp(xi^d^ dt)`.
    """

    def f_joint(
        obs: Observation, u: Control, u_ref: Control, params: DynamicsParams, dt: float
    ) -> Observation:
        state, state_ref = lift(obs)
        return encode(f(state, u, params, dt), f(state_ref, u_ref, params, dt))

    return f_joint


@dataclass(frozen=True)
class ObservationEncoding:
    """What a policy is allowed to see, and how to roll it forward."""

    name: str
    dim: int
    encode: Callable[[State, State], Observation]
    lift: Callable[[Observation], tuple[State, State]]
    step: Callable[..., Observation]


FULL_OBSERVATION = ObservationEncoding(
    name="full-state (baseline)",
    dim=FULL_OBS_DIM,
    encode=get_full_observation,
    lift=lift_full_observation,
    step=make_f_joint(lift_full_observation, get_full_observation),
)

REDUCED_OBSERVATION = ObservationEncoding(
    name="reduced (SE(3) quotient)",
    dim=REDUCED_OBS_DIM,
    encode=get_reduced_observation,
    lift=lift_reduced_observation,
    step=make_f_joint(lift_reduced_observation, get_reduced_observation),
)


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
