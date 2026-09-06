"""Astrobee environment (arXiv:2409.11238, Example 2).

State is (q, xi) in SE(3) x R^6: the pose q (a `jaxlie.SE3`) and the *body-frame*
twist xi = (omega, v), packed in arrays as (v, omega) to match jaxlie (see
`Twist`). The control is the applied wrench u = (mu, f) in R^6. The
reduced state is the SE(3) quotient of eq. (55), p(s) = (q^-1 q^d, xi, xi^d),
carried as 19 numbers (the pose error as a quaternion plus a translation). The
policy is fed `get_observation` of that, which re-expresses the pose error as a
rotation matrix plus a translation, with the reference action u^d fed to the
policy separately.

Paper dynamics (52):

    q_{t+1}     = q_t exp(xi_t^ dt)
    v_{t+1}     = v_t + (1/m) f_t dt
    omega_{t+1} = omega_t + J^{-1} (mu_t - omega_t x J omega_t) dt

Note (52b) is missing a Coriolis term: since xi is expressed in the body frame,
Newton's law reads m (vdot + omega x v) = f, so the correct update is

    v_{t+1} = v_t + ((1/m) f_t - omega_t x v_t) dt

which is what is implemented below, mirroring the omega x J omega term in (52c).
"""

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jaxlie
import jaxtyping
import matplotlib.pyplot as plt
from jaxtyping import Float
from matplotlib import animation

from environments.base import EvaluationMetric, Rollout


ENV_NAME = "astrobee"


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


def _flatten_pose(q: jaxlie.SE3) -> Float[jax.Array, "7"]:
    """Flatten a pose as (quaternion wxyz, translation xyz), jaxlie's ordering."""
    return q.parameters()


def _unflatten_pose(flat: Float[jax.Array, "7"]) -> jaxlie.SE3:
    """Inverse of `_flatten_pose`.

    The quaternion is taken as-is. It drifts off unit norm by float32 roundoff
    in the quaternion products (~3e-6 over a 200-step rollout), but jaxlie's
    readouts are scale-invariant -- `as_matrix` divides through by the squared
    norm, `log` goes through atan2(|xyz|, w) -- so a slightly-off-norm
    quaternion still denotes the right rotation.
    """
    return jaxlie.SE3(flat)


def get_reduced_state(state: State, state_ref: State) -> ReducedState:
    """Reduce a state pair to p(s) = (q^-1 q^d, xi, xi^d) (paper eq. 55).

    The symmetry group is K = SE(3) acting on the left, Psi_k(q, xi) = (kq, xi),
    with H = {1} acting trivially on the wrench; Theorem 3 with lambda(s) = q
    gives this quotient map. The twists are *not* differenced (they are already
    body-frame, and K acts trivially on them), and because h_s = id the policy's
    action needs no u^d shift when lifted -- both unlike the Particle of eq. (44).
    """
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
    """A representative of the orbit p^-1(s~), taken on the section q = identity.

    This is the choice the paper makes in eq. (30a)/(46) to define the reduced
    reward, and it is exact for every quantity the cost depends on: the pose
    error, the twists, and the wrench are all K-invariant.
    """
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


# --- analytic VJP of f_q_joint ---------------------------------------------
#
# Everything below works in *ambient* coordinates: a pose is the 7 numbers
# (quaternion wxyz, translation) and a cotangent is another 7 numbers under the
# plain Euclidean pairing. No tangent-space trivialization is involved, so the
# cotangents here are directly comparable to what autodiff produces.
#
# Two primitives suffice, applied at each node of F = A_1 Z A_2:
#   * `_compose_vjp`, the VJP of an SE(3) product, and
#   * `_exp_vjp`, which pulls an ambient cotangent on A = exp(d) back to d.
# The second is the only place Lie theory enters, via the right Jacobian J_r.

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
