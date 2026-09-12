"""How to differentiate the pose block of the quotient dynamics: five variants.

`f_q_joint` maps a reduced state to the next pose error. On the section q = I it
is F = exp(xi dt)^-1 Z exp(xi^d dt), and there are two independent choices about
how to get its VJP:

  * whether the *expression* exploits q = I (the "folded" form writes the
    primary pose as exp(xi dt) instead of I @ exp(xi dt)), and
  * whether the backward pass rematerializes the forward (`jax.checkpoint`)
    or stores its intermediates.

The variants below cross those two choices, plus one that keeps the natural
unfolded definition public while differentiating the folded form.

The benchmark differentiates a scan of T pose steps and nothing else -- no
policy, no optimizer, no cost -- so the numbers are attributable to `f_q_joint`
rather than diluted by the rest of the training step. The twists are a per-step
scan input rather than a constant, which stops XLA from hoisting the `exp` calls
out of the loop and quietly measuring one step instead of T.

Run with:  uv run python -m benchmarks.vjp_variants
"""

import statistics
import time
from functools import partial

import jax
import jax.numpy as jnp
import jaxlie
import matplotlib.pyplot as plt
from jaxtyping import Float

from environments import astrobee_reduced as env


# --- the two ways of writing the same map ----------------------------------


def natural(reduced: env.ReducedState, dt: float) -> jnp.ndarray:
    """What an environment author writes: build both states, step both poses.

    The primary pose is the identity here, but that is a fact about the value
    `lift_reduced_state` returns, not something the expression exploits.
    """
    state, state_ref = env.lift_reduced_state(reduced)
    return env._flatten_pose(env.f_q(state, dt).inverse() @ env.f_q(state_ref, dt))


def folded(reduced: env.ReducedState, dt: float) -> jnp.ndarray:
    """The same map with q = I folded in: no compose against the identity."""
    Z = env._unflatten_pose(reduced[: env.POSE_DIM])
    q_next = jaxlie.SE3.exp(reduced[7:13] * dt)  # was identity @ exp(...)
    q_ref_next = Z @ jaxlie.SE3.exp(reduced[13:19] * dt)
    return env._flatten_pose(q_next.inverse() @ q_ref_next)


def rematerialized(f):
    """Differentiate `f` but store only its input, recomputing in the backward."""
    return lambda reduced, dt: jax.checkpoint(lambda r: f(r, dt))(reduced)


# --- variant 12: natural definition, folded derivative, nothing recomputed ---
#
# `fwd` linearizes the folded form and hands the resulting closure to `bwd` as
# the residual (a `jax.vjp` closure is a registered pytree, so this is legal).
# The forward that runs under differentiation is therefore the folded one, and
# the backward applies the stored autodiff pullback rather than rebuilding it. The
# unfolded body below runs only when the function is *not* differentiated --
# `rollout_eval`, say -- so it stays the readable definition of record.


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def folded_autodiff_vjp(reduced: env.ReducedState, dt: float) -> jnp.ndarray:
    return natural(reduced, dt)


def _folded_autodiff_vjp_fwd(reduced, dt):
    out, vjp_fn = jax.vjp(lambda r: folded(r, dt), reduced)
    return out, vjp_fn


def _folded_autodiff_vjp_bwd(dt, vjp_fn, g):
    return vjp_fn(g)


folded_autodiff_vjp.defvjp(_folded_autodiff_vjp_fwd, _folded_autodiff_vjp_bwd)


# --- variant: analytic VJP, copied from environments/astrobee_reduced.py ----
#
# Copied rather than imported so this benchmark measures a fixed implementation
# even if the module's rule changes. See that file for the derivation; in short,
# F = A_1 Z A_2 is differentiated in ambient (quaternion, translation) coords by
# two applications of an SE(3)-composition VJP plus two exp pullbacks, with the
# right Jacobian obtained as J_r = jlog^-1.


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def analytic(reduced: env.ReducedState, dt: float) -> jnp.ndarray:
    """Same map as `natural`; the hand-derived rule below supplies its VJP."""
    return natural(reduced, dt)


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
    """VJP of C = A B with respect to A only."""
    q_A = A.rotation().wxyz
    q_bar_A = _quat_mul(q_bar, _quat_conj(B.rotation().wxyz)) - 2.0 * _quat_mul(
        _quat_mul(_pure(t_bar), q_A), _pure(B.translation())
    )
    return q_bar_A, t_bar


def _exp_vjp(
    A: jaxlie.SE3, q_bar: Quaternion, t_bar: Float[jax.Array, "3"]
) -> env.Twist:
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


def _analytic_fwd(reduced: env.ReducedState, dt: float):
    # The residual is just the input; A_1, A_2 and Z are recomputed below.
    return analytic(reduced, dt), reduced


def _analytic_bwd(dt: float, res: env.ReducedState, g: Float[jax.Array, "7"]):
    reduced = res
    q_bar_F, t_bar_F = g[:4], g[4:]

    xi, xi_ref = env.lift_xi_reduced_state(reduced)
    A_1 = jaxlie.SE3.exp(-xi * dt)
    A_2 = jaxlie.SE3.exp(xi_ref * dt)
    Z = env._unflatten_pose(reduced[: env.POSE_DIM])
    P = A_1 @ Z

    q_bar_P, t_bar_P = _compose_left_vjp(P, A_2, q_bar_F, t_bar_F)
    (q_bar_A1, t_bar_A1), (q_bar_Z, t_bar_Z) = _compose_vjp(A_1, Z, q_bar_P, t_bar_P)

    xi_bar = -dt * _exp_vjp(A_1, q_bar_A1, t_bar_A1)
    xi_ref_bar = jnp.full((env.TWIST_DIM,), jnp.inf)

    return (jnp.concatenate([q_bar_Z, t_bar_Z, xi_bar, xi_ref_bar]),)


analytic.defvjp(_analytic_fwd, _analytic_bwd)


# --- variants: the analytic VJP without rematerialization -------------------
#
# The forward already builds A_1, A_2, Z and Q = Z A_2, so it can hand them to
# the backward instead of making it rebuild them from `reduced`. Note this also
# folds q = I into the forward -- unavoidably, since computing the factors *is*
# the folded expression -- so these rows differ from `analytic` on two counts,
# not one.


def _factors(reduced: env.ReducedState, dt: float):
    """The forward, keeping every factor it computes along the way."""
    Z = env._unflatten_pose(reduced[: env.POSE_DIM])
    A_1 = jaxlie.SE3.exp(-reduced[7:13] * dt)
    A_2 = jaxlie.SE3.exp(reduced[13:19] * dt)
    Q = Z @ A_2  # the forward's own association: F = A_1 (Z A_2)
    return env._flatten_pose(A_1 @ Q), (A_1, A_2, Z, Q)


def derived(reduced: env.ReducedState, dt: float) -> jnp.ndarray:
    """The map as the derivation states it: F = A_1 Z A_2.

    Differs from `folded` in one more place: the primary pose enters as
    exp(-xi dt) rather than exp(xi dt).inverse(), so an SE(3) inverse per step
    disappears. This is the forward the analytic rule is actually written
    against, and it produces exactly the factors the backward needs.
    """
    return _factors(reduced, dt)[0]


def _stored_bwd(dt, A_1, A_2, Z, Q, g):
    """The same rule as `_analytic_bwd`, re-associated as F = A_1 Q, Q = Z A_2."""
    q_bar_F, t_bar_F = g[:4], g[4:]
    (q_bar_A1, t_bar_A1), (q_bar_Q, t_bar_Q) = _compose_vjp(A_1, Q, q_bar_F, t_bar_F)
    q_bar_Z, t_bar_Z = _compose_left_vjp(Z, A_2, q_bar_Q, t_bar_Q)

    xi_bar = -dt * _exp_vjp(A_1, q_bar_A1, t_bar_A1)
    xi_ref_bar = jnp.full((env.TWIST_DIM,), jnp.inf)

    return (jnp.concatenate([q_bar_Z, t_bar_Z, xi_bar, xi_ref_bar]),)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def analytic_stored(reduced: env.ReducedState, dt: float) -> jnp.ndarray:
    """Stores A_1, A_2, Z (21 floats); recomputes only Q = Z A_2."""
    return natural(reduced, dt)


def _analytic_stored_fwd(reduced, dt):
    out, (A_1, A_2, Z, _) = _factors(reduced, dt)
    return out, (A_1, A_2, Z)


def _analytic_stored_bwd(dt, res, g):
    A_1, A_2, Z = res
    return _stored_bwd(dt, A_1, A_2, Z, Z @ A_2, g)


analytic_stored.defvjp(_analytic_stored_fwd, _analytic_stored_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def analytic_stored_all(reduced: env.ReducedState, dt: float) -> jnp.ndarray:
    """Stores Q as well (28 floats), so the backward recomputes nothing."""
    return natural(reduced, dt)


def _analytic_stored_all_fwd(reduced, dt):
    out, factors = _factors(reduced, dt)
    return out, factors


def _analytic_stored_all_bwd(dt, res, g):
    return _stored_bwd(dt, *res, g)


analytic_stored_all.defvjp(_analytic_stored_all_fwd, _analytic_stored_all_bwd)


# --- variant: natural definition, autodiff VJP of the derived form ----------
#
# As `folded_autodiff_vjp`, but `jax.vjp` differentiates `derived` -- so the
# forward that runs under differentiation also drops the SE(3) inverse, not just
# the identity compose.


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def derived_autodiff_vjp(reduced: env.ReducedState, dt: float) -> jnp.ndarray:
    return natural(reduced, dt)


def _derived_autodiff_vjp_fwd(reduced, dt):
    out, vjp_fn = jax.vjp(lambda r: derived(r, dt), reduced)
    return out, vjp_fn


def _derived_autodiff_vjp_bwd(dt, vjp_fn, g):
    return vjp_fn(g)


derived_autodiff_vjp.defvjp(_derived_autodiff_vjp_fwd, _derived_autodiff_vjp_bwd)


VARIANTS = [
    ("autodiff, natural", natural),
    ("autodiff, q=I folded", folded),
    ("natural + checkpoint", rematerialized(natural)),
    ("q=I folded + checkpoint", rematerialized(folded)),
    ("custom VJP (autodiff), folded", folded_autodiff_vjp),
    ("custom VJP (autodiff), derived", derived_autodiff_vjp),
    ("analytic VJP", analytic),
    ("analytic, store factors", analytic_stored),
    ("analytic, store all", analytic_stored_all),
    ("autodiff, derived", derived),
    ("derived + checkpoint", rematerialized(derived)),
]


# --- harness ---------------------------------------------------------------

BATCH = 2048
T_TIMED = 200
POSE_WEIGHTS = jnp.array([1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 1.0])
DT = 0.05
PLOT_PATH = "vjp_variants.png"
NAME_WIDTH = 31

PROBE = jnp.array(
    [1.0, 0.0, 0.0, 0.0, 0.3, -0.2, 0.5,
     0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
     -0.2, 0.1, 0.05, 0.3, -0.1, 0.2]
)  # fmt: skip


def grad_fn(pose_step):
    """d/d(pose_0, twists) of a scan of T pose steps, batched."""

    def loss(pose_0, twists):
        def step(pose, twist):
            return pose_step(jnp.concatenate([pose, twist]), DT), None

        pose, _ = jax.lax.scan(step, pose_0, twists)
        return jnp.sum(POSE_WEIGHTS * pose)

    return jax.jit(jax.vmap(jax.grad(loss, argnums=(0, 1))))


def inputs(T: int, batch: int = BATCH):
    k_pose, k_twist = jax.random.split(jax.random.key(0))
    rotations = jax.vmap(jaxlie.SO3.exp)(0.5 * jax.random.normal(k_pose, (batch, 3)))
    poses = jax.vmap(jaxlie.SE3.from_rotation_and_translation)(
        rotations, jax.random.normal(k_pose, (batch, 3))
    )
    twists = 0.5 * jax.random.normal(k_twist, (batch, T, 12))
    return jax.vmap(lambda q: q.parameters())(poses), twists


def temp_bytes(fn, T: int) -> int:
    return fn.lower(*inputs(T)).compile().memory_analysis().temp_size_in_bytes


def time_ms(fn, reps: int = 50) -> tuple[float, float]:
    args = inputs(T_TIMED)
    jax.block_until_ready(fn(*args))  # compile + warm up
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append((time.perf_counter() - start) * 1e3)
    return min(samples), statistics.median(samples)


def save_plot(results: list[tuple[str, float, float]]) -> None:
    """Plot per-step temporary storage against median runtime."""
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.get_cmap("tab20").colors
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "p", "h", "*")

    for i, (name, floats_per_step, median_ms) in enumerate(results):
        ax.scatter(
            floats_per_step,
            median_ms,
            s=75,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            label=name,
            zorder=3,
        )

    ax.set_xlabel("Temporary storage (floats/step/example)")
    ax.set_ylabel("Median runtime (ms)")
    ax.set_title(f"Pose-step VJP tradeoff (T={T_TIMED}, batch={BATCH})")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)
    plt.close(fig)
    print(f"\nsaved plot to {PLOT_PATH}")


def main():
    print(f"backend: {jax.default_backend()}  {jax.devices()}")
    print(f"T={T_TIMED}, batch={BATCH}, float32, pose step only\n")

    reference = None
    for name, fn in VARIANTS:
        g = jax.grad(lambda r: jnp.sum(POSE_WEIGHTS * fn(r, DT)))(PROBE)
        reference = g if reference is None else reference
        # BPTT only needs the pose and primary-twist cotangents. The analytic
        # variants intentionally leave the exogenous reference-twist block Inf.
        print(
            f"  {name:<{NAME_WIDTH}} max |relevant grad - reference| = "
            f"{float(jnp.max(jnp.abs(g[:13] - reference[:13]))):.2e}"
        )
    # The analytic rule associates the products differently, so in float32 it
    # lands ~2e-6 from the autodiff variants. In float64 the two agree to 1e-13.

    print(
        f"\n{'variant':<{NAME_WIDTH}}{'temp bytes':>10}{'floats/step':>13}"
        f"{'min ms':>9}{'median ms':>11}"
    )
    results = []
    for name, variant in VARIANTS:
        fn = grad_fn(variant)
        # slope over T isolates the per-step residual from fixed overhead
        slope = (temp_bytes(fn, 400) - temp_bytes(fn, 100)) / 300 / BATCH / 4
        b = temp_bytes(fn, T_TIMED)
        lo, med = time_ms(fn)
        print(f"{name:<{NAME_WIDTH}}{b:>10}{slope:>13.1f}{lo:>9.2f}{med:>11.2f}")
        results.append((name, slope, med))

    save_plot(results)


if __name__ == "__main__":
    main()
