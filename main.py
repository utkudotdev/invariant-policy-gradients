from functools import partial
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from matplotlib import animation


def f(state, u, m, dt):
    """Discretized point-particle dynamics.

    state = (r, v) with r, v in R^2. Applies control force u over one step:
        r_{t+1} = r_t + v_t dt
        v_{t+1} = v_t + (1/m) u_t dt
    """
    r, v = state
    r_next = r + v * dt
    v_next = v + (u / m) * dt
    return (r_next, v_next)


@partial(jax.jit, static_argnums=(3,))
def rollout(state0, controls, m, dt):
    """Roll out the dynamics under a fixed sequence of controls.

    controls has shape (T, 2). Returns (rs, vs), each of shape (T + 1, 2).
    """

    def step(state, u):
        next_state = f(state, u, m, dt)
        return next_state, next_state

    _, (rs, vs) = jax.lax.scan(step, state0, controls)
    r0, v0 = state0
    rs = jnp.concatenate([r0[None], rs], axis=0)
    vs = jnp.concatenate([v0[None], vs], axis=0)
    return rs, vs


def sample_reference_action(key, sigma):
    """Draw a reference action u^d ~ N(0, sigma^2 I).

    The paper (arXiv:2409.11238, eq. 10-11c) models the reference action
    distribution rho for the Particle as an isotropic Gaussian N(0, Sigma).
    """
    return sigma * jax.random.normal(key, (2,))


class FullStatePolicy(eqx.Module):
    """MLP policy on the paper's baseline observation (x, x^d, u^d).

    Observes the primary state, reference state, and reference action --
    flattened to a length-10 vector -- and outputs a 2D force.
    """

    mlp: eqx.nn.MLP

    def __init__(self, key, width=32, depth=2):
        self.mlp = eqx.nn.MLP(
            in_size=10, out_size=2, width_size=width, depth=depth, key=key
        )

    def __call__(self, state, ref_state, u_ref):
        r, v = state
        r_ref, v_ref = ref_state
        obs = jnp.concatenate([r, v, r_ref, v_ref, u_ref])
        return self.mlp(obs)


class ReducedPolicy(eqx.Module):
    """MLP policy on the position+velocity reduced observation.

    Uses the paper's translation+velocity (TR^3) state reduction:
        p(s) = (r - r^d, v - v^d, u^d)                              (eq. 44)
    i.e. the position error, velocity error, and reference action -- a
    6-vector instead of the baseline's 10. This exploits the fact that the
    dynamics and cost depend only on the errors (and u^d), not on absolute
    position/velocity, so the policy sees a lower-dimensional input.
    """

    mlp: eqx.nn.MLP

    def __init__(self, key, width=32, depth=2):
        self.mlp = eqx.nn.MLP(
            in_size=6, out_size=2, width_size=width, depth=depth, key=key
        )

    def __call__(self, state, ref_state, u_ref):
        r, v = state
        r_ref, v_ref = ref_state
        obs = jnp.concatenate([r - r_ref, v - v_ref, u_ref])
        return self.mlp(obs)


class Rollout(eqx.Module):
    """A jointly-rolled-out reference and primary trajectory.

    Position/velocity arrays have shape (T + 1, 2); action arrays (T, 2).
    """

    rs: jax.Array
    vs: jax.Array
    rs_ref: jax.Array
    vs_ref: jax.Array
    us: jax.Array
    us_ref: jax.Array


def rollout_joint(policy, state0, ref_state0, key, m, dt, sigma, T):
    """Jointly roll out the reference and primary (policy) trajectories.

    At each step we draw a reference action u^d ~ N(0, sigma^2 I), advance the
    reference state under f, query the policy for the primary action u, and
    advance the primary state under f. Rolling both out in a single scan lets
    the primary action depend on the reference state and reference action.
    """

    def step(carry, key_t):
        state, ref_state = carry

        u_ref = sample_reference_action(key_t, sigma)
        u = policy(state, ref_state, u_ref)

        next_state = f(state, u, m, dt)
        next_ref = f(ref_state, u_ref, m, dt)
        return (next_state, next_ref), (next_state, next_ref, u, u_ref)

    keys = jax.random.split(key, T)
    _, (states, ref_states, us, us_ref) = jax.lax.scan(step, (state0, ref_state0), keys)
    rs, vs = states
    rs_ref, vs_ref = ref_states

    r0, v0 = state0
    r0_ref, v0_ref = ref_state0
    return Rollout(
        rs=jnp.concatenate([r0[None], rs], axis=0),
        vs=jnp.concatenate([v0[None], vs], axis=0),
        rs_ref=jnp.concatenate([r0_ref[None], rs_ref], axis=0),
        vs_ref=jnp.concatenate([v0_ref[None], vs_ref], axis=0),
        us=us,
        us_ref=us_ref,
    )


class Coeffs(NamedTuple):
    """Coefficients for the tracking cost (paper eq. 8/12)."""

    c_r: float = 1.0  # linear position-error penalty
    a_r: float = 5.0  # sharpness of the near-zero tracking bonus
    c_v: float = 0.5  # velocity-error penalty
    c_u: float = 0.1  # effort penalty (deviation from reference action)


def tracking_cost(out: Rollout, coeffs: Coeffs):
    """Mean per-step tracking cost, i.e. the negative of the paper's reward.

        alpha(y)   = c_r ||y|| + tanh(a_r ||y||) - 1                    (eq. 8a)
        cost(s, a) = alpha(r - r^d) + c_v ||v - v^d|| + c_u ||u - u^d||  (eq. 12)

    alpha is minimized (-1) at zero position error, with a sharp tanh bonus
    near zero on top of a linear penalty. We average over the trajectory.
    """
    r_err = jnp.linalg.norm(out.rs[:-1] - out.rs_ref[:-1], axis=1)  # (T,)
    v_err = jnp.linalg.norm(out.vs[:-1] - out.vs_ref[:-1], axis=1)
    u_err = jnp.linalg.norm(out.us - out.us_ref, axis=1)

    alpha = coeffs.c_r * r_err + jnp.tanh(coeffs.a_r * r_err) - 1.0
    cost = alpha + coeffs.c_v * v_err + coeffs.c_u * u_err
    return jnp.mean(cost)


def sample_initial_states(key, batch, pos_std, vel_std):
    """Reference starts at rest at the origin; primary starts perturbed.

    The perturbation gives the policy a nonzero initial tracking error to
    correct (the paper trains from a randomly sampled initial state).
    """
    kp, kv = jax.random.split(key)
    r0 = pos_std * jax.random.normal(kp, (batch, 2))
    v0 = vel_std * jax.random.normal(kv, (batch, 2))
    ref_r0 = jnp.zeros((batch, 2))
    ref_v0 = jnp.zeros((batch, 2))
    return (r0, v0), (ref_r0, ref_v0)


def batched_loss(policy, key, m, dt, sigma, T, coeffs, batch, pos_std, vel_std):
    """Mean tracking cost over a batch of freshly sampled rollouts."""
    init_key, roll_key = jax.random.split(key)
    state0, ref_state0 = sample_initial_states(init_key, batch, pos_std, vel_std)
    roll_keys = jax.random.split(roll_key, batch)

    def one(s0, r0, k):
        out = rollout_joint(policy, s0, r0, k, m, dt, sigma, T)
        return tracking_cost(out, coeffs)

    costs = jax.vmap(one, in_axes=((0, 0), (0, 0), 0))(state0, ref_state0, roll_keys)
    return jnp.mean(costs)


@eqx.filter_jit
def train_step(
    policy, opt_state, optim, key, m, dt, sigma, T, coeffs, batch, pos_std, vel_std
):
    loss, grads = eqx.filter_value_and_grad(batched_loss)(
        policy, key, m, dt, sigma, T, coeffs, batch, pos_std, vel_std
    )
    updates, opt_state = optim.update(grads, opt_state, policy)
    policy = eqx.apply_updates(policy, updates)
    return policy, opt_state, loss


def report_kernel_memory(name, train_step_fn, args):
    """Compile the training kernel and print its XLA memory analysis."""
    ma = train_step_fn.lower(*args).compile().compiled.memory_analysis()
    print(
        f"[{name}] temp={ma.temp_size_in_bytes} bytes  "
        f"output={ma.output_size_in_bytes / 1e6:.3f}MB  "
        f"args={ma.argument_size_in_bytes / 1e6:.3f}MB  "
        f"code={ma.generated_code_size_in_bytes / 1e6:.3f}MB"
    )
    return ma


def train(policy, key, m, dt, sigma, T, coeffs, iters, batch, pos_std, vel_std, lr):
    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(policy, eqx.is_array))

    losses = []
    for i in range(iters):
        key, step_key = jax.random.split(key)
        policy, opt_state, loss = train_step(
            policy,
            opt_state,
            optim,
            step_key,
            m,
            dt,
            sigma,
            T,
            coeffs,
            batch,
            pos_std,
            vel_std,
        )
        losses.append(float(loss))
        if i % 100 == 0 or i == iters - 1:
            print(f"iter {i:5d}  loss {float(loss):+.4f}")
    return policy, jnp.array(losses)


def save_trajectory_gif(
    out: Rollout, dt, path, title="trained policy tracking", stride=2
):
    """Animate the reference and primary trajectories and save as a GIF."""
    rs, rs_ref = out.rs, out.rs_ref
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


def evaluate(policy, key, m, dt, sigma, T, pos_std, vel_std):
    """Roll out the policy once from a random offset start; return it."""
    ei, er = jax.random.split(key)
    (r0, v0), (ref_r0, ref_v0) = sample_initial_states(ei, 1, pos_std, vel_std)
    return rollout_joint(
        policy, (r0[0], v0[0]), (ref_r0[0], ref_v0[0]), er, m, dt, sigma, T
    )


def main():
    m = 1.0
    dt = 0.05
    T = 200
    sigma = 2.0
    coeffs = Coeffs()
    pos_std, vel_std = 1.0, 0.5

    key = jax.random.PRNGKey(0)
    init_key, train_key, eval_key = jax.random.split(key, 3)
    full_key, reduced_key = jax.random.split(init_key)

    # Memory footprint of the full-dynamics training kernel (both primary and
    # reference simulated jointly, backprop through the 200-step scan).
    _p = FullStatePolicy(full_key)
    _optim = optax.adam(1e-3)
    _os = _optim.init(eqx.filter(_p, eqx.is_array))
    report_kernel_memory(
        "full-dynamics train_step",
        train_step,
        (_p, _os, _optim, train_key, m, dt, sigma, T, coeffs, 64, pos_std, vel_std),
    )

    # Same train_key and eval_key for both so batches/references match -> fair comparison.
    variants = {
        "full-state (baseline)": FullStatePolicy(full_key),
        "reduced (pos+vel error)": ReducedPolicy(reduced_key),
    }

    results = {}
    for name, policy in variants.items():
        print(f"\n=== training {name} ===")
        trained, losses = train(
            policy,
            train_key,
            m,
            dt,
            sigma,
            T,
            coeffs,
            iters=1500,
            batch=64,
            pos_std=pos_std,
            vel_std=vel_std,
            lr=1e-3,
        )
        out = evaluate(trained, eval_key, m, dt, sigma, T, pos_std, vel_std)
        pos_err = jnp.linalg.norm(out.rs - out.rs_ref, axis=1)
        print(
            f"eval: initial pos error = {pos_err[0]:.3f}, final = {pos_err[-1]:.3f}, "
            f"mean = {pos_err.mean():.3f}"
        )
        results[name] = (losses, out)

    # Comparison of training curves.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (name, (losses, _)), color in zip(results.items(), ["tab:gray", "tab:green"]):
        ax.plot(losses, color=color, lw=1, label=name)
    ax.set_xlabel("training iteration")
    ax.set_ylabel("mean tracking cost")
    ax.set_title("policy-gradient training: full-state vs reduced observation")
    ax.legend()
    fig.tight_layout()
    fig.savefig("training_curve.png", dpi=120)
    print("\nsaved plot to training_curve.png")

    for name, fname in [
        ("full-state (baseline)", "trajectory_full.gif"),
        ("reduced (pos+vel error)", "trajectory_reduced.gif"),
    ]:
        _, out = results[name]
        save_trajectory_gif(out, dt, fname, title=f"{name} tracking")


if __name__ == "__main__":
    main()
