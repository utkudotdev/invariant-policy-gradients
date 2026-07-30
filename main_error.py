"""Train the same tracking controller directly in the *error dynamics*.

The paper's reduction (arXiv:2409.11238, eq. 43-50) shows the Particle's
tracking MDP is homomorphic to a lower-dimensional "error" MDP. Writing the
error state e = (r^e, v^e) = (r - r^d, v - v^d) and the reduced action
u^e = u - u^d, the reference action u^d cancels out of the error dynamics:

    r^e_{t+1} = r^e_t + v^e_t dt
    v^e_{t+1} = v^e_t + (1/m) u^e_t dt                              (eq. 49)

so the reduced MDP is *deterministic* -- no reference to simulate, no u^d to
sample. The reduced cost (eq. 46-47) depends only on the error and effort:

    cost(e, u^e) = alpha(r^e) + c_v ||v^e|| + c_u ||u^e||

We train a policy pi_tilde: (r^e, v^e) -> u^e purely in this frame and compare
its training-kernel memory footprint against the full-dynamics version in
main.py, to see whether the reduction buys anything beyond faster convergence.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from main import (
    Coeffs,
    Rollout,
    f,
    report_kernel_memory,
    sample_reference_action,
    save_trajectory_gif,
)


class ErrorPolicy(eqx.Module):
    """MLP policy on the pure error observation (r^e, v^e) -> u^e.

    This is the full TR^3 x R^3 reduction: the policy sees only the 4-dim
    error and outputs the reduced action u^e directly (no u^d in the loop).
    """

    mlp: eqx.nn.MLP

    def __init__(self, key, width=32, depth=2):
        self.mlp = eqx.nn.MLP(
            in_size=4, out_size=2, width_size=width, depth=depth, key=key
        )

    def __call__(self, error_state):
        r_e, v_e = error_state
        return self.mlp(jnp.concatenate([r_e, v_e]))


class ErrorRollout(eqx.Module):
    """An error-frame trajectory. res/ves: (T+1, 2); ues: (T, 2)."""

    res: jax.Array
    ves: jax.Array
    ues: jax.Array


def rollout_error(policy, error0, m, dt, T):
    """Deterministically roll out the error dynamics under the policy."""

    def step(estate, _):
        u_e = policy(estate)
        next_e = f(estate, u_e, m, dt)
        return next_e, (next_e, u_e)

    _, (estates, ues) = jax.lax.scan(step, error0, None, length=T)
    res, ves = estates
    re0, ve0 = error0
    return ErrorRollout(
        res=jnp.concatenate([re0[None], res], axis=0),
        ves=jnp.concatenate([ve0[None], ves], axis=0),
        ues=ues,
    )


def reduced_cost(out: ErrorRollout, coeffs: Coeffs):
    """Mean reduced tracking cost (paper eq. 46-47)."""
    r_err = jnp.linalg.norm(out.res[:-1], axis=1)
    v_err = jnp.linalg.norm(out.ves[:-1], axis=1)
    u_eff = jnp.linalg.norm(out.ues, axis=1)

    alpha = coeffs.c_r * r_err + jnp.tanh(coeffs.a_r * r_err) - 1.0
    cost = alpha + coeffs.c_v * v_err + coeffs.c_u * u_eff
    return jnp.mean(cost)


def sample_initial_errors(key, batch, pos_std, vel_std):
    """Initial error = primary offset from a reference at rest at the origin,
    matching the full-dynamics setup so convergence is directly comparable."""
    kp, kv = jax.random.split(key)
    re0 = pos_std * jax.random.normal(kp, (batch, 2))
    ve0 = vel_std * jax.random.normal(kv, (batch, 2))
    return re0, ve0


def batched_loss(policy, key, m, dt, T, coeffs, batch, pos_std, vel_std):
    re0, ve0 = sample_initial_errors(key, batch, pos_std, vel_std)

    def one(re, ve):
        return reduced_cost(rollout_error(policy, (re, ve), m, dt, T), coeffs)

    return jnp.mean(jax.vmap(one)(re0, ve0))


@eqx.filter_jit
def train_step(
    policy, opt_state, optim, key, m, dt, T, coeffs, batch, pos_std, vel_std
):
    loss, grads = eqx.filter_value_and_grad(batched_loss)(
        policy, key, m, dt, T, coeffs, batch, pos_std, vel_std
    )
    updates, opt_state = optim.update(grads, opt_state, policy)
    policy = eqx.apply_updates(policy, updates)
    return policy, opt_state, loss


def train(policy, key, m, dt, T, coeffs, iters, batch, pos_std, vel_std, lr):
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


def reconstruct_rollout(err_out: ErrorRollout, key, m, dt, sigma):
    """Rebuild an absolute (reference, primary) rollout for visualization.

    Because the error dynamics are independent of u^d, we can pair the trained
    error trajectory with *any* reference: sample u^d, integrate the reference
    from the origin, then set r = r^d + r^e (and v = v^d + v^e). The primary
    thus starts offset by the initial error and converges as the error decays.
    """
    T = err_out.ues.shape[0]
    keys = jax.random.split(key, T)
    u_refs = jax.vmap(lambda k: sample_reference_action(k, sigma))(keys)

    def step(state, u_ref):
        next_state = f(state, u_ref, m, dt)
        return next_state, next_state

    _, (rs_ref, vs_ref) = jax.lax.scan(step, (jnp.zeros(2), jnp.zeros(2)), u_refs)
    rs_ref = jnp.concatenate([jnp.zeros((1, 2)), rs_ref], axis=0)
    vs_ref = jnp.concatenate([jnp.zeros((1, 2)), vs_ref], axis=0)

    return Rollout(
        rs=rs_ref + err_out.res,
        vs=vs_ref + err_out.ves,
        rs_ref=rs_ref,
        vs_ref=vs_ref,
        us=err_out.ues + u_refs,
        us_ref=u_refs,
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

    policy = ErrorPolicy(init_key)

    # Memory footprint of the error-dynamics training kernel.
    _optim = optax.adam(1e-3)
    _os = _optim.init(eqx.filter(policy, eqx.is_array))
    report_kernel_memory(
        "error-dynamics train_step",
        train_step,
        (policy, _os, _optim, train_key, m, dt, T, coeffs, 64, pos_std, vel_std),
    )

    print("\n=== training error-dynamics policy ===")
    policy, losses = train(
        policy,
        train_key,
        m,
        dt,
        T,
        coeffs,
        iters=1500,
        batch=64,
        pos_std=pos_std,
        vel_std=vel_std,
        lr=1e-3,
    )

    # Evaluate from a single random initial error.
    ei, er = jax.random.split(eval_key)
    re0, ve0 = sample_initial_errors(ei, 1, pos_std, vel_std)
    err_out = rollout_error(policy, (re0[0], ve0[0]), m, dt, T)
    err_norm = jnp.linalg.norm(err_out.res, axis=1)
    print(
        f"eval: initial pos error = {err_norm[0]:.3f}, final = {err_norm[-1]:.3f}, "
        f"mean = {err_norm.mean():.3f}"
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(losses, color="tab:green", lw=1)
    ax.set_xlabel("training iteration")
    ax.set_ylabel("mean reduced tracking cost")
    ax.set_title("policy-gradient training in the error dynamics")
    fig.tight_layout()
    fig.savefig("training_curve_error.png", dpi=120)
    print("saved plot to training_curve_error.png")

    recon = reconstruct_rollout(err_out, er, m, dt, sigma)
    save_trajectory_gif(
        recon, dt, "trajectory_error.gif", title="error-dynamics policy tracking"
    )


if __name__ == "__main__":
    main()
