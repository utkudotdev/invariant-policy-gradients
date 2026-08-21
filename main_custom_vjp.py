from dataclasses import dataclass
import functools as ft
from typing import Callable, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float
import matplotlib.pyplot as plt
import optax
from matplotlib import animation
import tqdm
import jaxtyping


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

Observation = Float[jax.Array, "4"]


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
    sigma: float
    pos_std: float
    vel_std: float


@jax.tree_util.register_dataclass
@dataclass
class TrainingParams:
    cost_coeffs: CostCoeffs
    T: int
    iters: int
    batch: int
    lr: float


Policy = Callable[[Observation, Control], Control]


@jax.tree_util.register_dataclass
@dataclass
class Rollout:
    # not supported by jaxtyping yet, but these states are batched
    s: State
    s_ref: State
    us: Float[Control, "steps"]
    us_ref: Float[Control, "steps"]


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


def sample_reference_action(key, sigma):
    """Draw a reference action u^d ~ N(0, sigma^2 I).

    The paper (arXiv:2409.11238, eq. 10-11c) models the reference action
    distribution rho for the Particle as an isotropic Gaussian N(0, Sigma).
    """
    return sigma * jax.random.normal(key, (2,))


def get_observation(state: State, state_ref: State) -> Observation:
    return jnp.concatenate([state.r - state_ref.r, state.v - state_ref.v])


def lift_observation(obs: Observation) -> tuple[State, State]:
    return State(r=obs[:2], v=obs[2:]), State(r=jnp.zeros(2), v=jnp.zeros(2))


# TODO: in theory i think we have linear output tangents so this could be done using custom_jvp and
# letting jax handle the transposition, but I don't know if this will do what we expect in practice.
@jax.custom_vjp
def f_joint(
    obs: Observation, u: Control, u_ref: Control, params: DynamicsParams, dt: float
) -> Observation:
    state, state_ref = lift_observation(obs)
    new_state = f(state, u, params, dt)
    new_state_ref = f(state_ref, u_ref, params, dt)
    return get_observation(new_state, new_state_ref)


def f_joint_fwd(
    obs: Observation, u: Control, u_ref: Control, params: DynamicsParams, dt: float
):
    return f_joint(obs, u, u_ref, params, dt), (params, dt)


def f_joint_bwd(res: tuple[DynamicsParams, float], g: Observation):
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


def rollout_joint(
    policy: Policy,
    s0: State,
    r0: State,
    key: jaxtyping.Key,
    dynamics_params: DynamicsParams,
    dt: float,
    sigma: float,
    T: int,
) -> Rollout:
    obs0 = get_observation(s0, r0)

    def step(obs_t, key_t):
        u_ref = sample_reference_action(key_t, sigma)
        u = policy(obs_t, u_ref)
        next_obs = f_joint(obs_t, u, u_ref, dynamics_params, dt)
        return next_obs, (next_obs, u, u_ref)

    keys = jax.random.split(key, T)
    _, (observations, us, us_ref) = jax.lax.scan(step, obs0, keys)

    states, ref_states = jax.vmap(lift_observation)(
        jnp.concatenate([jnp.expand_dims(obs0, axis=0), observations])
    )

    return Rollout(
        s=states,
        s_ref=ref_states,
        us=us,
        us_ref=us_ref,
    )


def sample_initial_states(key, batch, pos_std, vel_std) -> tuple[State, State]:
    """Reference starts at rest at the origin; primary starts perturbed.

    The perturbation gives the policy a nonzero initial tracking error to
    correct (the paper trains from a randomly sampled initial state).
    """
    kp, kv = jax.random.split(key)
    r0 = pos_std * jax.random.normal(kp, (batch, 2))
    v0 = vel_std * jax.random.normal(kv, (batch, 2))
    ref_r0 = jnp.zeros((batch, 2))
    ref_v0 = jnp.zeros((batch, 2))
    return State(r0, v0), State(ref_r0, ref_v0)


def tracking_cost(out: Rollout, coeffs: CostCoeffs):
    r_err = jnp.linalg.norm(out.s.r[:-1] - out.s_ref.r[:-1], axis=1)
    v_err = jnp.linalg.norm(out.s.v[:-1] - out.s_ref.v[:-1], axis=1)
    u_err = jnp.linalg.norm(out.us - out.us_ref, axis=1)

    alpha = coeffs.c_r * r_err + jnp.tanh(coeffs.a_r * r_err) - 1.0
    cost = alpha + coeffs.c_v * v_err + coeffs.c_u * u_err
    return jnp.mean(cost)


def batched_loss(
    policy: Policy,
    key: jaxtyping.Key,
    dynamics_params: DynamicsParams,
    dt: float,
    env_params: EnvParams,
    train_params: TrainingParams,
):
    """Mean tracking cost over a batch of freshly sampled rollouts."""
    init_key, roll_key = jax.random.split(key)
    state0, ref_state0 = sample_initial_states(
        init_key, train_params.batch, env_params.pos_std, env_params.vel_std
    )
    roll_keys = jax.random.split(roll_key, train_params.batch)

    def one(s0, r0, key):
        out = rollout_joint(
            policy,
            s0,
            r0,
            key,
            dynamics_params,
            dt,
            env_params.sigma,
            train_params.T,
        )
        return tracking_cost(out, train_params.cost_coeffs)

    costs = jax.vmap(one, in_axes=(0, 0, 0))(state0, ref_state0, roll_keys)
    return jnp.mean(costs)


@eqx.filter_jit
def train_step(
    policy: Policy,
    opt_state: optax.OptState,
    optim,
    key: jaxtyping.Key,
    dynamics_params: DynamicsParams,
    dt: float,
    env_params: EnvParams,
    train_params: TrainingParams,
):
    loss, grads = eqx.filter_value_and_grad(batched_loss)(
        policy, key, dynamics_params, dt, env_params, train_params
    )
    updates, opt_state = optim.update(grads, opt_state, policy)
    policy = eqx.apply_updates(policy, updates)
    return policy, opt_state, loss


def train(
    policy: Policy,
    key: jaxtyping.Key,
    dynamics_params: DynamicsParams,
    dt: float,
    env_params: EnvParams,
    train_params: TrainingParams,
) -> tuple[Policy, Float[jax.Array, "{train_params.iters}"]]:
    optim = optax.adam(train_params.lr)
    opt_state = optim.init(eqx.filter(policy, eqx.is_array))

    losses = []
    for i in tqdm.trange(train_params.iters):
        key, step_key = jax.random.split(key)
        policy, opt_state, loss = train_step(
            policy,
            opt_state,
            optim,
            step_key,
            dynamics_params,
            dt,
            env_params,
            train_params,
        )
        losses.append(float(loss))
        if i % 100 == 0 or i == train_params.iters - 1:
            tqdm.tqdm.write(f"iter {i:5d}  loss {float(loss):+.4f}")

    return policy, jnp.array(losses)


def batched_pytree_get_first(x):
    return jax.tree.map(lambda y: y[0], x)


def batched_pytree_prepend(x, y):
    return jax.tree.map(lambda a, b: jnp.concatenate([a[None], b], axis=0), x, y)


def rollout_eval(
    policy: Policy,
    s0: State,
    r0: State,
    key: jaxtyping.Key,
    dynamics_params: DynamicsParams,
    dt: float,
    sigma: float,
    T: int,
) -> Rollout:
    def step(carry, key_t):
        state, ref_state = carry

        u_ref = sample_reference_action(key_t, sigma)
        obs = get_observation(state, ref_state)
        u = policy(obs, u_ref)

        next_state = f(state, u, dynamics_params, dt)
        next_ref = f(ref_state, u_ref, dynamics_params, dt)
        return (next_state, next_ref), (next_state, next_ref, u, u_ref)

    keys = jax.random.split(key, T)
    _, (states, ref_states, us, us_ref) = jax.lax.scan(step, (s0, r0), keys)

    return Rollout(
        s=batched_pytree_prepend(s0, states),
        s_ref=batched_pytree_prepend(r0, ref_states),
        us=us,
        us_ref=us_ref,
    )


def evaluate(
    policy: Policy,
    key: jaxtyping.Key,
    dynamics_params: DynamicsParams,
    dt: float,
    env_params: EnvParams,
    T: int,
) -> Rollout:
    sample_key, rollout_key = jax.random.split(key)
    s0, r0 = sample_initial_states(
        sample_key, 1, env_params.pos_std, env_params.vel_std
    )
    s0, r0 = batched_pytree_get_first(s0), batched_pytree_get_first(r0)

    out = rollout_eval(
        policy, s0, r0, rollout_key, dynamics_params, dt, env_params.sigma, T
    )

    return out


class ReducedPolicy(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key, width=32, depth=2):
        self.mlp = eqx.nn.MLP(
            in_size=6, out_size=2, width_size=width, depth=depth, key=key
        )

    def __call__(self, state_obs: Observation, u_ref: Control):
        obs = jnp.concatenate([state_obs, u_ref])
        return self.mlp(obs)


def save_trajectory_gif(
    out: Rollout, dt, path, title="trained policy tracking", stride=2
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


def report_kernel_memory(name, train_step_fn, args):
    """Compile the training kernel and print its XLA memory analysis."""
    ma = train_step_fn.lower(*args).compile().compiled.memory_analysis()
    print(
        f"[{name}] temp={ma.temp_size_in_bytes / 1e6:.3f}MB  "
        f"output={ma.output_size_in_bytes / 1e6:.3f}MB  "
        f"args={ma.argument_size_in_bytes / 1e6:.3f}MB  "
        f"code={ma.generated_code_size_in_bytes / 1e6:.3f}MB"
    )
    return ma


def main():
    key = jax.random.key(0)
    init_key, train_key, eval_key = jax.random.split(key, 3)

    dynamics_params = DynamicsParams(m=1.0)
    env_params = EnvParams(
        sigma=2.0,
        pos_std=1.0,
        vel_std=0.5,
    )
    dt = 0.05
    train_params = TrainingParams(
        cost_coeffs=CostCoeffs(c_r=1.0, a_r=5.0, c_v=0.5, c_u=0.1),
        T=200,
        iters=1500,
        batch=64,
        lr=1e-3,
    )
    policy = ReducedPolicy(init_key)

    temp_optim = optax.adam(1e-3)
    temp_optim_state = temp_optim.init(eqx.filter(policy, eqx.is_array))
    report_kernel_memory(
        "reduced dynamics",
        train_step,
        (
            policy,
            temp_optim_state,
            temp_optim,
            train_key,
            dynamics_params,
            dt,
            env_params,
            train_params,
        ),
    )

    trained, losses = train(
        policy,
        train_key,
        dynamics_params,
        dt,
        env_params,
        train_params,
    )
    out = evaluate(trained, eval_key, dynamics_params, dt, env_params, T=200)
    pos_err = jnp.linalg.norm(out.s.r - out.s_ref.r, axis=1)
    print(
        f"eval: initial pos error = {pos_err[0]:.3f}, final = {pos_err[-1]:.3f}, "
        f"mean = {pos_err.mean():.3f}"
    )

    # Comparison of training curves.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(losses, color="tab:gray", lw=1, label="reduced")
    ax.set_xlabel("training iteration")
    ax.set_ylabel("mean tracking cost")
    ax.set_title("policy-gradient training: full-state vs reduced observation")
    ax.legend()
    fig.tight_layout()
    fig.savefig("training_curve_custom.png", dpi=120)
    print("saved plot to training_curve.png")

    save_trajectory_gif(
        out, dt, "trajectory_reduced_custom.gif", title="reduced tracking"
    )


if __name__ == "__main__":
    main()
