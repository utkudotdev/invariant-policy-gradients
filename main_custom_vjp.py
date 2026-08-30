from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping
import matplotlib.pyplot as plt
import optax
import tqdm
from jaxtyping import Float

from environments import (
    Policy,
    Rollout,
    batched_pytree_get_first,
    batched_pytree_prepend,
)
from environments import astrobee as env

State = env.State
Control = env.Control
ReducedState = env.ReducedState
Observation = env.Observation
DynamicsParams = env.DynamicsParams
EnvParams = env.EnvParams


@jax.tree_util.register_dataclass
@dataclass
class TrainingParams:
    T: int
    iters: int
    batch: int
    lr: float


def rollout_joint(
    policy: Policy,
    s0: State,
    r0: State,
    key: jaxtyping.Key,
    dynamics_params: DynamicsParams,
    dt: float,
    env_params: EnvParams,
    T: int,
) -> Rollout[State, Control]:
    reduced0 = env.get_reduced_state(s0, r0)

    def step(reduced_t, key_t):
        u_ref = env.sample_reference_action(key_t, env_params)
        u = policy(env.get_observation(reduced_t), u_ref)
        next_reduced = env.f_joint(reduced_t, u, u_ref, dynamics_params, dt)
        return next_reduced, (next_reduced, u, u_ref)

    keys = jax.random.split(key, T)
    _, (reduced_states, us, us_ref) = jax.lax.scan(step, reduced0, keys)

    states, ref_states = jax.vmap(env.lift_reduced_state)(
        jnp.concatenate([jnp.expand_dims(reduced0, axis=0), reduced_states])
    )

    return Rollout(
        s=states,
        s_ref=ref_states,
        us=us,
        us_ref=us_ref,
    )


def rollout_eval(
    policy: Policy,
    s0: State,
    r0: State,
    key: jaxtyping.Key,
    dynamics_params: DynamicsParams,
    dt: float,
    env_params: EnvParams,
    T: int,
) -> Rollout[State, Control]:
    def step(carry, key_t):
        state, ref_state = carry

        u_ref = env.sample_reference_action(key_t, env_params)
        obs = env.get_observation(env.get_reduced_state(state, ref_state))
        u = policy(obs, u_ref)

        next_state = env.f(state, u, dynamics_params, dt)
        next_ref = env.f(ref_state, u_ref, dynamics_params, dt)
        return (next_state, next_ref), (next_state, next_ref, u, u_ref)

    keys = jax.random.split(key, T)
    _, (states, ref_states, us, us_ref) = jax.lax.scan(step, (s0, r0), keys)

    return Rollout(
        s=batched_pytree_prepend(s0, states),
        s_ref=batched_pytree_prepend(r0, ref_states),
        us=us,
        us_ref=us_ref,
    )


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
    state0, ref_state0 = env.sample_initial_states(
        init_key, train_params.batch, env_params
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
            env_params,
            train_params.T,
        )
        return env.cost(out, env_params)

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


def evaluate(
    policy: Policy,
    key: jaxtyping.Key,
    dynamics_params: DynamicsParams,
    dt: float,
    env_params: EnvParams,
    T: int,
) -> Rollout[State, Control]:
    sample_key, rollout_key = jax.random.split(key)
    s0, r0 = env.sample_initial_states(sample_key, 1, env_params)
    s0, r0 = batched_pytree_get_first(s0), batched_pytree_get_first(r0)

    return rollout_eval(policy, s0, r0, rollout_key, dynamics_params, dt, env_params, T)


class MLPPolicy(eqx.Module):
    """Policy on the observation (`env.get_observation` of the reduced state
    p(s)), plus the reference action.

    The output is squashed through the Astrobee actuation limits; without a bound
    the untrained policy drives ||omega|| high enough that the explicit Euler step
    on the Euler equations diverges mid-rollout.
    """

    mlp: eqx.nn.MLP
    limits: jax.Array

    def __init__(self, key, width=64, depth=2):
        self.mlp = eqx.nn.MLP(
            in_size=env.OBS_DIM + env.CONTROL_DIM,
            out_size=env.CONTROL_DIM,
            width_size=width,
            depth=depth,
            key=key,
        )
        self.limits = env.control_limits()

    def __call__(self, state_obs: Observation, u_ref: Control):
        obs = jnp.concatenate([state_obs, u_ref])
        return self.limits * jnp.tanh(self.mlp(obs))


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


def main():
    key = jax.random.key(0)
    init_key, train_key, eval_key = jax.random.split(key, 3)

    dynamics_params = env.default_dynamics_params()
    env_params = env.EnvParams(
        cost_coeffs=env.CostCoeffs(c_r=1.0, a_r=5.0, c_R=1.0, c_xi=0.5, c_u=0.1),
        sigma_torque=0.03,
        sigma_force=0.3,
        pos_std=0.5,
        att_std=0.3,
        vel_std=0.1,
        omega_std=0.1,
    )
    dt = 0.05
    train_params = TrainingParams(
        T=200,
        iters=1500,
        batch=64,
        lr=1e-3,
    )
    policy = MLPPolicy(init_key)

    temp_optim = optax.adam(1e-3)
    temp_optim_state = temp_optim.init(eqx.filter(policy, eqx.is_array))
    report_kernel_memory(
        "astrobee (SE(3) quotient)",
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
    out = evaluate(trained, eval_key, dynamics_params, dt, env_params, T=train_params.T)
    pos_err = env.tracking_error(out)
    att_err = env.attitude_error(out)
    print(
        f"eval: initial pos error = {pos_err[0]:.3f} m, final = {pos_err[-1]:.3f} m, "
        f"mean = {pos_err.mean():.3f} m"
    )
    print(
        f"eval: initial att error = {att_err[0]:.3f} rad, final = {att_err[-1]:.3f} rad, "
        f"mean = {att_err.mean():.3f} rad"
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(losses, color="tab:gray", lw=1)
    ax.set_xlabel("training iteration")
    ax.set_ylabel("mean tracking cost")
    ax.set_title("astrobee policy-gradient training (SE(3)-reduced observation)")
    fig.tight_layout()
    fig.savefig("training_curve_astrobee.png", dpi=120)
    print("saved plot to training_curve_astrobee.png")

    env.save_trajectory_gif(
        out, dt, "trajectory_astrobee.gif", title="astrobee tracking (reduced)"
    )


if __name__ == "__main__":
    main()
