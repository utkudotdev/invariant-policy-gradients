"""Environment-agnostic rollout and policy-gradient training utilities."""

from dataclasses import dataclass
from types import ModuleType

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping
import optax
import tqdm
from jaxtyping import Float

from environments import Policy, Rollout


@jax.tree_util.register_dataclass
@dataclass
class TrainingParams:
    T: int
    iters: int
    batch: int
    lr: float


def rollout_joint(
    env: ModuleType,
    policy: Policy,
    s0,
    r0,
    key: jaxtyping.Key,
    dynamics_params,
    dt: float,
    env_params,
    T: int,
) -> Rollout:
    """Roll out primary and reference dynamics in an environment's joint state."""
    reduced0 = env.get_reduced_state(s0, r0)

    def step(reduced_t, key_t):
        u_ref = env.sample_reference_action(key_t, env_params)
        u = policy(env.get_observation(reduced_t), u_ref)
        next_reduced = env.f_joint(reduced_t, u, u_ref, dynamics_params, dt)
        return next_reduced, (next_reduced, u, u_ref)

    keys = jax.random.split(key, T)
    _, (reduced_states, us, us_ref) = jax.lax.scan(step, reduced0, keys)
    states, ref_states = jax.vmap(env.lift_reduced_state)(
        jnp.concatenate([reduced0[None], reduced_states])
    )
    return Rollout(s=states, s_ref=ref_states, us=us, us_ref=us_ref)


def batched_loss(
    env: ModuleType,
    policy: Policy,
    key: jaxtyping.Key,
    dynamics_params,
    dt: float,
    env_params,
    train_params: TrainingParams,
):
    """Mean tracking cost over a batch of freshly sampled rollouts."""
    init_key, roll_key = jax.random.split(key)
    state0, ref_state0 = env.sample_initial_states(
        init_key, train_params.batch, env_params
    )
    roll_keys = jax.random.split(roll_key, train_params.batch)

    def one(s0, r0, rollout_key):
        out = rollout_joint(
            env,
            policy,
            s0,
            r0,
            rollout_key,
            dynamics_params,
            dt,
            env_params,
            train_params.T,
        )
        return env.cost(out, env_params)

    return jnp.mean(jax.vmap(one)(state0, ref_state0, roll_keys))


def make_train_step(env: ModuleType):
    """Build a compiled training step with the environment bound statically."""

    @eqx.filter_jit
    def train_step(
        policy: Policy,
        opt_state: optax.OptState,
        optim,
        key: jaxtyping.Key,
        dynamics_params,
        dt: float,
        env_params,
        train_params: TrainingParams,
    ):
        def loss_fn(policy):
            return batched_loss(
                env, policy, key, dynamics_params, dt, env_params, train_params
            )

        loss, grads = eqx.filter_value_and_grad(loss_fn)(policy)
        updates, next_opt_state = optim.update(grads, opt_state, policy)
        return eqx.apply_updates(policy, updates), next_opt_state, loss

    return train_step


def train(
    env: ModuleType,
    policy: Policy,
    key: jaxtyping.Key,
    dynamics_params,
    dt: float,
    env_params,
    train_params: TrainingParams,
) -> tuple[Policy, Float[jax.Array, " iters"]]:
    optim = optax.adam(train_params.lr)
    opt_state = optim.init(eqx.filter(policy, eqx.is_array))
    train_step = make_train_step(env)

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


class MLPPolicy(eqx.Module):
    """Bounded MLP policy sized for a supplied environment module."""

    mlp: eqx.nn.MLP
    limits: jax.Array

    def __init__(self, env: ModuleType, key, width: int = 64, depth: int = 2):
        self.mlp = eqx.nn.MLP(
            in_size=env.OBS_DIM + env.CONTROL_DIM,
            out_size=env.CONTROL_DIM,
            width_size=width,
            depth=depth,
            key=key,
        )
        self.limits = env.control_limits()

    def __call__(self, state_obs, u_ref):
        return self.limits * jnp.tanh(self.mlp(jnp.concatenate([state_obs, u_ref])))


def compile_memory(train_step, args):
    """Compile a training step and return its XLA memory analysis."""
    return train_step.lower(*args).compile().compiled.memory_analysis()
