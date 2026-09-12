"""Environment-agnostic policy evaluation utilities."""

from types import ModuleType

import jax
import jaxtyping

from environments import (
    Policy,
    Rollout,
    batched_pytree_get_first,
    batched_pytree_prepend,
)


def rollout_eval(
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
    """Roll out absolute states, avoiding lift/reduce round trips during evaluation."""

    def step(carry, key_t):
        state, ref_state = carry
        u_ref = env.sample_reference_action(key_t, env_params)
        u = policy(env.get_observation(env.get_reduced_state(state, ref_state)), u_ref)
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


def evaluate(
    env: ModuleType,
    policy: Policy,
    key: jaxtyping.Key,
    dynamics_params,
    dt: float,
    env_params,
    T: int,
) -> Rollout:
    """Evaluate one rollout from a freshly sampled initial state."""
    sample_key, rollout_key = jax.random.split(key)
    s0, r0 = env.sample_initial_states(sample_key, 1, env_params)
    return rollout_eval(
        env,
        policy,
        batched_pytree_get_first(s0),
        batched_pytree_get_first(r0),
        rollout_key,
        dynamics_params,
        dt,
        env_params,
        T,
    )
