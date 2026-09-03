"""Do the Particle's reduced dynamics store anything per step?

`f_joint` for the Particle is *linear* in (reduced, u, u^d), with coefficients
built only from m and dt:

    r^e_{t+1} = r^e_t + v^e_t dt
    v^e_{t+1} = v^e_t + (u_t - u^d_t) dt / m

The VJP of a linear map does not depend on where it was evaluated, so nothing
about step t is needed to transpose step t. The module's custom rule says as
much -- its residual is `(params, dt)` and nothing else. The question this
benchmark asks is whether autodiff works that out on its own, i.e. whether the
compiled temp memory is flat in T for both.

Run with:  uv run python -m benchmarks.particle_residuals
"""

import jax
import jax.numpy as jnp

from environments import particle as env

BATCH = 2048
DT = 0.05
PARAMS = env.DynamicsParams(m=1.0)
WEIGHTS = jnp.array([1.0, 2.0, -1.0, 0.5])


def autodiff_f_joint(reduced, u, u_ref, params, dt):
    """`env.f_joint`'s body, left for autodiff to transpose."""
    state, state_ref = env.lift_reduced_state(reduced)
    return env.get_reduced_state(
        env.f(state, u, params, dt), env.f(state_ref, u_ref, params, dt)
    )


def grad_fn(step):
    """d/d(reduced_0, us, us_ref) of a scan of T reduced steps, batched."""

    def loss(reduced_0, us, us_ref):
        def body(reduced, controls):
            u, u_ref = controls
            return step(reduced, u, u_ref, PARAMS, DT), None

        reduced, _ = jax.lax.scan(body, reduced_0, (us, us_ref))
        return jnp.sum(WEIGHTS * reduced)

    return jax.jit(jax.vmap(jax.grad(loss, argnums=(0, 1, 2))))


def inputs(T: int, batch: int = BATCH):
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    return (
        jax.random.normal(k0, (batch, 4)),
        0.5 * jax.random.normal(k1, (batch, T, 2)),
        0.5 * jax.random.normal(k2, (batch, T, 2)),
    )


def temp_bytes(fn, T: int) -> int:
    return fn.lower(*inputs(T)).compile().memory_analysis().temp_size_in_bytes


def main():
    print(f"backend: {jax.default_backend()}  {jax.devices()}")
    print(f"batch={BATCH}, float32\n")

    lengths = (100, 200, 400, 800)
    variants = [("custom VJP (module)", env.f_joint), ("autodiff", autodiff_f_joint)]

    # the two rules must agree before their memory is worth comparing
    args = inputs(50, batch=4)
    grads = []
    for name, step in variants:
        try:
            grads.append(jax.block_until_ready(grad_fn(step)(*args)))
            print(f"  {name:<22} gradient computed")
        except Exception as exc:  # noqa: BLE001 - reporting, not handling
            grads.append(None)
            print(f"  {name:<22} FAILED: {type(exc).__name__}: {str(exc)[:140]}")
    if all(g is not None for g in grads):
        drift = max(
            float(jnp.max(jnp.abs(a - b)))
            for a, b in zip(jax.tree.leaves(grads[0]), jax.tree.leaves(grads[1]))
        )
        print(f"  max |custom - autodiff| = {drift:.2e}")

    header = f"\n{'variant':<22}" + "".join(f"{'T=' + str(T):>12}" for T in lengths)
    print(header + f"{'bytes/step':>13}{'floats/step':>13}")
    for name, step in variants:
        if grads[variants.index((name, step))] is None:
            print(f"{name:<22}  (skipped -- gradient failed)")
            continue
        fn = grad_fn(step)
        temps = [temp_bytes(fn, T) for T in lengths]
        slope = (temps[-1] - temps[0]) / (lengths[-1] - lengths[0])
        row = "".join(f"{t:>12,}" for t in temps)
        print(f"{name:<22}{row}{slope / BATCH:>13,.7f}{slope / BATCH / 4:>13.7f}")


if __name__ == "__main__":
    main()
