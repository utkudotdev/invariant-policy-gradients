"""Benchmark one complete policy-gradient step for every Astrobee environment.

Each measurement includes a batched rollout, tracking loss, reverse-mode
differentiation, Adam update, and parameter update. Run with:

    uv run python -m benchmarks.astrobee_training_steps
"""

import argparse
import statistics
import time
from types import ModuleType

import equinox as eqx
import jax
import matplotlib.pyplot as plt
import optax

from environments import (
    astrobee,
    astrobee_reduced,
    astrobee_reduced_derived_remat,
    astrobee_reduced_remat,
    astrobee_remat,
)
from train import MLPPolicy, TrainingParams, make_train_step

VARIANTS = (
    ("unreduced", astrobee),
    ("unreduced + checkpoint", astrobee_remat),
    ("reduced + analytic VJP", astrobee_reduced),
    ("reduced + checkpoint", astrobee_reduced_remat),
    ("derived + checkpoint", astrobee_reduced_derived_remat),
)


def step_args(env: ModuleType, steps: int, batch: int):
    key = jax.random.key(0)
    policy_key, train_key = jax.random.split(key)
    policy = MLPPolicy(env, policy_key)
    train_params = TrainingParams(T=steps, iters=1, batch=batch, lr=1e-3)
    optim = optax.adam(train_params.lr)
    opt_state = optim.init(eqx.filter(policy, eqx.is_array))
    return (
        policy,
        opt_state,
        optim,
        train_key,
        env.default_dynamics_params(),
        0.05,
        env.default_env_params(),
        train_params,
    )


def benchmark(env: ModuleType, steps: int, batch: int, reps: int):
    train_step = make_train_step(env)
    args = step_args(env, steps, batch)
    compiled = train_step.lower(*args).compile()
    memory = compiled.compiled.memory_analysis()

    output = compiled(*args)
    jax.block_until_ready(output)
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        output = compiled(*args)
        jax.block_until_ready(output)
        samples.append((time.perf_counter() - start) * 1e3)

    return {
        "loss": float(output[2]),
        "temp_mib": memory.temp_size_in_bytes / 2**20,
        "args_mib": memory.argument_size_in_bytes / 2**20,
        "output_mib": memory.output_size_in_bytes / 2**20,
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
    }


def save_plot(results, path: str, steps: int, batch: int):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.get_cmap("tab10").colors
    markers = ("o", "s", "^", "D", "P")

    for i, (name, result) in enumerate(results):
        ax.scatter(
            result["temp_mib"],
            result["median_ms"],
            s=90,
            color=colors[i],
            marker=markers[i],
            label=name,
            zorder=3,
        )

    ax.set_xlabel("XLA temporary memory (MiB)")
    ax.set_ylabel("Median full training-step runtime (ms)")
    ax.set_title(f"Astrobee training-step tradeoff (T={steps}, batch={batch})")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"\nsaved plot to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--plot", default="astrobee_training_steps.png")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"backend: {jax.default_backend()}  {jax.devices()}")
    print(f"T={args.steps}, batch={args.batch}, repetitions={args.reps}\n")
    print(
        f"{'variant':<29}{'temp MiB':>11}{'args MiB':>11}{'output MiB':>13}"
        f"{'min ms':>11}{'median ms':>13}{'loss':>12}"
    )

    results = []
    for name, env in VARIANTS:
        result = benchmark(env, args.steps, args.batch, args.reps)
        results.append((name, result))
        print(
            f"{name:<29}{result['temp_mib']:>11.2f}{result['args_mib']:>11.2f}"
            f"{result['output_mib']:>13.2f}{result['min_ms']:>11.2f}"
            f"{result['median_ms']:>13.2f}{result['loss']:>12.4f}"
        )

    save_plot(results, args.plot, args.steps, args.batch)


if __name__ == "__main__":
    main()
