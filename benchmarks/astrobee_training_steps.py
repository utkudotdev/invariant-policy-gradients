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
    astrobee_reduced_analytic,
    astrobee_reduced_derived_remat,
    astrobee_reduced_remat,
    astrobee_remat,
)
from train import MLPPolicy, TrainingParams, make_train_step

VARIANTS = (
    ("unreduced", astrobee),
    ("unreduced + checkpoint", astrobee_remat),
    ("reduced (autodiff)", astrobee_reduced),
    ("reduced + analytic VJP", astrobee_reduced_analytic),
    ("reduced + checkpoint", astrobee_reduced_remat),
    ("reduced + derived + checkpoint", astrobee_reduced_derived_remat),
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


def temp_bytes(train_step, env: ModuleType, steps: int, batch: int) -> int:
    args = step_args(env, steps, batch)
    return (
        train_step.lower(*args).compile().compiled.memory_analysis().temp_size_in_bytes
    )


def benchmark(
    env: ModuleType,
    steps: int,
    batch: int,
    reps: int,
    slope_steps: tuple[int, int],
):
    train_step = make_train_step(env)
    args = step_args(env, steps, batch)
    compiled = train_step.lower(*args).compile()
    memory = compiled.compiled.memory_analysis()
    low_steps, high_steps = slope_steps
    low_temp = temp_bytes(train_step, env, low_steps, batch)
    high_temp = temp_bytes(train_step, env, high_steps, batch)
    floats_per_step = (high_temp - low_temp) / (4 * batch * (high_steps - low_steps))

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
        "floats_per_step": floats_per_step,
        "args_mib": memory.argument_size_in_bytes / 2**20,
        "output_mib": memory.output_size_in_bytes / 2**20,
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
    }


def save_plot(
    results,
    path: str,
    steps: int,
    batch: int,
    slope_steps: tuple[int, int],
):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.get_cmap("tab10").colors
    markers = ("o", "s", "^", "D", "P", "X")

    for i, (name, result) in enumerate(results):
        ax.scatter(
            result["floats_per_step"],
            result["median_ms"],
            s=90,
            color=colors[i],
            marker=markers[i],
            label=name,
            zorder=3,
        )

    ax.set_xlabel("XLA temporary-memory slope (floats / step / example)")
    ax.set_ylabel("Median full training-step runtime (ms)")
    ax.set_title(
        "Astrobee training-step tradeoff\n"
        f"(runtime T={steps}, memory T={slope_steps[0]}–{slope_steps[1]}, "
        f"batch={batch})"
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
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
    parser.add_argument(
        "--slope-steps",
        type=int,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(100, 400),
        help="rollout lengths used to estimate per-step temporary memory",
    )
    parser.add_argument("--plot", default="astrobee_training_steps.png")
    args = parser.parse_args()
    if args.slope_steps[0] >= args.slope_steps[1]:
        parser.error("--slope-steps requires LOW < HIGH")
    return args


def main():
    args = parse_args()
    print(f"backend: {jax.default_backend()}  {jax.devices()}")
    print(f"T={args.steps}, batch={args.batch}, repetitions={args.reps}\n")
    print(
        f"{'variant':<29}{'temp MiB':>11}{'floats/step':>14}{'args MiB':>11}"
        f"{'output MiB':>13}{'min ms':>11}{'median ms':>13}{'loss':>12}"
    )

    results = []
    for name, env in VARIANTS:
        result = benchmark(
            env, args.steps, args.batch, args.reps, tuple(args.slope_steps)
        )
        results.append((name, result))
        print(
            f"{name:<29}{result['temp_mib']:>11.2f}"
            f"{result['floats_per_step']:>14.2f}{result['args_mib']:>11.2f}"
            f"{result['output_mib']:>13.2f}{result['min_ms']:>11.2f}"
            f"{result['median_ms']:>13.2f}{result['loss']:>12.4f}"
        )

    save_plot(results, args.plot, args.steps, args.batch, tuple(args.slope_steps))


if __name__ == "__main__":
    main()
