"""Train and evaluate an Astrobee tracking policy."""

import equinox as eqx
import jax
import matplotlib.pyplot as plt
import optax

from environments import astrobee as env
from evaluate import evaluate
from train import MLPPolicy, TrainingParams, compile_memory, make_train_step, train


def main():
    key = jax.random.key(0)
    init_key, train_key, eval_key = jax.random.split(key, 3)
    dynamics_params = env.default_dynamics_params()
    env_params = env.default_env_params()
    dt = 0.05
    train_params = TrainingParams(T=200, iters=1500, batch=64, lr=1e-3)
    policy = MLPPolicy(env, init_key)

    optim = optax.adam(train_params.lr)
    opt_state = optim.init(eqx.filter(policy, eqx.is_array))
    memory = compile_memory(
        make_train_step(env),
        (
            policy,
            opt_state,
            optim,
            train_key,
            dynamics_params,
            dt,
            env_params,
            train_params,
        ),
    )
    print(
        f"[train_step] temp={memory.temp_size_in_bytes} bytes  "
        f"output={memory.output_size_in_bytes / 1e6:.3f}MB  "
        f"args={memory.argument_size_in_bytes / 1e6:.3f}MB  "
        f"code={memory.generated_code_size_in_bytes / 1e6:.3f}MB"
    )

    trained, losses = train(
        env, policy, train_key, dynamics_params, dt, env_params, train_params
    )
    out = evaluate(
        env, trained, eval_key, dynamics_params, dt, env_params, train_params.T
    )
    for metric in env.evaluation_metrics(out):
        unit = f" {metric.unit}" if metric.unit else ""
        print(
            f"eval: initial {metric.name} = {metric.values[0]:.3f}{unit}, "
            f"final = {metric.values[-1]:.3f}{unit}, "
            f"mean = {metric.values.mean():.3f}{unit}"
        )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(losses, color="tab:gray", lw=1)
    ax.set_xlabel("training iteration")
    ax.set_ylabel("mean tracking cost")
    ax.set_title(f"{env.ENV_NAME} policy-gradient training")
    fig.tight_layout()
    training_curve_path = f"training_curve_{env.ENV_NAME}.png"
    fig.savefig(training_curve_path, dpi=120)
    plt.close(fig)
    print(f"saved plot to {training_curve_path}")

    trajectory_path = f"trajectory_{env.ENV_NAME}.gif"
    env.save_trajectory_gif(
        out, dt, trajectory_path, title=f"{env.ENV_NAME} policy tracking"
    )


if __name__ == "__main__":
    main()
