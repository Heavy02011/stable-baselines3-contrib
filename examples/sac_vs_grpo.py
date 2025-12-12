"""
Side-by-side training comparison between SAC (Stable-Baselines3)
and GRPO (sb3-contrib) on MountainCarContinuous-v0.

The script trains each agent until it reaches the specified reward
threshold (default: 90) or hits the maximum timesteps budget.
It records evaluation rewards along the way and produces a small
comparison plot so you can see which learner solves the task faster.

Usage:
    python examples/sac_vs_grpo.py --max-timesteps 150000 --threshold 90

See examples/sac_vs_grpo.md for the tuned hyperparameters and context.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import GRPO


ENV_ID = "MountainCarContinuous-v0"


@dataclass
class RunStats:
    algo: str
    rewards: list[float]
    timesteps: list[int]
    wallclock_s: float
    solved: bool


def format_stats(stats: RunStats) -> str:
    status = "✅" if stats.solved else "⚠️"
    best = f"{max(stats.rewards):.1f}" if stats.rewards else "N/A"
    steps = stats.timesteps[-1] if stats.timesteps else 0
    return f"{status} {stats.algo.upper()}: best={best}, steps={steps}, wallclock={stats.wallclock_s:.1f}s"


def make_env(seed: int) -> gym.Env:
    env = Monitor(gym.make(ENV_ID))
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def train_until_solved(
    algo: str,
    max_timesteps: int,
    threshold: float,
    eval_episodes: int,
    seed: int,
    eval_every: int = 10_000,
    n_envs: int = 1,
) -> tuple[RunStats, SAC | GRPO]:
    # SAC trains on a single monitored env (off-policy buffer handles decorrelation);
    # GRPO follows its tuned on-policy setup with a vectorized env for stable batches.
    train_env = make_env(seed) if algo == "sac" else make_vec_env(ENV_ID, n_envs=n_envs, seed=seed)
    eval_env = make_env(seed)
    if algo == "sac":
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=0.0003,
            gamma=0.9999,
            buffer_size=50_000,
            batch_size=512,
            train_freq=32,
            gradient_steps=32,
            tau=0.01,
            ent_coef=0.1,
            use_sde=False,
            verbose=1,
            seed=seed,
        )
    elif algo == "grpo":
        # Hyperparameters that solve MountainCarContinuous-v0 (mean reward > 90)
        # with 8 parallel environments in ~60k-80k timesteps (5 deterministic evals).
        # Uses gSDE exploration (sde_sample_freq=4) with a 2-layer 256-unit policy.
        # batch_size==n_steps (512) so each epoch processes all collected transitions in a single minibatch.
        model = GRPO(
            "MlpPolicy",
            train_env,
            learning_rate=4e-4,
            n_steps=512,
            batch_size=512,
            n_epochs=20,
            gamma=0.999,
            gae_lambda=0.95,
            group_size=4,
            kl_coef=0.02,
            clip_range=0.25,
            ent_coef=0.0,
            vf_coef=0.5,
            clip_range_vf=0.2,
            max_grad_norm=0.5,
            use_sde=True,
            sde_sample_freq=4,
            policy_kwargs=dict(net_arch=[256, 256]),
            seed=seed,
            verbose=1,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    rewards, steps = [], []
    solved = False
    start = time.time()

    while model.num_timesteps < max_timesteps:
        chunk = min(eval_every, max_timesteps - model.num_timesteps)
        if chunk <= 0:
            break
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes, deterministic=True)
        rewards.append(mean_reward)
        steps.append(model.num_timesteps)
        print(f"[{algo.upper()}] {model.num_timesteps} steps -> mean_reward={mean_reward:.2f}")
        if mean_reward >= threshold:
            solved = True
            break

    wallclock_s = time.time() - start
    train_env.close()
    eval_env.close()
    return RunStats(algo=algo, rewards=rewards, timesteps=steps, wallclock_s=wallclock_s, solved=solved), model


def plot_progress(
    results: list[RunStats], output_path: Path, threshold: float, eval_episodes: int | None = None
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot creation.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for stats in results:
        ax.plot(stats.timesteps, stats.rewards, marker="o", label=f"{stats.algo.upper()}")
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=1, label=f"target reward ({threshold})")
    ax.set_xlabel("Timesteps")
    episode_label = f"{eval_episodes} eval episodes" if eval_episodes is not None else "evaluation rollouts"
    ax.set_ylabel(f"Mean reward ({episode_label})")
    ax.set_title("MountainCarContinuous-v0: SAC vs GRPO")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved comparison plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=90.0, help="Reward threshold for success.")
    parser.add_argument("--max-timesteps", type=int, default=400_000, help="Per-agent training budget.")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes used for evaluation rollouts.")
    parser.add_argument("--eval-every", type=int, default=20_000, help="Train this many timesteps between evals.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel environments for on-policy (GRPO).")
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path(__file__).with_name("sac_vs_grpo.png"),
        help="Where to store the comparison plot.",
    )
    args = parser.parse_args()

    print(f"Training SAC and GRPO on {ENV_ID} until reward >= {args.threshold}")
    results: list[RunStats] = []
    for algo_name in ("sac", "grpo"):
        stats, _ = train_until_solved(
            algo_name,
            max_timesteps=args.max_timesteps,
            threshold=args.threshold,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            eval_every=args.eval_every,
            n_envs=args.n_envs,
        )
        results.append(stats)

    sac_stats, grpo_stats = results
    plot_progress(results, args.plot_path, args.threshold, args.eval_episodes)

    print("Summary:")
    print("  " + format_stats(sac_stats))
    print("  " + format_stats(grpo_stats))
    if sac_stats.solved != grpo_stats.solved:
        winner = "SAC" if sac_stats.solved else "GRPO"
        print(f"=> {winner} solved the task while the other did not.")
    elif sac_stats.rewards and grpo_stats.rewards:
        if max(sac_stats.rewards) > max(grpo_stats.rewards):
            print("=> SAC reached the higher peak reward.")
        elif max(grpo_stats.rewards) > max(sac_stats.rewards):
            print("=> GRPO reached the higher peak reward.")
        else:
            print("=> Both matched on peak reward; check timesteps for speed.")


if __name__ == "__main__":
    main()
