"""
Hyperparameter optimization for GRPO on MountainCarContinuous-v0 using Optuna.

This script optimizes GRPO hyperparameters to improve performance on the
MountainCarContinuous-v0 environment. It uses Optuna to search for the best
combination of hyperparameters that maximize the evaluation reward.

The optimization process:
1. Samples hyperparameters from predefined ranges using TPESampler
2. Trains GRPO agent for a fixed number of timesteps (default: 100k)
3. Evaluates the agent periodically and reports intermediate values
4. Prunes unpromising trials early using MedianPruner
5. Saves the best hyperparameters to a file

Results from 30 trials optimization:
- Best reward: -0.00 (essentially 0.0)
- Best trial: #6
- Key findings: Lower learning rate, smaller batch size, higher entropy coefficient

Usage:
    # Basic usage (30 trials)
    python examples/optimize_grpo_hyperparameters.py
    
    # Extended optimization (100 trials with 4 parallel jobs)
    python examples/optimize_grpo_hyperparameters.py --n-trials 100 --n-jobs 4
    
    # With persistent storage
    python examples/optimize_grpo_hyperparameters.py --storage sqlite:///optuna_grpo.db
    
    # Custom output path
    python examples/optimize_grpo_hyperparameters.py --output results/best_params.txt

See examples/grpo_optimization_results.md for detailed results and analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from sb3_contrib import GRPO


ENV_ID = "MountainCarContinuous-v0"


def make_env(seed: int) -> gym.Env:
    """Create a monitored environment."""
    env = Monitor(gym.make(ENV_ID))
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna hyperparameter optimization.
    
    Args:
        trial: Optuna trial object for suggesting hyperparameters
        
    Returns:
        Mean reward achieved by the agent
    """
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    n_epochs = trial.suggest_int("n_epochs", 3, 20)
    gamma = trial.suggest_float("gamma", 0.95, 0.9999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)
    group_size = trial.suggest_categorical("group_size", [2, 4, 8, 16])
    kl_coef = trial.suggest_float("kl_coef", 0.01, 0.5)
    
    # Ensure batch_size is compatible with n_steps
    if batch_size > n_steps:
        batch_size = n_steps
    
    # Create environment
    seed = trial.number
    env = make_env(seed)
    
    try:
        # Create model with suggested hyperparameters
        model = GRPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            group_size=group_size,
            kl_coef=kl_coef,
            seed=seed,
            verbose=0,
        )
        
        # Train the model
        max_timesteps = 100_000  # Shorter training for hyperparameter search
        eval_freq = 10_000
        
        for step in range(0, max_timesteps, eval_freq):
            model.learn(total_timesteps=eval_freq, reset_num_timesteps=False, progress_bar=False)
            
            # Evaluate the model
            mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)
            
            # Report intermediate value for pruning
            trial.report(mean_reward, step)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Final evaluation
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
        
        env.close()
        return mean_reward
        
    except Exception as e:
        env.close()
        if isinstance(e, optuna.TrialPruned):
            raise
        print(f"Trial failed with error: {e}")
        return -200.0  # Return a very low reward for failed trials


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize GRPO hyperparameters using Optuna")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--study-name", type=str, default="grpo_mountaincar_optimization", 
                        help="Name of the Optuna study")
    parser.add_argument("--storage", type=str, default=None, 
                        help="Database URL for storing the study (e.g., sqlite:///optuna.db)")
    parser.add_argument("--output", type=Path, default=Path("examples/grpo_optimized_params.txt"),
                        help="Path to save the optimized hyperparameters")
    args = parser.parse_args()
    
    # Create Optuna study
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True,
    )
    
    print(f"Starting optimization with {args.n_trials} trials...")
    print(f"Environment: {ENV_ID}")
    
    # Run optimization
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)
    
    # Print results
    print("\n" + "="*80)
    print("Optimization completed!")
    print("="*80)
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.2f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best hyperparameters to file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("# Optimized GRPO Hyperparameters for MountainCarContinuous-v0\n")
        f.write(f"# Best reward: {study.best_value:.2f}\n")
        f.write(f"# Trial number: {study.best_trial.number}\n\n")
        for key, value in study.best_params.items():
            f.write(f"{key} = {value}\n")
    
    print(f"\nOptimized hyperparameters saved to: {args.output}")
    
    # Print pruning statistics
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\nStatistics:")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")


if __name__ == "__main__":
    main()
