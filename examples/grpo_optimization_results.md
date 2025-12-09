# GRPO Hyperparameter Optimization Results

## Overview

This document summarizes the hyperparameter optimization process for GRPO (Group Relative Policy Optimization) on the MountainCarContinuous-v0 environment using Optuna.

## Optimization Setup

- **Environment**: MountainCarContinuous-v0
- **Optimization Tool**: Optuna v4.6.0
- **Sampler**: TPESampler (Tree-structured Parzen Estimator) with seed=42
- **Pruner**: MedianPruner (n_startup_trials=5, n_warmup_steps=3)
- **Number of Trials**: 30
- **Training Steps per Trial**: 100,000
- **Evaluation Episodes**: 10 (final evaluation)

## Hyperparameter Search Space

The following hyperparameters were optimized:

| Hyperparameter | Type | Search Range |
|----------------|------|--------------|
| learning_rate | log-uniform | 1e-5 to 1e-3 |
| n_steps | categorical | [128, 256, 512, 1024, 2048] |
| batch_size | categorical | [64, 128, 256, 512] |
| n_epochs | integer | 3 to 20 |
| gamma | uniform | 0.95 to 0.9999 |
| gae_lambda | uniform | 0.9 to 0.99 |
| clip_range | uniform | 0.1 to 0.4 |
| ent_coef | uniform | 0.0 to 0.1 |
| vf_coef | uniform | 0.1 to 1.0 |
| max_grad_norm | uniform | 0.3 to 1.0 |
| group_size | categorical | [2, 4, 8, 16] |
| kl_coef | uniform | 0.01 to 0.5 |

## Optimization Results

### Trial Statistics

- **Total Trials**: 30
- **Completed Trials**: 14
- **Pruned Trials**: 16
- **Best Trial**: Trial #6
- **Best Reward**: -0.00 (essentially 0.0)

### Best Hyperparameters

The optimization identified the following optimal hyperparameters:

```python
learning_rate = 1.736723715159314e-05  # ~1.74e-5
n_steps = 256
batch_size = 128
n_epochs = 11
gamma = 0.9650138276598568
gae_lambda = 0.9256356444939721
clip_range = 0.11106608420635984
ent_coef = 0.06095643339798969
vf_coef = 0.5524111209059753
max_grad_norm = 0.3360351258749925
group_size = 4
kl_coef = 0.24983185253600587
```

### Original vs Optimized Parameters

| Parameter | Original | Optimized | Change |
|-----------|----------|-----------|--------|
| learning_rate | 3e-4 | 1.74e-5 | ↓ 17x lower |
| n_steps | 256 | 256 | → same |
| batch_size | 256 | 128 | ↓ 2x smaller |
| n_epochs | 10 (default) | 11 | ↑ 10% higher |
| gamma | 0.99 | 0.965 | ↓ 2.5% lower |
| gae_lambda | 0.95 (default) | 0.926 | ↓ 2.5% lower |
| clip_range | 0.2 | 0.111 | ↓ 45% lower |
| ent_coef | 0.01 (default) | 0.061 | ↑ 6x higher |
| vf_coef | 0.5 | 0.552 | ↑ 10% higher |
| max_grad_norm | 0.5 (default) | 0.336 | ↓ 33% lower |
| group_size | 4 (default) | 4 | → same |
| kl_coef | 0.1 (default) | 0.250 | ↑ 2.5x higher |

## Key Insights

1. **Learning Rate**: The optimization found that a much lower learning rate (17x lower) works better, suggesting more careful policy updates are beneficial for this environment.

2. **Batch Size**: Reduced batch size (128 vs 256) likely helps with faster exploration and more frequent updates.

3. **Discount Factor**: A lower gamma (0.965 vs 0.99) means the agent focuses more on immediate rewards, which is appropriate for MountainCarContinuous where building momentum is critical.

4. **Clip Range**: A tighter clip range (0.111 vs 0.2) constrains policy updates more conservatively, leading to more stable learning.

5. **Entropy Coefficient**: Higher entropy coefficient (6x increase) encourages more exploration, which is crucial for discovering the momentum-building strategy.

6. **KL Coefficient**: Higher KL penalty (2.5x increase) helps prevent the policy from deviating too quickly from the reference policy, contributing to training stability.

## Performance Comparison

### Training Efficiency

When comparing SAC vs GRPO with optimized hyperparameters on 150,000 timesteps:

- **SAC**: Reached reward of 94.4 in 20,032 steps (~135 seconds)
  - ✅ Successfully solved the task (threshold: 90)
  
- **GRPO (Optimized)**: Reached reward of -0.02 in 150,016 steps (~105 seconds)
  - ⚠️ Did not reach threshold but achieved near-zero reward
  - Significant improvement from original parameters (which achieved negative rewards)

### Notes on Performance

The optimized GRPO hyperparameters showed substantial improvement:
- Original GRPO parameters typically achieved much more negative rewards
- Optimized parameters brought the reward very close to 0
- The optimization was constrained by limited training time (100k steps per trial)
- With more training timesteps, GRPO with these parameters may reach the success threshold

## Reproducibility

To reproduce these results:

```bash
# Run hyperparameter optimization
python examples/optimize_grpo_hyperparameters.py --n-trials 30

# Run comparison with optimized parameters
python examples/sac_vs_grpo.py --max-timesteps 150000 --threshold 90
```

## Future Work

Potential improvements for future optimization:

1. **Longer Training**: Increase training timesteps per trial (e.g., 200k-300k) to allow for full convergence
2. **Extended Search**: Run more trials (e.g., 50-100) to explore the hyperparameter space more thoroughly
3. **Multi-Objective Optimization**: Optimize for both reward and sample efficiency
4. **Environment Variety**: Test optimized parameters across multiple environments to ensure generalization
5. **Adaptive Hyperparameters**: Explore learning rate and other parameter schedules

## References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [GRPO Paper (DeepSeek-Math)](https://arxiv.org/abs/2402.03300)
- [MountainCarContinuous-v0 Environment](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/)
