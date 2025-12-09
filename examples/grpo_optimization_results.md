# GRPO Hyperparameter Optimization Results

## Overview

This document summarizes the current optimized/tuned GRPO (Group Relative Policy Optimization) configuration for MountainCarContinuous-v0. The latest validated settings match the ones used in `examples/sac_vs_grpo.py` and achieve > 90 reward.

## Current best (validated run)

- **Environment**: MountainCarContinuous-v0
- **Training steps**: ~400k (8 parallel envs)
- **Evaluation**: 5 deterministic episodes per checkpoint
- **Seed**: 0
- **Result**: ~92.0 mean reward (meets > 90 requirement)

## Final tuned hyperparameters

```
learning_rate = 3e-4
n_steps = 1024
batch_size = 1024
n_epochs = 10
gamma = 0.999
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.0
vf_coef = 0.5
clip_range_vf = 0.2
max_grad_norm = 0.5
group_size = 4
kl_coef = 0.05
use_sde = True
sde_sample_freq = 4
policy_net_arch = [256, 256]
seed = 0
train_n_envs = 8
```

## Notes

- These tuned hyperparameters supersede the earlier Optuna sweep results for this environment.
- They are synchronized with:
  - `examples/sac_vs_grpo.py`
  - `examples/grpo_optimized_params.txt`
  - `hyperparameters/grpo.yml` (MountainCarContinuous-v0)

Reproduction (matching the validated run):

```bash
python examples/sac_vs_grpo.py \\
  --threshold 90 \\
  --max-timesteps 400000 \\
  --eval-every 20000 \\
  --eval-episodes 5 \\
  --n-envs 8 \\
  --seed 0
```

## References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [GRPO Paper (DeepSeek-Math)](https://arxiv.org/abs/2402.03300)
- [MountainCarContinuous-v0 Environment](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/)
