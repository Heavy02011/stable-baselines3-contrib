# SAC vs. GRPO on MountainCarContinuous-v0

This walkthrough trains **SAC** (from Stable-Baselines3) and **GRPO** (from sb3-contrib) side by side on `MountainCarContinuous-v0` until they exceed a reward of **90**. The helper script logs which algorithm solves the task faster and produces a compact plot.

## How to run

```bash
python examples/sac_vs_grpo.py \
  --threshold 90 \
  --max-timesteps 400000 \
  --eval-every 20000 \
  --n-envs 8 \
  --eval-episodes 5
```

- The script will print intermediate evaluation rewards for both agents.
- Training stops early as soon as the current agent crosses the reward threshold or the budget is spent.
- A plot named `sac_vs_grpo.png` is written next to the script (requires `matplotlib`; install with `pip install matplotlib` if you want the graphic).

## Notes on current results

- GRPO now uses the MountainCar-proven hyperparameters (lr=3e-4, n_steps=1024, batch_size=1024, group_size=4, clip_range=0.2, vf_coef=0.5, gSDE on, net_arch=[256, 256]) with 8 parallel environments.
- In a recent run these settings reached a mean reward of ~90.5 after ~340k timesteps (5 evaluation episodes), satisfying the >90 criterion.
- SAC keeps the previous configuration and still solves the task quickly; the plot generation remains the same (`sac_vs_grpo.png`).
