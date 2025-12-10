# SAC vs. GRPO on MountainCarContinuous-v0

This walkthrough trains **SAC** (from Stable-Baselines3) and **GRPO** (from sb3-contrib) side by side on `MountainCarContinuous-v0` until they exceed a reward of **90**. The helper script logs which algorithm solves the task faster and produces a compact plot.

> This is the maintained comparison example for this fork. It supersedes the older notebook/RL-Zoo snippets and mirrors the tuned parameters summarized in `grpo_optimization_results.md` / `grpo_optimized_params.txt`.

## How to run

```bash
python examples/sac_vs_grpo.py \
  --threshold 90 \
  --max-timesteps 150000 \
  --eval-every 10000 \
  --n-envs 8 \
  --eval-episodes 5 \
  --seed 0
```

- The script will print intermediate evaluation rewards for both agents.
- Training stops early as soon as the current agent crosses the reward threshold or the budget is spent.
- A plot named `sac_vs_grpo.png` is written next to the script (requires `matplotlib`; install with `pip install matplotlib` if you want the graphic).
![](sac_vs_grpo.png)

## Notes on current results

- GRPO now uses the faster MountainCar configuration (lr=4e-4, n_steps=512, batch_size=512, n_epochs=20, group_size=4, kl_coef=0.02, clip_range=0.25, vf_coef=0.5, gSDE on, net_arch=[256, 256]) with 8 parallel environments.
- With these hyperparameters a recent run crossed 90 reward after ~60k steps (5 eval episodes), an order-of-magnitude reduction compared to the previous ~340k budget.
- SAC keeps the previous configuration and still solves the task quickly; the plot generation remains the same (`sac_vs_grpo.png`).
