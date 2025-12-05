# TensorBoard Guide for PPO vs GRPO Comparison

## Overview

The `grpo_vs_baseline_mountaincar.ipynb` notebook now includes TensorBoard logging to track and visualize training metrics in real-time.

## Quick Start

### Option 1: Launch from Terminal
```bash
cd /home/rainer/pr/github/stable-baselines3-contrib/examples
./launch_tensorboard.sh
```
Then open your browser to: **http://localhost:6006**

### Option 2: Launch from Jupyter Notebook
Run the TensorBoard cell (cell 27) in the notebook:
```python
%load_ext tensorboard
%tensorboard --logdir examples/tensorboard_logs
```

### Option 3: Manual Launch
```bash
tensorboard --logdir examples/tensorboard_logs --port 6006
```

## What Gets Logged

### Training Metrics (Both PPO and GRPO)
- **rollout/ep_rew_mean**: Average episode reward
  - Target: 90-100 (goal reached)
  - Initial: ~0 (not reaching goal)
  
- **rollout/ep_len_mean**: Average episode length
  - Success: ~100-300 steps
  - Initial: ~999 steps (timeout)

- **time/fps**: Training speed (frames per second)
- **time/iterations**: Number of training iterations
- **time/total_timesteps**: Cumulative environment steps

### PPO-Specific Metrics
- **train/value_loss**: Value function (critic) loss
- **train/policy_gradient_loss**: Policy gradient loss
- **train/entropy_loss**: Entropy regularization loss
- **train/approx_kl**: Approximate KL divergence
- **train/clip_fraction**: Fraction of clipped policy updates
- **train/explained_variance**: How well the value function explains returns

### GRPO-Specific Metrics
- **train/policy_loss**: Combined policy loss
- **train/kl_loss**: KL divergence regularization loss
- **train/group_advantage_std**: Standard deviation of advantages within groups

## Comparing PPO vs GRPO

### In TensorBoard UI:

1. **Open the Scalars Tab**
   - You'll see separate runs for PPO and GRPO

2. **Enable Run Comparison**
   - Check both PPO and GRPO runs in the left sidebar
   - TensorBoard will overlay the curves

3. **Key Comparisons to Make**

   **Training Efficiency:**
   - Compare `rollout/ep_rew_mean` curves
   - Which algorithm reaches 90+ reward first?
   - Which has smoother learning progress?

   **Sample Efficiency:**
   - At the same number of timesteps, which has higher reward?
   - Compare `time/total_timesteps` vs `rollout/ep_rew_mean`

   **Stability:**
   - Which algorithm has less variance in rewards?
   - Use TensorBoard's smoothing slider to see trends

   **Computational Cost:**
   - Compare `time/fps` (frames per second)
   - GRPO should be faster (no critic network)

## Expected Results

### Successful Training
Both algorithms should show:
- `rollout/ep_rew_mean` rising from ~0 to 90-100
- `rollout/ep_len_mean` dropping from ~999 to ~100-300
- Smooth convergence over 1M timesteps

### Failed Training
If you see:
- `rollout/ep_rew_mean` staying near 0
- `rollout/ep_len_mean` staying near 999
- Consider increasing timesteps or adjusting hyperparameters

## Directory Structure

```
examples/
├── tensorboard_logs/
│   ├── PPO_1/              # PPO training run
│   │   └── events.out.tfevents.*
│   └── GRPO_1/             # GRPO training run
│       └── events.out.tfevents.*
├── models/
│   ├── ppo_mountaincar.zip
│   └── grpo_mountaincar.zip
└── videos/
    ├── ppo_mountaincar/
    ├── grpo_mountaincar/
    └── ppo_vs_grpo_mountaincar_side_by_side.mp4
```

## Tips and Tricks

### Smoothing Curves
- Use the smoothing slider (top left) to reduce noise in plots
- Recommended: 0.6-0.8 for episode rewards

### Downloading Data
- Click the download icon to export metrics as CSV or JSON
- Useful for creating custom plots in matplotlib or seaborn

### Comparing Multiple Runs
- Run training multiple times with different seeds
- TensorBoard will show all runs, helping assess robustness

### Custom Scalars
- The notebook logs custom metrics via the callback
- All episode rewards and lengths are automatically tracked

## Troubleshooting

### TensorBoard Won't Start
```bash
# Check if port 6006 is already in use
lsof -i :6006

# Kill existing TensorBoard process
pkill -f tensorboard

# Try a different port
tensorboard --logdir examples/tensorboard_logs --port 6007
```

### No Data Showing
- Ensure you've run the training cells in the notebook
- Check that `examples/tensorboard_logs/` directory exists
- Verify event files exist: `ls -la examples/tensorboard_logs/*/`

### Old Data Showing
- TensorBoard caches data
- Refresh the page (Ctrl+R or Cmd+R)
- Or restart TensorBoard

## Advanced Usage

### Remote TensorBoard Access
If running on a remote server:
```bash
# On remote server
tensorboard --logdir examples/tensorboard_logs --host 0.0.0.0 --port 6006

# On local machine (SSH tunnel)
ssh -L 6006:localhost:6006 user@remote-server
```
Then access: http://localhost:6006

### Exporting for Papers/Presentations
1. Take screenshots from TensorBoard UI
2. Or download CSV and create custom plots:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load exported CSV from TensorBoard
df_ppo = pd.read_csv('ppo_rewards.csv')
df_grpo = pd.read_csv('grpo_rewards.csv')

plt.plot(df_ppo['Step'], df_ppo['Value'], label='PPO')
plt.plot(df_grpo['Step'], df_grpo['Value'], label='GRPO')
plt.legend()
plt.xlabel('Timesteps')
plt.ylabel('Episode Reward')
plt.savefig('ppo_vs_grpo_comparison.png', dpi=300)
```

## Resources

- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [Stable-Baselines3 Logging](https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html)
- [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) - More benchmark results
