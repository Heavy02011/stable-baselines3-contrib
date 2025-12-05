# Vergleich SAC / GRPO

## SAC
```yaml
python -m rl_zoo3.train \
  --algo sac \
  --env MountainCarContinuous-v0 \
  --tensorboard-log /tmp/tensorboard_logs/ \
  --eval-freq 10000 \
  --save-freq 50000 \
  --seed 0 \
  -params learning_rate:3e-4 gamma:0.9999 batch_size:512 buffer_size:50000 \
          gradient_steps:32 train_freq:32 tau:0.01 ent_coef:0.1
```

## GRPO

```yaml
python -m rl_zoo3.train \
  --algo grpo \
  --env MountainCarContinuous-v0 \
  --tensorboard-log /tmp/tensorboard_logs/ \
  --eval-freq 10000 \
  --save-freq 50000 \
  --seed 0 \
  -params learning_rate:3e-4 gamma:0.999 \
          n_steps:2048 batch_size:256 \
          clip_range:0.2 gae_lambda:0.95 \
          vf_coef:0.5 max_grad_norm:0.5
```
