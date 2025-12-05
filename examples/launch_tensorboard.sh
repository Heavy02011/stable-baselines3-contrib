#!/bin/bash
# Launch TensorBoard to view PPO vs GRPO training results

echo "Starting TensorBoard..."
echo "Navigate to: http://localhost:6006"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo ""

tensorboard --logdir examples/tensorboard_logs --port 6006
