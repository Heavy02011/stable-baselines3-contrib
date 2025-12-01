# This file defines MlpPolicy/CnnPolicy that work for GRPO
# GRPO uses actor-only policies (no critic) for continuous control
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, MultiInputActorCriticPolicy

MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy
