# This file defines MlpPolicy/CnnPolicy that work for GRPO
# GRPO uses ActorCriticPolicy with optional value function support.
# The algorithm uses group-based advantage normalization and can
# optionally disable the critic through vf_coef=0 for pure GRPO.
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, MultiInputActorCriticPolicy

MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy
