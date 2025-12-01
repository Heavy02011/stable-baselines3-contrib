"""Tests for GRPO algorithm."""

import pytest
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.envs import IdentityEnv, IdentityEnvMultiBinary, IdentityEnvMultiDiscrete
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib import GRPO

DIM = 4


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])
def test_grpo(env_id):
    """Test basic GRPO training on discrete and continuous environments."""
    model = GRPO(
        "MlpPolicy",
        env_id,
        n_steps=128,
        seed=0,
        policy_kwargs=dict(net_arch=[16]),
        verbose=1,
    )
    model.learn(total_timesteps=500)


def test_grpo_params():
    """Test GRPO with various parameters including gSDE."""
    model = GRPO(
        "MlpPolicy",
        "Pendulum-v1",
        n_steps=64,
        batch_size=32,
        use_sde=True,
        group_size=8,
        kl_coef=0.05,
        target_kl=0.02,
        seed=0,
        policy_kwargs=dict(net_arch=dict(pi=[32], vf=[32])),
        verbose=1,
    )
    model.learn(total_timesteps=500)


@pytest.mark.parametrize("group_size", [2, 4, 8])
def test_grpo_group_sizes(group_size):
    """Test GRPO with different group sizes."""
    model = GRPO(
        "MlpPolicy",
        "Pendulum-v1",
        n_steps=64,
        batch_size=32,
        group_size=group_size,
        seed=0,
        policy_kwargs=dict(net_arch=[16]),
        verbose=1,
    )
    model.learn(total_timesteps=300)


@pytest.mark.parametrize("kl_coef", [0.0, 0.1, 0.5])
def test_grpo_kl_coef(kl_coef):
    """Test GRPO with different KL coefficients."""
    model = GRPO(
        "MlpPolicy",
        "Pendulum-v1",
        n_steps=64,
        batch_size=32,
        kl_coef=kl_coef,
        seed=0,
        policy_kwargs=dict(net_arch=[16]),
        verbose=1,
    )
    model.learn(total_timesteps=300)


def test_grpo_multi_env():
    """Test GRPO with multiple environments."""
    env = make_vec_env("Pendulum-v1", n_envs=2)
    model = GRPO(
        "MlpPolicy",
        env,
        n_steps=64,
        batch_size=32,
        seed=0,
        policy_kwargs=dict(net_arch=[16]),
        verbose=1,
    )
    model.learn(total_timesteps=300)


@pytest.mark.parametrize("env", [IdentityEnv(DIM), IdentityEnvMultiDiscrete(DIM), IdentityEnvMultiBinary(DIM)])
def test_grpo_discrete(env):
    """Test GRPO on discrete action space environments."""
    vec_env = DummyVecEnv([lambda: env])
    model = GRPO(
        "MlpPolicy",
        vec_env,
        n_steps=256,
        learning_rate=1e-3,
        gamma=0.4,
        seed=0,
    )
    model.learn(total_timesteps=1500)

    evaluate_policy(model, vec_env, n_eval_episodes=5, warn=False)


@pytest.mark.parametrize("normalize_advantage", [False, True])
def test_grpo_advantage_normalization(normalize_advantage):
    """Test GRPO with and without advantage normalization."""
    model = GRPO(
        "MlpPolicy",
        "CartPole-v1",
        n_steps=64,
        normalize_advantage=normalize_advantage,
        seed=0,
        policy_kwargs=dict(net_arch=[16]),
        verbose=1,
    )
    model.learn(total_timesteps=300)


def test_grpo_hybrid():
    """Test GRPO with value function (hybrid approach)."""
    model = GRPO(
        "MlpPolicy",
        "Pendulum-v1",
        n_steps=64,
        batch_size=32,
        vf_coef=0.5,  # Enable value function
        clip_range_vf=0.2,
        seed=0,
        policy_kwargs=dict(net_arch=[16]),
        verbose=1,
    )
    model.learn(total_timesteps=300)
