from imitation.policies.base import NormalizeFeaturesExtractor
from main import set_seed
from imitation.util.util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from imitation.data.wrappers import RolloutInfoWrapper
import torch.nn as nn
import numpy as np
from stable_baselines3.ppo import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium import Wrapper
import torch
from stable_baselines3.common.vec_env import VecNormalize as _VecNormalize
from huggingface_sb3 import load_from_hub


class RewardReplaceWrapper(Wrapper):
    def __init__(self, env, reward_path):
        super().__init__(env)
        self.reward_fn = torch.load(reward_path, weights_only=False)
        self.prev_obs = None

    def step(self, action):
        # Unpack gymnasium step tuple (obs, reward, terminated, truncated, info)
        obs, _, terminated, truncated, info = self.env.step(action)
        # Compute custom reward using previous obs, action, and new obs
        if self.prev_obs is None:
            # On first step, no previous obs: use zero reward or original
            new_reward = 0.0
        else:
            # Call loaded reward function; convert inputs if needed
            # Assume reward_fn takes (state, action, next_state) and returns a scalar
            new_reward = self.reward_fn(self.prev_obs, action, obs)
            # If torch tensor, get Python float
            if isinstance(new_reward, torch.Tensor):
                new_reward = new_reward.item()
        # Update prev_obs
        self.prev_obs = obs
        return obs, new_reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Reset environment, handling gymnasium API
        reset_out = self.env.reset(**kwargs)
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, info = reset_out
        else:
            # Fallback if environment returns only obs
            obs, info = reset_out, {}
        # Initialize prev_obs
        self.prev_obs = obs
        return obs, info

if __name__ == '__main__':
    SEED = 123
    set_seed(SEED)

    env = make_vec_env(
        "Humanoid-v4",
        rng=np.random.default_rng(SEED),
        n_envs=1,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
    )

    env = VecNormalize(env,gamma=0.999 ,norm_obs=False, norm_reward=True)

    policy_kwags = {
        "log_std_init": -1,
        "activation_fn": nn.ReLU,
        "features_extractor_class": NormalizeFeaturesExtractor,
        'net_arch': dict(pi=[256, 256], vf=[256, 256])
    }
    learner = PPO(
                    env=env,
                    policy=MlpPolicy,
                    batch_size=256,
                    n_steps=2048,
                    ent_coef=2.07e-05,
                    gae_lambda=0.92,
                    learning_rate=2.03e-5,
                    gamma=0.999,
                    clip_range=0.2,
                    vf_coef=0.8192,
                    n_epochs=20,
                    seed=SEED,
                    policy_kwargs=policy_kwags,
                    max_grad_norm=0.5,
                    tensorboard_log="./log/",
                    verbose=1,
                   )
    learner.learn(total_timesteps=int(1.5e6))
    learner.save("./model/expert/ppo-Humanoid-v4.zip")

    # Reload VecNormalize statistics
    learner = PPO.load("./model/expert/ppo-Humanoid-v4.zip")
    learner_rewards_after_training, _ = evaluate_policy(
        learner, env, 100, return_episode_rewards=True,
    )
    print("mean reward after training:", np.mean(learner_rewards_after_training))