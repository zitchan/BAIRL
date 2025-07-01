import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from imitation.util.networks import BaseNorm, RunningNorm
from stable_baselines3.common.torch_layers import CombinedExtractor, BaseFeaturesExtractor
from imitation.rewards.reward_nets import BasicShapedRewardNet
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Type, cast
from imitation.rewards.reward_nets import ShapedRewardNet, BasicPotentialMLP, BasicRewardNet
from imitation.algorithms import base
from imitation.algorithms.adversarial import common
from imitation.algorithms.adversarial.airl import AIRL
from typing import Optional, Mapping
import torch
from imitation.rewards.reward_nets import BasicShapedRewardNet, NormalizedRewardNet, RewardNet
from imitation.util.networks import RunningNorm



class Bayesian_reward_net(RewardNet):
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            *,
            use_state: bool = True,
            use_action: bool = True,
            use_next_state: bool = False,
            use_done: bool = False,
            discount_factor: float = 0.99,
            dropout: float = 0.0,
            normalize_output_layer=RunningNorm,
            **kwargs,
    ):

        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
        )
        self.observation_space = observation_space
        self.action_space = action_space
        self.use_state = use_state
        self.use_next_state = use_next_state
        self.use_done = use_done
        self.use_action = use_action
        self.discount_factor = discount_factor
        self.normalize_output_layer = normalize_output_layer(1)
        self.dropout = dropout
        # Compute observation dimension
        if hasattr(self.observation_space, 'shape'):
            self.obs_dim = int(np.prod(self.observation_space.shape))
        elif hasattr(self.observation_space, 'n'):
            self.obs_dim = self.observation_space.n
        else:
            raise ValueError('Unsupported observation space type')
        # Compute action dimension
        if hasattr(self.action_space, 'n'):
            self.act_is_discrete = True
            self.act_dim = self.action_space.n
        elif hasattr(self.action_space, 'shape'):
            self.act_is_discrete = False
            self.act_dim = int(np.prod(self.action_space.shape))
        else:
            raise ValueError('Unsupported action space type')


        if use_action:
            self.mlp = BayesianNet(self.obs_dim + self.act_dim,  hidden_dim=128,dropout_p=dropout)
        else:
            self.mlp = BayesianNet(self.obs_dim,  hidden_dim=128,dropout_p=dropout)
        self.potential = BayesianNet(self.obs_dim,  hidden_dim=128,dropout_p=dropout)


    def forward(self, obs, action, next_obs, done, **kwargs):
        inputs = []
        if self.use_state:
            inputs.append(torch.flatten(obs, 1).float())
        if self.use_action:
            inputs.append(torch.flatten(action, 1).float())
        combined_inputs = torch.cat(inputs, dim=-1)
        reward = self.mlp(combined_inputs)
        if self.use_next_state:
            new_shaping_output = self.potential(next_obs.float()).flatten()
            old_shaping_output = self.potential(obs.float()).flatten()
            reward = reward.squeeze(-1)
            old_shaping_output = old_shaping_output.squeeze(-1)
            new_shaping_output = new_shaping_output.squeeze(-1)
            new_shapping = (1 - done.float()) * new_shaping_output
            shaped = self.discount_factor * new_shapping - old_shaping_output
            final_reward = (reward + shaped)
        else:
            final_reward = reward.squeeze(-1)

        final_reward = self.normalize_output_layer(final_reward)
        with torch.no_grad():
            self.normalize_output_layer.update_stats(final_reward)
        assert final_reward.shape == obs.shape[:1]
        return final_reward

    def predict_processed(self, state, action, next_state, done, update_stats=True, **kwargs):
        with torch.no_grad():
            if self.act_is_discrete:
                # 转 one-hot
                if torch.is_tensor(action):
                    action = torch.nn.functional.one_hot(action.long(), num_classes=self.act_dim).float()
                else:
                    # numpy
                    action = np.array(action).reshape(-1)
                    action = torch.tensor(np.eye(self.act_dim)[action], dtype=torch.float32)
            rew_th = self.forward(
                torch.as_tensor(state, device=self.device, dtype=torch.float32),
                torch.as_tensor(action, device=self.device, dtype=torch.float32),
                torch.as_tensor(next_state, device=self.device, dtype=torch.float32),
                torch.as_tensor(done, device=self.device)
            )
            rew = self.normalize_output_layer(rew_th).detach().cpu().numpy().flatten()
            if update_stats:
                self.normalize_output_layer.update_stats(rew_th)
        assert rew.shape == state.shape[:1]
        return rew


class BayesianNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_p=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.mlp(x)

    def sample_reward(self, x, n_samples=20):
        self.train()
        rewards = []
        for _ in range(n_samples):
            rewards.append(self.forward(x).detach())
        rewards = torch.stack(rewards, dim=0)
        mean = rewards.mean(dim=0).squeeze(-1)
        std = rewards.std(dim=0).squeeze(-1)
        return mean, std
