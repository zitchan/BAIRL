from imitation.scripts.ingredients.reward import normalize_output_running
from stable_baselines3.common import base_class, policies, vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac import policies as sac_policies
from stable_baselines3.ppo import MlpPolicy
import numpy as np
import torch.nn as nn
from stable_baselines3.ppo import PPO
from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.policies import ActorCriticPolicy
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.rewards.reward_nets import BasicRewardNet, BasicShapedRewardNet
from imitation.util.networks import BaseNorm, RunningNorm
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from imitation.policies.base import NormalizeFeaturesExtractor
import gymnasium as gym
import torch
from imitation.data import serialize
import os
from gymnasium.envs.registration import register
import subprocess
import threading
import matplotlib
import matplotlib.pyplot as plt
from torch.ao.quantization.utils import activation_dtype
from pretrain_ppo import pretrain
from reward_net import Bayesian_reward_net

from airl import custom_AIRL

from stable_baselines3.common.logger import configure as sb3_configure

matplotlib.use("Agg")
STOCHASTIC_POLICIES = (sac_policies.SACPolicy, policies.ActorCriticPolicy)


def graph_reward_box(data, step, name):
    reward_arr = np.stack([np.array(r).flatten() for r in data], axis=0)  # [num_steps, batch_size]
    steps = np.array(step)
    num_show = 30
    show_idx = np.linspace(0, len(steps) - 1, num_show, dtype=int)
    plt.figure(figsize=(12, 6))
    plt.boxplot(
        reward_arr[show_idx].T,
        positions=steps[show_idx],
        widths=15,
        patch_artist=True,
        boxprops=dict(facecolor='C0', alpha=0.5)
    )
    plt.plot(
        steps[show_idx],
        reward_arr[show_idx].mean(axis=1),
        'o-',
        color='orange',
        label='Mean'
    )
    plt.xlabel("Discriminator Step")
    plt.ylabel(f"Reward{name}")
    plt.title("Reward Distribution Over Training")
    plt.grid(axis='y')
    plt.legend()
    plt.savefig(f"reward_boxplot{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


def set_seed(random_seed):
    set_random_seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    SEED = 43
    set_seed(SEED)
    env = make_vec_env(
        "Ant-v4",
        rng=np.random.default_rng(SEED),
        n_envs=1,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
    )

    env = VecNormalize(env,gamma=0.995 ,norm_obs=False, norm_reward=True)

    # expert = load_policy(
    #     "ppo-huggingface",
    #     organization="sb3",
    #     env_name="HalfCheetah-v3",
    #     venv=env,
    # )

    expert = PPO.load("./model/expert/ppo-ant-v4.zip")
    # reward, _ = evaluate_policy(
    #     expert, env, 100, return_episode_rewards=True,
    # )
    # print(np.mean(reward))
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_episodes=100),
        rng=np.random.default_rng(SEED),

    )

    policy_kwags = {
        "log_std_init": -1,
        "activation_fn": nn.modules.activation.Tanh,
        "features_extractor_class": NormalizeFeaturesExtractor,
        'net_arch': dict(pi=[64, 64], vf=[64, 64])
    }

    learner = PPO(
        env=env,
        policy=MlpPolicy,
        batch_size=16,
        n_steps=2048,
        ent_coef=3.1441389214159857e-06,
        gae_lambda=0.8,
        learning_rate=0.00017959211641976886,
        gamma=0.995,
        clip_range=0.1,
        vf_coef=0.4351450387648799,
        n_epochs=10,
        seed=SEED,
        policy_kwargs=policy_kwags,
        max_grad_norm=0.9,
        tensorboard_log="./log/",
        verbose=1,
    )

    b_reward_net = Bayesian_reward_net(observation_space=env.observation_space,
                                       action_space=env.action_space,
                                       use_state=True,
                                       use_action=True,
                                       use_next_state=True,
                                       use_done=True,
                                       feature_extractor=None,
                                       dropout=0.3,
                                       hidden_size=256,
                                       )


    airl_trainer = custom_AIRL(
                                demonstrations=rollouts,
                                demo_batch_size=2048,
                                gen_replay_buffer_capacity=512,
                                n_disc_updates_per_round=8,
                                venv=env,
                                gen_algo=learner,
                                reward_net=b_reward_net,
                                reg="B",  # Adversarial_Augmentation:AA, Gradient penalty: GP, Bayesian
                                lambda_reg=-0.1,
                                init_tensorboard_graph=True,
                                init_tensorboard=True,
                                log_dir="./log/",
                                allow_variable_horizon=True
                               )

    # # Attach SB3 TensorBoard logger to the generator policy
    sb3_logger = sb3_configure("./log/ppo", ["stdout", "tensorboard"])
    airl_trainer.gen_algo.set_logger(sb3_logger)
    env.seed(SEED)

    learner_rewards_before_training, _ = evaluate_policy(
        expert, env, 100, return_episode_rewards=True,
    )

    print(airl_trainer.lambda_reg)
    # airl_trainer.train(1000_000)
    for n in range(50):
        airl_trainer.train(20000)
        if n > 2:
            airl_trainer.lambda_reg *= 0.5
    airl_trainer.gen_algo.save("./model/airl/model/test")
    torch.save(b_reward_net, "model/airl/model/test.pth")
    # env.seed(SEED)
    learner_rewards_after_training, _ = evaluate_policy(
        airl_trainer.gen_algo, env, 100, return_episode_rewards=True,
    )
    print("mean reward after training:", np.mean(learner_rewards_after_training))
    # print("mean reward before training:", np.mean(learner_rewards_before_training))
    #
