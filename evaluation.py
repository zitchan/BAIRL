from stable_baselines3.ppo import PPO
import os
from imitation.data import serialize
from stable_baselines3.common.utils import set_random_seed
from gymnasium.envs.registration import register
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from main import set_seed
import subprocess
import threading
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np

matplotlib.use("Agg")


# def start_external_program():
#     subprocess.Popen(['H:\\sample\\Easy\\Blocks.exe', '-windowed', '-ResX=1280', '-ResY=720'])


def graph_reward_box(data, step, name, save_path=None):
    steps = np.arange(1, step + 1, 1)
    data = np.array(data)
    # data = data.reshape(1080, 1, 2048).mean(axis=1)
    reward_arr = np.stack([np.array(r).flatten() for r in data], axis=0)  # [num_steps, batch_size]
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


def graph_reward_line(reward_arr_exp, reward_arr_gen, steps, name,name_exp="Expert", name_gen="Gen", save_path=None):
    # 计算均值和上下分位
    reward_arr_exp = np.array(reward_arr_exp).squeeze()
    reward_arr_gen = np.array(reward_arr_gen).squeeze()
    steps = np.arange(1, steps+1)
    mean_exp = reward_arr_exp.mean(axis=1)
    q25_exp = np.percentile(reward_arr_exp, 25, axis=1)
    q75_exp = np.percentile(reward_arr_exp, 75, axis=1)

    mean_gen = reward_arr_gen.mean(axis=1)
    q25_gen = np.percentile(reward_arr_gen, 25, axis=1)
    q75_gen = np.percentile(reward_arr_gen, 75, axis=1)

    plt.figure(figsize=(12, 6))

    # Expert
    plt.fill_between(steps, q25_exp, q75_exp, color='tab:blue', alpha=0.2, label=f"{name_exp} IQR (25~75%)")
    plt.plot(steps, mean_exp, '-', color='tab:blue', label=f"{name_exp} Mean")

    # Gen
    plt.fill_between(steps, q25_gen, q75_gen, color='tab:orange', alpha=0.2, label=f"{name_gen} IQR (25~75%)")
    plt.plot(steps, mean_gen, '-', color='tab:orange', label=f"{name_gen} Mean")

    plt.xlabel("Discriminator Step")
    plt.ylabel("Reward")
    plt.title(f"Reward Trend: {name_exp} vs {name_gen}")
    plt.legend()
    plt.grid(axis='y')
    plt.savefig(f"reward_line{name}.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    # thread = threading.Thread(target=start_external_program)
    # thread.start()
    SEED = 43
    set_seed(SEED)

    # config = {
    #     "policy_type": "MlpPolicy",
    #     "total_timesteps": 250000,
    #     "env_id": "airsim",
    # }
    #
    # register(
    #     id='AirSim/AirSimEnv',
    #     entry_point='airsim_env:AirSimEnv_v3',  # 指向自定义环境的模块和类
    #     max_episode_steps=300,
    # )
    #
    # env = make_vec_env(
    #     'AirSim/AirSimEnv',
    #     rng=np.random.default_rng(SEED),
    #     n_envs=1,
    #     post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
    # )

    # policy = PPO.load("model/airl/policy/ppo_policy_8.zip")
    logs = torch.load("model/airl/log/logs_hc_airl64", weights_only=False)

    # env.seed(SEED)
    # env.seed(SEED)
    # learner_rewards_after_training, _ = evaluate_policy(
    #     policy, env, 30, return_episode_rewards=True,
    # # )
    # print("mean reward after training:", np.mean(learner_rewards_after_training))

    # graph_reward_box(logs["exp_reward"], logs["step"], "expert-mountain")
    # graph_reward_box(logs["gen_reward"], logs["step"], "gen-mountain")

    graph_reward_line(logs["exp_reward"], logs["gen_reward"], logs["step"], "halfcheetah")
    # graph_reward_line(logs["gen_reward"], logs["step"], "gen-mountain")
