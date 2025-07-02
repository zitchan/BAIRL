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
from stable_baselines3.common.vec_env import VecNormalize as _VecNormalize
from huggingface_sb3 import load_from_hub


if __name__ == '__main__':
    SEED = 42
    set_seed(SEED)

    env = make_vec_env(
        "Ant-v4",
        rng=np.random.default_rng(SEED),
        n_envs=1,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
    )

    env = VecNormalize(env,gamma=0.995 ,norm_obs=False, norm_reward=True)

    policy_kwags = {
        "log_std_init": -1,
        "activation_fn": nn.Tanh,
        "features_extractor_class": NormalizeFeaturesExtractor,
        'net_arch': dict(pi=[64, 64], vf=[64, 64])
    }
    learner = PPO(
                    env=env,
                    policy=MlpPolicy,
                    batch_size=16,
                    n_steps=2048,
                    ent_coef=3.144e-06,
                    gae_lambda=0.8,
                    learning_rate=0.0001,
                    gamma=0.995,
                    clip_range=0.3,
                    vf_coef=0.435,
                    n_epochs=10,
                    normalize_advantage=True,
                    seed=SEED,
                    policy_kwargs=policy_kwags,
                    max_grad_norm=0.9,
                    tensorboard_log="./log/",
                    verbose=1,
                   )
    learner.learn(total_timesteps=int(1e+6))
    learner.save("./model/expert/ppo-test.zip")

    # Reload VecNormalize statistics
    learner = PPO.load("./model/expert/ppo-test.zip", env=env)
    learner_rewards_after_training, _ = evaluate_policy(
        learner, env, 100, return_episode_rewards=True,
    )
    # learner.save("./model/expert/ppo-halfcheetah-v4")
    print("mean reward after training:", np.mean(learner_rewards_after_training))