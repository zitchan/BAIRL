from typing import Optional, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common import base_class, policies, vec_env
from stable_baselines3.sac import policies as sac_policies
from imitation.util import logger
from stable_baselines3.ppo import MlpPolicy
import numpy as np
from stable_baselines3.ppo import PPO
from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicShapedRewardNet, NormalizedRewardNet, RewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from imitation.algorithms import base
from imitation.algorithms.adversarial import common
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.rewards import reward_nets
from typing import Optional, Sequence
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
STOCHASTIC_POLICIES = (sac_policies.SACPolicy, policies.ActorCriticPolicy)


def dynamic_lambda_reg(disc_loss, target=0.693, scale=0.2, min_val=0.01, max_val=1.5):
    diff = target - disc_loss
    lam = scale * np.tanh(diff * 3) + scale  # diff=0时lambda=scale，diff越大lambda到2*scale
    lam = np.clip(lam, min_val, max_val)
    return float(lam)



class custom_AIRL(AIRL):
    def __init__(
            self,
            *,
            demonstrations: base.AnyTransitions,
            demo_batch_size: int,
            venv: vec_env.VecEnv,
            gen_algo: base_class.BaseAlgorithm,
            reward_net: reward_nets.RewardNet,
            reg: str,
            lambda_reg: float,
            **kwargs,
    ):
        super().__init__(
            demonstrations=demonstrations,
            demo_batch_size=demo_batch_size,
            venv=venv,
            gen_algo=gen_algo,
            reward_net=reward_net,
            **kwargs,
        )
        if not isinstance(self.gen_algo.policy, STOCHASTIC_POLICIES):
            raise TypeError(
                "AIRL needs a stochastic policy to compute the discriminator output.",
            )
        self.reg = reg
        if self.reg == "AA":
            if lambda_reg is not None:
                self.lambda_reg = lambda_reg
            else:
                self.lambda_reg = 1e-2
        elif self.reg == "B":
            if lambda_reg is not None:
                self.lambda_reg = lambda_reg
            else:
                self.lambda_reg = 1
        else:
            if lambda_reg is not None:
                self.lambda_reg = lambda_reg
            else:
                self.lambda_reg = 30

        self.logger = logger.configure('log', ["stdout", "tensorboard"])
        self.reward_net = reward_net
        self.reward_gen = []
        self.reward_exp = []
        self.reward_steps = []
        self.lambda_regs = []
        self.reg_list = []
        self.binary_losses = []

    def get_reward(self, batch):
        batch_size = len(batch["state"]) // 2

        # expert 部分
        s_exp = batch["state"][:batch_size]
        a_exp = batch["action"][:batch_size]
        s_next_exp = batch["next_state"][:batch_size]
        d_exp = batch["done"][:batch_size]
        # rollout/generator 部分
        s_gen = batch["state"][batch_size:]
        a_gen = batch["action"][batch_size:]
        s_next_gen = batch["next_state"][batch_size:]
        d_gen = batch["done"][:batch_size]

        rewards_exp = self.reward_train(s_exp, a_exp, s_next_exp,
                                        d_exp,
                                        )
        rewards_exp = rewards_exp.detach().cpu().numpy().flatten()

        rewards_gen = self.reward_train(s_gen, a_gen, s_next_gen,
                                        d_gen)
        rewards_gen = rewards_gen.detach().cpu().numpy().flatten()

        return rewards_exp, rewards_gen

    def train_disc(
            self,
            *,
            expert_samples: Optional[Mapping] = None,
            gen_samples: Optional[Mapping] = None,
    ) -> Mapping[str, float]:
        with self.logger.accumulate_means("disc"):
            # optionally write TB summaries for collected ops
            write_summaries = self._init_tensorboard and self._global_step % 20 == 0

            # compute loss
            self._disc_opt.zero_grad()

            batch_iter = self._make_disc_train_batches(
                gen_samples=gen_samples,
                expert_samples=expert_samples,
            )
            b_r_exp = []
            b_r_gen = []
            for batch in batch_iter:
                disc_logits = self.logits_expert_is_high(
                    batch["state"],
                    batch["action"],
                    batch["next_state"],
                    batch["done"],
                    batch["log_policy_act_prob"],
                )
                r_exp, r_gen = self.get_reward(batch)
                # reg = self.compute_Gradient_Penalty(batch, lambda_gp=10)
                # reg = self.compute_Adversarial_Augmentation(batch)
                if self.lambda_reg != 0 :
                    reward, reg = self.compute_uncertance(batch)
                else:
                    reg = torch.tensor(0)
                loss = F.binary_cross_entropy_with_logits(
                    disc_logits,
                    batch["labels_expert_is_one"].float(),
                )
                # self.lambda_reg = dynamic_lambda_reg(loss.item())
                self.lambda_regs.append(self.lambda_reg)

                reg = reg * self.lambda_reg
                self.binary_losses.append(loss.item())

                loss = loss + reg
                self.reg_list.append(reg.item())

                assert len(batch["state"]) == 2 * self.demo_minibatch_size
                loss *= self.demo_minibatch_size / self.demo_batch_size
                loss.backward()
            b_r_exp.append(r_exp)
            b_r_gen.append(r_gen)

            if np.mean(self.binary_losses) > 0.69:
                self.lambda_reg = self.lambda_reg
            else:
                self.lambda_reg = self.lambda_reg * 0.8

            # do gradient step
            self._disc_opt.step()
            self._disc_step += 1
            self.reward_exp.append(b_r_exp)
            self.reward_gen.append(b_r_gen)
            self.reward_steps.append(self._disc_step)
            logs = {"exp_reward": self.reward_exp, "gen_reward": self.reward_gen, "step": self._disc_step}
            if self._disc_step % 4 == 0 and self._disc_step != 0:
                torch.save(logs, f"model/airl/log/logs_hc_bairl")
            # compute/write stats and TensorBoard data
            with torch.no_grad():
                train_stats = common.compute_train_stats(
                    disc_logits,
                    batch["labels_expert_is_one"],
                    loss,
                )
            self.logger.record("global_step", self._global_step)
            self.logger.record("reg", np.mean(self.reg_list))
            self.logger.record('lambda_reg', np.mean(self.lambda_regs))
            self.logger.record('binary_cross_entropy_loss', np.mean(self.binary_losses))
            for k, v in train_stats.items():
                self.logger.record(k, v)
            self.logger.dump(self._disc_step)
            if write_summaries:
                self._summary_writer.add_histogram("disc_logits", disc_logits.detach())

        return train_stats

    def compute_uncertance(self, batch):
        s = batch["state"].detach().clone().requires_grad_()
        a = batch["action"].detach().clone().requires_grad_()
        # print(s.shape, a.shape)
        s_next = batch["next_state"].detach().clone().requires_grad_()

        combined_input = torch.cat([s, a], dim=-1)
        r_mean, r_std = self.reward_net.mlp.sample_reward(combined_input, n_samples=20)
        old_shaping_mean, old_shaping_std = self.reward_net.potential.sample_reward(s, n_samples=20)
        new_shaping_mean, new_shaping_std = self.reward_net.potential.sample_reward(s_next, n_samples=20)
        shaped = self.reward_net.discount_factor * new_shaping_mean - old_shaping_mean
        final_reward = r_mean + shaped
        final_penalty = r_std.mean() + old_shaping_std.mean() + new_shaping_std.mean()
        return final_reward, final_penalty

    # def compute_Adversarial_Augmentation(self, batch, eps=1e-2):
    #     s = batch["state"].detach().clone().requires_grad_()
    #     a = batch["action"].detach().clone().requires_grad_()
    #     s_next = batch["next_state"].detach().clone().requires_grad_()
    #     done = batch["done"].detach().clone().requires_grad_()
    #     output = self.reward_net(s, a, s_next, done)
    #     grad_s = torch.autograd.grad(outputs=output.sum(), inputs=s, create_graph=True)[0]
    #     grad_a = torch.autograd.grad(outputs=output.sum(), inputs=a, create_graph=True)[0]
    #     grad_s_next = torch.autograd.grad(outputs=output.sum(), inputs=s_next, create_graph=True)[0]
    #
    #     s_adv = s + eps * grad_s.sign()
    #     a_adv = a + eps * grad_a.sign()
    #     s_next_adv = s_next + eps * grad_s_next.sign()
    #
    #     output_adv = self.reward_net(s_adv, a_adv, s_next_adv, done)
    #     adv_reg = torch.mean((output_adv - output).pow(2))
    #     return adv_reg
    #
    # def compute_Gradient_Penalty(self, batch,
    #                              expert_samples: Optional[Mapping] = None,
    #                              gen_samples: Optional[Mapping] = None):
    #
    #     batch_size = len(batch['state']) // 2
    #     s_exp = batch["state"][:batch_size]
    #     a_exp = batch["action"][:batch_size]
    #     s_next_exp = batch["next_state"][:batch_size]
    #
    #     s_gen = batch["state"][batch_size:]
    #     a_gen = batch["action"][batch_size:]
    #     s_next_gen = batch["next_state"][batch_size:]
    #
    #     epsilon = torch.rand(batch_size, 1, device=s_exp.device)
    #     # 适配维度，如果 s 是 [batch, obs_dim]，可直接乘法
    #     s_interp = epsilon * s_exp + (1 - epsilon) * s_gen
    #     a_interp = epsilon * a_exp + (1 - epsilon) * a_gen
    #     s_next_interp = epsilon * s_next_exp + (1 - epsilon) * s_next_gen
    #     s_interp.requires_grad_()
    #     a_interp.requires_grad_()
    #     s_next_interp.requires_grad_()
    #
    #     disc_interp = self.logits_expert_is_high(
    #         s_interp,
    #         a_interp,
    #         s_next_interp,
    #         torch.zeros_like(batch["done"][:batch_size]),
    #         batch["log_policy_act_prob"][:batch_size],
    #     )
    #     grad = torch.autograd.grad(
    #         outputs=disc_interp,
    #         inputs=(s_interp, a_interp, s_next_interp),
    #         grad_outputs=torch.ones_like(disc_interp),
    #         create_graph=True
    #     )
    #     grad_cat = torch.cat([g.view(batch_size, -1) for g in grad], dim=1)
    #     grad_norm = grad_cat.norm(2, dim=1)
    #     gp = ((grad_norm - 1) ** 2).mean()
    #     print(gp)
    #     gradient_penalty = gp
    #     # print(gradient_penalty)
    #     return gradient_penalty