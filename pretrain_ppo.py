import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import subprocess
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from imitation.data import serialize
from torch.utils.data import Dataset, DataLoader
import matplotlib
# matplotlib.use('TkAgg')
torch.backends.cudnn.benchmark = True


def start_external_program(exe_path):
    subprocess.Popen([exe_path, '-windowed', '-ResX=1280', '-ResY=720'])


def load_data(data):
    # expert (high performance)
    # reward_net = torch.load('./model/airl/reward/reward_net65.pth', weights_only=False
    #                         , map_location=torch.device("cuda:0"))
    # reward_net.eval()
    if data is not None:
        bc_data = data
    else:
        bc_path = './model/expert_data_all.pkl'
        bc_data = serialize.load(bc_path)

    h_obs = []
    h_acts = []
    h_rewards = []
    h_next_obs = []
    h_done = []
    h_step = []
    for traj in tqdm(bc_data, desc='Expert traj. loding'):
        obs = traj.obs[:-1]
        length = obs.shape[0]
        acts = traj.acts
        next_obs = traj.obs[1:]
        dones = np.zeros(length, dtype='float')
        dones[-1] = 1.0
        rewards = np.zeros(length, dtype='float')
        h_obs.append(obs)
        h_acts.append(acts)
        h_rewards.append(rewards)
        h_next_obs.append(next_obs)
        h_done.append(dones)
        h_step.append(length)
    obs = np.concatenate(h_obs, axis=0)
    acts = np.concatenate(h_acts, axis=0)
    rewards = np.concatenate(h_rewards, axis=0)
    next_obs = np.concatenate(h_next_obs, axis=0)
    done = np.concatenate(h_done, axis=0)

    print(f'successfully loaded {len(obs)}')
    h_data = {
        'obs': obs,
        'acts': acts,
        'rewards': rewards,
        'next_obs': next_obs,
        'dones': done,
    }

    return h_data


class MDPData(Dataset):
    def __init__(self, transition, device):
        self.data = transition
        self.device = device

    def get_distribution(self):
        return np.mean(self.data['rewards']), np.std(self.data['rewards'])

    def __len__(self):
        return len(self.data['obs'])

    def __getitem__(self, idx):
        obs = self.data['obs'][idx]
        action = self.data['acts'][idx]
        next_obs = self.data['next_obs'][idx]
        dones = self.data['dones'][idx]
        reward = self.data['rewards'][idx]

        mean, std = self.get_distribution()
        reward = (reward - mean) / (std + 1e-6)

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)

        return obs, action, reward, next_obs, dones


def pretrain(learner, env, n_epoch, data=None, testing=False, testing_episode=1, figure=True):

    expert_data = load_data(data)
    batch_size = learner.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MDPData(expert_data, device)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0)

    policy = learner.policy.to(device)
    policy.log_std.requires_grad = False
    optimizer = optim.Adam(policy.parameters(), lr=5e-4)
    losses = []
    best_loss = float('inf')

    for epoch in tqdm(range(n_epoch), desc="pretraining"):
        total_loss = 0.0
        for obs, action, reward, next_obs, dones in dataloader:
            feats = policy.features_extractor(obs)
            latent_pi, _ = policy.mlp_extractor(feats)
            dist = policy._get_action_dist_from_latent(latent_pi)

            log_prob = dist.log_prob(action)  # (B, A)
            log_prob = log_prob.sum(dim=-1)  # joint log‐prob, shape (B,)
            loss = -log_prob.mean()  # scalar
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * obs.size(0)

        epoch_loss = total_loss / len(dataset)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            learner.save("./model/bc_pretrain/best_bcmodel")
        losses.append(epoch_loss)
        # print(f"[Epoch {epoch + 1}/{n_epoch}] loss={epoch_loss:.4f}")/

    learner.save("./model/bc_pretrain/pretrain_model")
    if figure:
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label="Training Loss", color="blue")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        # plt.ylim(-3, -0)
        plt.title("Loss Curve During Training")
        plt.legend()
        plt.grid()
        plt.show()

    if testing:
        log_std = learner.policy.log_std
        print("Original log_std:", log_std)
        for episode in tqdm(range(testing_episode), desc="Pretrain testing"):
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                # 使用模型选择动作
                action, _states = learner.predict(obs, deterministic=False)
                obs, reward, terminated, info = env.step(action)
                done = terminated
                total_reward += reward
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    policy.log_std.requires_grad = True
    return learner
