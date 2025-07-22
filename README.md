# BAIRL
The Bayesian inverse reinforcement learning experiment achieved the initial stable performance of the discriminator by introducing mc-dropout in AIRL.


## 🚀 Requirement
gymnasium >= 0.29.1<br>
stablebaseline3 = 2.2.1<br>
torch >= 2.6.0<br>
imitation = 1.0.1<br>


## 🔥 Training
set env <br>
Ant: <br>
policy Parameters: https://huggingface.co/HumanCompatibleAI/ppo-seals-Ant-v0
reward net: <br> demo_batch_size=2048, gen_replay_buffer_capacity=512, n_disc_updates_per_round=4, lambda_reg=0.01 decline every 20,000 steps.<br>

humanoid: <br>
policy Parameters: https://huggingface.co/HumanCompatibleAI/ppo-seals-humanoid-v0
reward net: <br> demo_batch_size=2048, gen_replay_buffer_capacity=1024, n_disc_updates_per_round=8, lambda_reg=0.01 decline every 20,000 steps.<br>
