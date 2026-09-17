import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mujoco
import pickle

# ======================
# 策略网络（BC和PPO共用）
# ======================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        # Actor：输出动作均值
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim)
        )
        # Critic：状态价值
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def get_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        mu = self.actor(obs)
        std = torch.ones_like(mu) * 0.2
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        value = self.critic(obs)
        return action.detach().numpy(), log_prob, value

    def evaluate(self, obs, actions):
        mu = self.actor(obs)
        std = torch.ones_like(mu) * 0.2
        dist = torch.distributions.Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)
        values = self.critic(obs).squeeze()
        return log_probs, values, entropy

    def bc_forward(self, obs):
        """BC阶段只用actor输出动作"""
        return self.actor(obs)


# ======================
# BC预训练模块
# ======================
class BCPretrain:
    def __init__(self, ac_model, lr=1e-3):
        self.model = ac_model
        self.optimizer = optim.Adam(self.model.actor.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train_step(self, obs_batch, act_batch):
        pred_act = self.model.bc_forward(obs_batch)
        loss = self.loss_fn(pred_act, act_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


# ======================
# PPO微调模块
# ======================
class PPO:
    def __init__(self, ac_model, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.ac = ac_model
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)

    def update(self, obs_buf, act_buf, log_prob_buf, reward_buf, done_buf):
        obs = torch.tensor(obs_buf, dtype=torch.float32)
        acts = torch.tensor(act_buf, dtype=torch.float32)
        log_probs_old = torch.tensor(log_prob_buf, dtype=torch.float32)
        rewards = torch.tensor(reward_buf, dtype=torch.float32)
        dones = torch.tensor(done_buf, dtype=torch.float32)

        values = self.ac.critic(obs).squeeze()
        advantages = torch.zeros_like(rewards)
        last_adv = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1]*(1-dones[t]) - values[t]
            advantages[t] = delta + self.gamma * 0.95 * (1-dones[t]) * last_adv
            last_adv = advantages[t]
        returns = advantages + values

        log_probs_new, values_new, entropy = self.ac.evaluate(obs, acts)
        ratio = torch.exp(log_probs_new - log_probs_old)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = ((values_new - returns)**2).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


# ======================
# MuJoCo人形环境
# ======================
class HumanoidEnv:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.obs_dim = self.model.nq + self.model.nv
        self.act_dim = self.model.nu

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        return obs

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = action * 15.0
        mujoco.mj_step(self.model, self.data)
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        forward_vel = self.data.qvel[0]
        reward = forward_vel - 0.01 * np.sum(np.square(action))
        done = self.data.qpos[2] < 0.6
        return obs, reward, done


# ======================
# 加载专家数据集
# ======================
def load_expert_dataset(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    obs = torch.tensor(data["observations"], dtype=torch.float32)
    acts = torch.tensor(data["actions"], dtype=torch.float32)
    return obs, acts


# ======================
# 主流程：BC预训练 → PPO微调
# ======================
if __name__ == "__main__":
    model_path = "./models/humanoid.xml"
    env = HumanoidEnv(model_path)
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # 1. 初始化Actor‑Critic网络
    ac = ActorCritic(obs_dim, act_dim)

    # ========== 阶段1：BC预训练，克隆专家动作 ==========
    print("===== BC预训练阶段 =====")
    obs_expert, act_expert = load_expert_dataset("./dataset/g1_motion.pkl")
    bc_trainer = BCPretrain(ac, lr=1e-3)

    bc_epochs = 300
    batch_size = 128
    dataset_size = len(obs_expert)
    for epoch in range(bc_epochs):
        total_loss = 0.0
        for i in range(0, dataset_size, batch_size):
            obs_batch = obs_expert[i:i+batch_size]
            act_batch = act_expert[i:i+batch_size]
            loss = bc_trainer.train_step(obs_batch, act_batch)
            total_loss += loss
        avg_loss = total_loss / ((dataset_size // batch_size)+1)
        if epoch % 20 == 0:
            print(f"BC Epoch {epoch:3d} | Loss: {avg_loss:.6f}")

    torch.save(ac.state_dict(), "bc_pretrained.pt")
    print("✅ BC预训练完成，保存 bc_pretrained.pt")

    # ========== 阶段2：PPO在线微调，修正分布偏移 ==========
    print("\n===== PPO微调阶段 =====")
    ppo = PPO(ac, lr=3e-4)
    max_episodes = 1500

    for ep in range(max_episodes):
        obs = env.reset()
        obs_buf, act_buf, log_prob_buf, reward_buf, done_buf = [], [], [], [], []
        ep_reward = 0
        for step in range(1000):
            act, logp, val = ac.get_action(obs)
            next_obs, r, done = env.step(act)

            obs_buf.append(obs)
            act_buf.append(act)
            log_prob_buf.append(logp)
            reward_buf.append(r)
            done_buf.append(done)

            obs = next_obs
            ep_reward += r
            if done:
                break
        loss = ppo.update(obs_buf, act_buf, log_prob_buf, reward_buf, done_buf)
        if ep % 20 == 0:
            print(f"PPO Episode {ep:4d} | Reward:{ep_reward:.2f} | Loss:{loss:.4f}")

    torch.save(ac.state_dict(), "bc_rl_final.pt")
    print("✅ BC‑RL训练完成，保存 bc_rl_final.pt")
