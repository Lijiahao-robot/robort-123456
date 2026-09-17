import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mujoco
import mujoco.viewer

# ---------------------------
# 策略网络 + 价值网络
# ---------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim)
        )
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


# ---------------------------
# PPO核心
# ---------------------------
class PPO:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.ac = ActorCritic(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)

    def update(self, obs_buf, act_buf, log_prob_buf, reward_buf, done_buf):
        obs = torch.tensor(obs_buf, dtype=torch.float32)
        acts = torch.tensor(act_buf, dtype=torch.float32)
        log_probs_old = torch.tensor(log_prob_buf, dtype=torch.float32)
        rewards = torch.tensor(reward_buf, dtype=torch.float32)
        dones = torch.tensor(done_buf, dtype=torch.float32)

        # 计算GAE优势
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
        loss = actor_loss + 0.5*critic_loss - 0.01*entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


# ---------------------------
# MuJoCo人形仿真环境
# ---------------------------
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
        self.data.ctrl[:] = action * 15.0  # 缩放力矩
        mujoco.mj_step(self.model, self.data)

        obs = np.concatenate([self.data.qpos, self.data.qvel])
        # 奖励：向前走，惩罚摔倒
        forward_vel = self.data.qvel[0]
        reward = forward_vel - 0.01 * np.sum(np.square(action))
        # 摔倒判定：基座z高度过低
        done = self.data.qpos[2] < 0.6
        return obs, reward, done


if __name__ == "__main__":
    env = HumanoidEnv("./models/humanoid.xml")
    ppo = PPO(obs_dim=env.obs_dim, act_dim=env.act_dim)

    max_episodes = 2000
    for ep in range(max_episodes):
        obs = env.reset()
        obs_buf, act_buf, log_prob_buf, reward_buf, done_buf = [], [], [], [], []
        ep_reward = 0
        for step in range(1000):
            act, logp, val = ppo.ac.get_action(obs)
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
            print(f"Episode {ep}, reward:{ep_reward:.2f}, loss:{loss:.4f}")

    # 保存训练权重
    torch.save(ppo.ac.state_dict(), "ppo_humanoid.pt")
