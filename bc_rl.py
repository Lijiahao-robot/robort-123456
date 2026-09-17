
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np

# -------------------------- Actor‑Critic网络 --------------------------
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
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action = self.actor(obs)
        return action.detach().numpy()[0]

    def evaluate(self, obs, actions):
        logits = self.actor(obs)
        value = self.critic(obs)
        return logits, value

# -------------------------- BC行为克隆预训练 --------------------------
def bc_pretrain(model, expert_path="expert_dataset.pkl", epochs=100, lr=1e-3, device="cpu"):
    """
    加载LIPM‑DCM生成的专家数据集做BC预训练
    expert_dataset.pkl 格式: {"observations": (N,obs_dim), "actions":(N,act_dim)}
    """
    print(f"===== 开始BC预训练，加载专家数据集 {expert_path} =====")
    with open(expert_path, "rb") as f:
        data = pickle.load(f)
    obs_expert = torch.tensor(data["observations"], dtype=torch.float32).to(device)
    act_expert = torch.tensor(data["actions"], dtype=torch.float32).to(device)

    optimizer = optim.Adam(model.actor.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        pred_act = model.actor(obs_expert)
        loss = loss_fn(pred_act, act_expert)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"BC epoch [{epoch+1}/{epochs}], loss={loss.item():.6f}")
    print("===== BC预训练完成 =====")

# -------------------------- PPO训练器 --------------------------
class PPOAgent:
    def __init__(self, obs_dim, act_dim,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, update_epochs=10, mini_batch_size=64,
                 device="cpu"):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size

        self.model = ActorCritic(obs_dim, act_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # 存储rollout数据
        self.obs_buf = []
        self.act_buf = []
        self.logprob_buf = []
        self.reward_buf = []
        self.done_buf = []
        self.value_buf = []

    def collect_step(self, obs, action, logprob, reward, done, value):
        self.obs_buf.append(obs)
        self.act_buf.append(action)
        self.logprob_buf.append(logprob)
        self.reward_buf.append(reward)
        self.done_buf.append(done)
        self.value_buf.append(value)

    def compute_gae(self):
        advantages = []
        returns = []
        last_advantage = 0
        last_return = 0

        for t in reversed(range(len(self.reward_buf))):
            delta = self.reward_buf[t] + self.gamma * last_return * (1 - self.done_buf[t]) - self.value_buf[t]
            advantage = delta + self.gamma * self.gae_lambda * (1 - self.done_buf[t]) * last_advantage
            last_advantage = advantage
            last_return = self.reward_buf[t] + self.gamma * last_return * (1 - self.done_buf[t])
            advantages.append(advantage)
            returns.append(last_return)

        advantages = torch.tensor(list(reversed(advantages)), dtype=torch.float32).to(self.device)
        returns = torch.tensor(list(reversed(returns)), dtype=torch.float32).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self):
        obs = torch.tensor(self.obs_buf, dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.act_buf, dtype=torch.float32).to(self.device)
        old_logprobs = torch.tensor(self.logprob_buf, dtype=torch.float32).to(self.device)

        advantages, returns = self.compute_gae()

        dataset_size = len(obs)
        indices = np.arange(dataset_size)

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]

                mb_obs = obs[mb_idx]
                mb_act = actions[mb_idx]
                mb_old_logprob = old_logprobs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]

                pred_act, value = self.model.evaluate(mb_obs, mb_act)
                # 简化高斯策略
                new_logprob = -0.5 * ((pred_act - mb_act)**2).sum(dim=1)
                ratio = torch.exp(new_logprob - mb_old_logprob)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.MSELoss()(value.squeeze(), mb_ret)
                total_loss = actor_loss + 0.5 * critic_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        # 清空缓冲区
        self.obs_buf.clear()
        self.act_buf.clear()
        self.logprob_buf.clear()
        self.reward_buf.clear()
        self.done_buf.clear()
        self.value_buf.clear()

    def save(self, path="bc_rl_final.pt"):
        torch.save(self.model.state_dict(), path)
        print(f"模型保存至 {path}")

    def load(self, path="bc_rl_final.pt"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"加载模型 {path}")


# -------------------------- 训练主函数示例 --------------------------
if __name__ == "__main__":
    # 示例：先BC预训练，再PPO训练
    obs_dim = 12
    act_dim = 6
    device = "cpu"

    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, device=device)

    # 1. BC预训练，加载LIPM‑DCM专家数据集
    bc_pretrain(agent.model, expert_path="expert_dataset.pkl", epochs=100, lr=1e-3, device=device)

    # 2. 保存BC预训练权重
    agent.save(path="bc_pretrain.pt")

    # 3. 之后就可以接入环境做PPO强化学习微调
    # env = HumanoidDCMEnv("./models/humanoid.xml")
    # 循环收集rollout，调用 agent.collect_step()，每隔N步 agent.update()
