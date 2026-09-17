import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mujoco

# ---------------------------
# BC策略网络：观测输入，输出关节动作
# ---------------------------
class BCPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, obs):
        return self.net(obs)

# ---------------------------
# BC训练器
# ---------------------------
class BehaviorCloning:
    def __init__(self, obs_dim, act_dim, lr=1e-3):
        self.policy = BCPolicy(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train_step(self, obs_batch, act_batch):
        """单步训练：观测，专家动作"""
        pred_act = self.policy(obs_batch)
        loss = self.loss_fn(pred_act, act_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_action(self, obs):
        """推理：输入观测，输出预测关节动作"""
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            act = self.policy(obs_tensor).squeeze().cpu().numpy()
        return act

# ---------------------------
# 数据集加载（适配G1‑Moves pkl动作数据集）
# ---------------------------
def load_expert_dataset(pkl_path):
    """
    数据集格式：
    {
        "observations": (N, obs_dim),
        "actions": (N, act_dim)
    }
    obs：qpos+qvel；action：目标关节角度/力矩
    """
    import pickle
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    obs = torch.tensor(data["observations"], dtype=torch.float32)
    acts = torch.tensor(data["actions"], dtype=torch.float32)
    return obs, acts

# ---------------------------
# MuJoCo仿真环境（BC推理测试）
# ---------------------------
class HumanoidBCEnv:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.obs_dim = self.model.nq + self.model.nv
        self.act_dim = self.model.nu

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        return obs

    def step(self, target_joint_pos):
        # PD控制跟踪BC输出的目标关节位置
        kp = 80.0
        kd = 4.0
        torque = kp * (target_joint_pos - self.data.qpos[7:]) - kd * self.data.qvel[6:]
        self.data.ctrl[:] = torque
        mujoco.mj_step(self.model, self.data)
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        done = self.data.qpos[2] < 0.6
        return obs, done

# ---------------------------
# 训练主程序
# ---------------------------
if __name__ == "__main__":
    # 1. 加载专家数据集（G1‑Moves动作pkl）
    obs_expert, act_expert = load_expert_dataset("./dataset/g1_motion.pkl")
    obs_dim = obs_expert.shape[1]
    act_dim = act_expert.shape[1]

    # 2. 初始化BC
    bc = BehaviorCloning(obs_dim, act_dim, lr=1e-3)

    # 3. 训练循环
    epochs = 500
    batch_size = 128
    dataset_size = len(obs_expert)

    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(0, dataset_size, batch_size):
            obs_batch = obs_expert[i:i+batch_size]
            act_batch = act_expert[i:i+batch_size]
            loss = bc.train_step(obs_batch, act_batch)
            total_loss += loss
        avg_loss = total_loss / (dataset_size // batch_size + 1)
        if epoch % 20 == 0:
            print(f"Epoch {epoch:4d} | Loss: {avg_loss:.6f}")

    # 保存模型权重
    torch.save(bc.policy.state_dict(), "bc_humanoid.pt")
    print("✅ BC模型保存完成 bc_humanoid.pt")

    # 4. 仿真测试BC推理
    env = HumanoidBCEnv("./models/humanoid.xml")
    obs = env.reset()

    import mujoco.viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            pred_action = bc.get_action(obs)
            obs, done = env.step(pred_action)
            if done:
                obs = env.reset()
            viewer.sync()
