import torch
import numpy as np
import mujoco
import mujoco.viewer
from controllers.wbc_qp import WBC_QP
from rl.bc_rl import ActorCritic

# ======================
# 仿真环境封装
# ======================
class HumanoidWBCEnv:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.obs_dim = self.model.nq + self.model.nv
        self.act_dim = self.model.nu
        # 初始化WBC全身控制器
        self.wbc = WBC_QP(self.model, self.data)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        return obs

    def step(self, target_joint_pos):
        """
        target_joint_pos: BC‑RL输出的目标关节角度 (nu,)
        返回：obs, reward, done
        """
        # 把目标关节位置转为期望关节速度（简单微分，也可以用轨迹生成）
        desired_joint_vel = (target_joint_pos - self.data.qpos[7:]) / 0.01

        # 基座保持静止：期望基座速度全0
        desired_base_vel = np.zeros(6)

        # WBC求解关节力矩
        tau = self.wbc.solve(desired_base_vel, desired_joint_vel)
        self.data.ctrl[:] = tau

        mujoco.mj_step(self.model, self.data)

        obs = np.concatenate([self.data.qpos, self.data.qvel])
        # 奖励函数：向前速度 + 惩罚力矩大小 + 惩罚姿态倾斜
        forward_vel = self.data.qvel[0]
        reward = forward_vel - 0.005 * np.sum(np.square(tau))
        done = self.data.qpos[2] < 0.6  # 摔倒判定

        return obs, reward, done


# ======================
# 主程序：加载训练好的BC‑RL权重 + WBC闭环仿真
# ======================
if __name__ == "__main__":
    model_path = "./models/humanoid.xml"
    env = HumanoidWBCEnv(model_path)

    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # 加载已经训练完成的 BC‑RL 模型
    ac = ActorCritic(obs_dim, act_dim)
    ac.load_state_dict(torch.load("bc_rl_final.pt", map_location=torch.device("cpu")))
    ac.eval()

    obs = env.reset()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            # BC‑RL 推理：输入观测，输出目标关节位置参考
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                target_joint_pos = ac.actor(obs_tensor).squeeze().cpu().numpy()

            # 限制关节范围，防止超出模型物理极限
            target_joint_pos = np.clip(target_joint_pos, -1.5, 1.5)

            obs, reward, done = env.step(target_joint_pos)

            if done:
                obs = env.reset()

            viewer.sync()
