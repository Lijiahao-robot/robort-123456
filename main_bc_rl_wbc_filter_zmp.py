import torch
import numpy as np
import mujoco
import mujoco.viewer
from controllers.wbc_qp import WBC_QP
from controllers.zmp import ZMPCalculator
from rl.bc_rl import ActorCritic


class LowPassFilter:
    def __init__(self, alpha=0.1, dim=None):
        self.alpha = alpha
        self.prev_val = None
        self.dim = dim

    def reset(self):
        self.prev_val = None

    def step(self, x):
        if self.prev_val is None:
            self.prev_val = x.copy()
        self.prev_val = self.alpha * x + (1 - self.alpha) * self.prev_val
        return self.prev_val.copy()


class HumanoidWBCEnv:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.obs_dim = self.model.nq + self.model.nv
        self.act_dim = self.model.nu
        self.wbc = WBC_QP(self.model, self.data)
        self.lpf = LowPassFilter(alpha=0.15, dim=self.act_dim)
        self.zmp_calc = ZMPCalculator(self.model, self.data, ground_z=0.0)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.lpf.reset()
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        return obs

    def step(self, raw_target_joint_pos):
        smooth_target = self.lpf.step(raw_target_joint_pos)

        desired_joint_vel = (smooth_target - self.data.qpos[7:]) / 0.01
        desired_base_vel = np.zeros(6)

        tau = self.wbc.solve(desired_base_vel, desired_joint_vel)
        self.data.ctrl[:] = tau

        mujoco.mj_step(self.model, self.data)

        obs = np.concatenate([self.data.qpos, self.data.qvel])
        forward_vel = self.data.qvel[0]
        reward = forward_vel - 0.005 * np.sum(np.square(tau))
        done = self.data.qpos[2] < 0.6

        # 计算ZMP
        zmp = self.zmp_calc.compute_zmp()
        return obs, reward, done, zmp


if __name__ == "__main__":
    model_path = "./models/humanoid.xml"
    env = HumanoidWBCEnv(model_path)

    obs_dim = env.obs_dim
    act_dim = env.act_dim

    ac = ActorCritic(obs_dim, act_dim)
    ac.load_state_dict(torch.load("bc_rl_final.pt", map_location=torch.device("cpu")))
    ac.eval()

    obs = env.reset()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                raw_target = ac.actor(obs_tensor).squeeze().cpu().numpy()

            raw_target = np.clip(raw_target, -1.5, 1.5)
            obs, reward, done, zmp = env.step(raw_target)

            if zmp is not None:
                print(f"ZMP: x={zmp[0]:.3f}, y={zmp[1]:.3f}")

            if done:
                obs = env.reset()
            viewer.sync()
