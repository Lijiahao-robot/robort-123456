import torch
import numpy as np
import mujoco
import mujoco.viewer
import pickle
from controllers.wbc_qp import WBC_QP
from controllers.zmp import ZMPCalculator
from motion.dcm_lipm_gait import DCMLIPMGaitPlanner
from motion.ik import HumanoidIK
from rl.bc_rl import ActorCritic
from utils.logger_plot import DataLogger

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
        self.prev_val = self.alpha * x + (1-self.alpha)*self.prev_val
        return self.prev_val.copy()

class HumanoidDCMEnv:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.obs_dim = self.model.nq + self.model.nv
        self.act_dim = self.model.nu
        self.wbc = WBC_QP(self.model, self.data)
        self.lpf = LowPassFilter(alpha=0.15, dim=self.act_dim)
        self.zmp_calc = ZMPCalculator(self.model, self.data, ground_z=0.0)
        self.dcm_planner = DCMLIPMGaitPlanner(com_height=0.8, dt=self.model.opt.timestep)
        self.ik = HumanoidIK()
        self.dt = self.model.opt.timestep
        self.logger = DataLogger()
        self.t_sim = 0.0

        self.t_plan, self.ref_zmp_x, self.ref_zmp_y, self.ref_cm_x, self.ref_cm_y, _, _ = self.dcm_planner.plan_one_step()
        self.plan_idx = 0

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.lpf.reset()
        self.dcm_planner.reset()
        self.t_sim = 0.0
        self.t_plan, self.ref_zmp_x, self.ref_zmp_y, self.ref_cm_x, self.ref_cm_y, _, _ = self.dcm_planner.plan_one_step()
        self.plan_idx = 0
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        return obs

    def step(self, raw_target_joint_pos):
        idx = self.plan_idx
        ref_zmp = np.array([self.ref_zmp_x[idx], self.ref_zmp_y[idx]])
        ref_cm = np.array([self.ref_cm_x[idx], self.ref_cm_y[idx], 0.8])
        self.plan_idx += 1
        if self.plan_idx >= len(self.t_plan):
            self.plan_idx = 0
            self.t_plan, self.ref_zmp_x, self.ref_zmp_y, self.ref_cm_x, self.ref_cm_y, _, _ = self.dcm_planner.plan_one_step()

        smooth_target = self.lpf.step(raw_target_joint_pos)
        desired_joint_vel = (smooth_target - self.data.qpos[7:]) / self.dt
        desired_base_vel = np.zeros(6)
        tau = self.wbc.solve(desired_base_vel, desired_joint_vel)
        self.data.ctrl[:] = tau
        mujoco.mj_step(self.model, self.data)

        obs = np.concatenate([self.data.qpos, self.data.qvel])
        forward_vel = self.data.qvel[0]
        real_zmp = self.zmp_calc.compute_zmp()
        real_com = self.data.subtree_com[self.model.body("root").id][:3]

        reward = forward_vel - 0.005 * np.sum(np.square(tau))
        if real_zmp is not None:
            zmp_err = np.linalg.norm(real_zmp - ref_zmp)
            reward -= 0.03 * zmp_err
        com_err = np.linalg.norm(real_com[:2] - ref_cm[:2])
        reward -= 0.02 * com_err

        # 日志记录
        joint_dict = {"left_hip_pitch": self.data.qpos[7+0], "left_knee": self.data.qpos[7+1],
                      "right_hip_pitch": self.data.qpos[7+2], "right_knee": self.data.qpos[7+3]}
        self.logger.log(self.t_sim, ref_zmp, real_zmp, ref_cm, real_com, joint_dict)
        self.t_sim += self.dt

        done = self.data.qpos[2] < 0.6
        return obs, reward, done, real_zmp, ref_zmp, real_com, ref_cm

if __name__ == "__main__":
    model_path = "./models/humanoid.xml"
    env = HumanoidDCMEnv(model_path)
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
            obs, reward, done, real_zmp, ref_zmp, real_com, ref_cm = env.step(raw_target)
            if real_zmp is not None:
                print(f"RefZMP:{ref_zmp[0]:.3f},{ref_zmp[1]:.3f} | RealZMP:{real_zmp[0]:.3f},{real_zmp[1]:.3f}")
            if done:
                obs = env.reset()
            viewer.sync()
    env.logger.save_csv()
    env.logger.plot()
