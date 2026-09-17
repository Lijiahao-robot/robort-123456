import torch
import numpy as np
import mujoco
import mujoco.viewer
from motion.foot_trajectory import FootTrajectoryGenerator
from motion.dcm_preview import DCMPreviewController
from motion.ik_g1 import HumanoidIK_G1
from controllers.wbc_qp_full import WBC_QP_Full
from controllers.zmp import ZMPCalculator
from rl.bc_rl import ActorCritic
from utils.logger_plot import DataLogger

class LowPassFilter:
    def __init__(self, alpha=0.15, dim=None):
        self.alpha = alpha
        self.prev = None
    def reset(self):
        self.prev = None
    def step(self, x):
        if self.prev is None:
            self.prev = x.copy()
        self.prev = self.alpha*x + (1-self.alpha)*self.prev
        return self.prev

class HumanoidG1Env:
    def __init__(self, model_path="./models/g1.xml"):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.obs_dim = self.model.nq + self.model.nv
        self.act_dim = self.model.nu

        self.lpf = LowPassFilter(alpha=0.15, dim=self.act_dim)
        self.zmp_calc = ZMPCalculator(self.model, self.data)
        self.foot_traj = FootTrajectoryGenerator(step_height=0.09, dt=self.model.opt.timestep)
        self.dcm_preview = DCMPreviewController(com_height=0.92, preview_horizon=12, dt=self.model.opt.timestep)
        self.ik = HumanoidIK_G1()
        self.wbc = WBC_QP_Full(self.model, self.data, com_height=0.92)
        self.logger = DataLogger(save_path="g1_log.csv")
        self.dt = self.model.opt.timestep
        self.t_sim = 0.0

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.lpf.reset()
        self.t_sim = 0.0
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        return obs

    def step(self, raw_action):
        smooth_act = self.lpf.step(raw_action)
        desired_joint_acc = (smooth_act - self.data.qpos[7:]) / (self.dt**2)

        # 足部摆动轨迹生成：摆动脚目标位置
        t = self.t_sim
        swing_foot_start = np.array([0.0, 0.08, 0.0])
        swing_foot_end = np.array([0.28, 0.08, 0.0])
        foot_pos, foot_vel = self.foot_traj.generate_foot_trajectory(
            t, t_swing_start=0.0, t_swing_end=0.7,
            foot_start=swing_foot_start, foot_end=swing_foot_end
        )

        # DCM‑Preview 得到参考ZMP
        dcm_current = 0.0
        zmp_ref_seq = np.array([0.0]*12)
        u_zmp = self.dcm_preview.preview_control(dcm_current, zmp_ref_seq)
        ref_zmp = np.array([u_zmp, 0.0])

        # WBC‑QP完整动力学，硬ZMP等式约束求解力矩
        tau = self.wbc.solve(desired_joint_acc, ref_zmp)
        self.data.ctrl[:] = tau
        mujoco.mj_step(self.model, self.data)

        obs = np.concatenate([self.data.qpos, self.data.qvel])
        real_zmp = self.zmp_calc.compute_zmp()
        real_com = self.data.subtree_com[self.model.body("root").id][:3]

        reward = self.data.qvel[0] - 0.005 * np.sum(np.square(tau))
        if real_zmp is not None:
            zmp_err = np.linalg.norm(real_zmp - ref_zmp)
            reward -= 0.04 * zmp_err

        joint_dict = {
            "left_hip_pitch": self.data.qpos[7+1],
            "left_knee": self.data.qpos[7+2],
            "right_hip_pitch": self.data.qpos[7+5],
            "right_knee": self.data.qpos[7+6],
        }
        self.logger.log(self.t_sim, ref_zmp, real_zmp, real_com[:2], real_com[:2], joint_dict)
        self.t_sim += self.dt
        done = self.data.qpos[2] < 0.75
        return obs, reward, done, real_zmp, ref_zmp, real_com

if __name__ == "__main__":
    env = HumanoidG1Env("./models/g1.xml")
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    agent = ActorCritic(obs_dim, act_dim)
    agent.load_state_dict(torch.load("bc_rl_final.pt", map_location="cpu"))
    agent.eval()

    obs = env.reset()
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            action = agent.get_action(obs)
            obs_next, reward, done, real_zmp, ref_zmp, real_com = env.step(action)
            if real_zmp is not None:
                print(f"RefZMP:{ref_zmp[0]:.3f},{ref_zmp[1]:.3f} RealZMP:{real_zmp[0]:.3f},{real_zmp[1]:.3f}")
            if done:
                obs = env.reset()
            obs = obs_next
            viewer.sync()
    env.logger.save_csv()
    env.logger.plot()
