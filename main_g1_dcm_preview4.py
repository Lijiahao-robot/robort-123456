
import torch
import numpy as np
import mujoco
import mujoco.viewer

from motion.foot_trajectory import FootTrajectoryGenerator
from motion.dcm_kajita_preview import DCMKajitaPreview
from motion.ik_g1 import HumanoidIK_G1
from controllers.wbc_qp_full import WBC_QP_Full
from controllers.zmp import ZMPCalculator
from rl.bc_rl import ActorCritic
from utils.logger_plot import DataLogger
from motion.support_state_machine import SupportStateMachine
from motion.arm_swing_trajectory import ArmSwingGenerator
from utils.disturbance_injector import DisturbanceInjector
from motion.gait_planner import GaitPlanner
from utils.zmp_stability_metric import ZMPStabilityMetric
from motion.gait_recovery import GaitRecovery

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
        self.dcm_preview = DCMKajitaPreview(com_height=0.92, preview_N=15, dt=self.model.opt.timestep)
        self.ik = HumanoidIK_G1()
        self.wbc = WBC_QP_Full(self.model, self.data, com_height=0.92)

        # 新增模块
        self.support_sm = SupportStateMachine(force_threshold=20.0)
        self.arm_swing = ArmSwingGenerator(swing_amp=0.35, freq=1.2, dt=self.model.opt.timestep)
        self.disturb = DisturbanceInjector()
        self.gait_planner = GaitPlanner(step_length=0.28, step_freq=1.2, step_height=0.09, lateral_offset=0.08)
        self.zmp_metric = ZMPStabilityMetric()
        self.gait_recovery = GaitRecovery(margin_threshold=-0.02, tilt_threshold=0.3)

        self.logger = DataLogger(save_path="g1_log.csv")
        self.dt = self.model.opt.timestep
        self.t_sim = 0.0

    def _get_foot_contact_force(self):
        left_force = 0.0
        right_force = 0.0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            geom1 = self.model.geom(con.geom1).name
            geom2 = self.model.geom(con.geom2).name
            if "left_foot" in geom1 or "left_foot" in geom2:
                left_force += self.data.contact[i].force[2]
            if "right_foot" in geom1 or "right_foot" in geom2:
                right_force += self.data.contact[i].force[2]
        return max(left_force, 0), max(right_force, 0)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.lpf.reset()
        self.disturb.reset()
        self.gait_planner.reset()
        self.gait_recovery = GaitRecovery(margin_threshold=-0.02, tilt_threshold=0.3)
        self.t_sim = 0.0
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        return obs

    def step(self, raw_action):
        smooth_act = self.lpf.step(raw_action)
        desired_joint_acc = (smooth_act - self.data.qpos[7:]) / (self.dt**2)

        # ==========1.支撑相状态机更新 ==========
        l_force, r_force = self._get_foot_contact_force()
        self.support_sm.update(l_force, r_force)
        support_state = self.support_sm.state
        zmp_x_min, zmp_x_max, zmp_y_min, zmp_y_max = self.support_sm.get_support_polygon()
        support_polygon = (zmp_x_min, zmp_x_max, zmp_y_min, zmp_y_max)

        # ==========2.步态规划器，获取下一步脚目标 ==========
        t_swing_start, t_swing_end, gait_phase = self.gait_planner.get_gait_timing(self.t_sim)
        swing_foot, foot_target = self.gait_planner.generate_next_step()
        l_foot_cur, r_foot_cur = self.gait_planner.get_foot_positions()

        if swing_foot == "right":
            foot_pos, foot_vel = self.foot_traj.generate_foot_trajectory(
                self.t_sim, t_swing_start, t_swing_end, r_foot_cur, foot_target
            )
        else:
            foot_pos, foot_vel = self.foot_traj.generate_foot_trajectory(
                self.t_sim, t_swing_start, t_swing_end, l_foot_cur, foot_target
            )

        # ==========3.手臂协同摆动 ==========
        l_shoulder_pitch, r_shoulder_pitch = self.arm_swing.generate_arm_joint(self.t_sim, gait_phase, support_state)

        # ==========4. Kajita‑DCM‑Preview ==========
        com = self.data.subtree_com[self.model.body("root").id][:3]
        dcom = self.data.subtree_linvel[self.model.body("root").id][:3]
        xi = self.dcm_preview.compute_dcm_state(com[0], dcom[0])
        zmp_ref_seq = np.ones(self.dcm_preview.N)*0.14
        self.dcm_preview.zmp_bound = (zmp_x_min, zmp_x_max)
        u_zmp = self.dcm_preview.get_current_zmp(xi, zmp_ref_seq)
        ref_zmp = np.array([u_zmp, 0.0])

        # ==========5. ZMP稳定裕度计算 ==========
        real_zmp = self.zmp_calc.compute_zmp()
        zmp_margin = 0.0
        stability_score = 0.0
        if real_zmp is not None:
            zmp_margin, is_stable = self.zmp_metric.compute_margin(real_zmp, support_polygon)
            stability_score = self.zmp_metric.compute_stability_score(zmp_margin)

        # ==========6.步态恢复策略：检测失稳触发恢复 ==========
        torso_tilt = self.data.qpos[3] # root俯仰角
        need_recover = self.gait_recovery.check_need_recovery(zmp_margin, torso_tilt)
        if need_recover and not self.gait_recovery.recovery_flag:
            self.gait_recovery.reset_gait_for_recovery(self.gait_planner, self.dcm_preview, com)
        if self.gait_recovery.recovery_flag:
            self.gait_recovery.update_recovery_timer(self.dt)

        # ==========7.WBC‑QP求解力矩 ==========
        tau = self.wbc.solve(desired_joint_acc, ref_zmp)

        # ==========8.施加扰动 ==========
        disturb_force = self.disturb.get_disturb_force(self.t_sim)
        if np.linalg.norm(disturb_force) > 1e-6:
            mujoco.mj_applyForce(self.model, self.data, self.model.body("root").id, disturb_force, np.zeros(3))

        self.data.ctrl[:] = tau
        mujoco.mj_step(self.model, self.data)

        obs = np.concatenate([self.data.qpos, self.data.qvel])
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
            "left_shoulder": l_shoulder_pitch,
            "right_shoulder": r_shoulder_pitch
        }
        self.logger.log(self.t_sim, ref_zmp, real_zmp, real_com[:2], real_com[:2], joint_dict)
        self.logger.log_metric("zmp_margin", zmp_margin)
        self.logger.log_metric("stability_score", stability_score)
        self.logger.log_metric("recovery_flag", float(self.gait_recovery.recovery_flag))

        self.t_sim += self.dt
        done = self.data.qpos[2] < 0.75
        return obs, reward, done, real_zmp, ref_zmp, real_com

    def run_disturbance_test(self, test_type="impulse", force=np.array([80,0,0]), test_duration=8.0):
        obs = self.reset()
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            t0 = self.t_sim
            if test_type == "impulse":
                self.disturb.apply_impulse_disturbance(t0+1.5, impulse_force=force, duration=0.2)
            elif test_type == "constant":
                self.disturb.apply_constant_disturbance(t0+1.5, const_force=force, duration=test_duration)

            while viewer.is_running() and self.t_sim < t0 + test_duration:
                action = self.agent.get_action(obs)
                obs_next, reward, done, real_zmp, ref_zmp, real_com = self.step(action)
                if real_zmp is not None:
                    print(f"t={self.t_sim:.2f} | ZMP_margin:{zmp_margin:.3f} | stable_score:{stability_score:.2f} | state:{self.support_sm.state} | recover:{self.gait_recovery.recovery_flag}")
                if done:
                    print("【测试失败】机器人摔倒")
                    break
                obs = obs_next
                viewer.sync()
        self.logger.save_csv()
        self.logger.plot()

if __name__ == "__main__":
    env = HumanoidG1Env("./models/g1.xml")
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    agent = ActorCritic(obs_dim, act_dim)
    agent.load_state_dict(torch.load("g1_bc_rl_final.pt", map_location="cpu"))
    agent.eval()
    env.agent = agent

    # 扰动测试示例：X方向脉冲扰动
    env.run_disturbance_test(test_type="impulse", force=np.array([80,0,0]), test_duration=8.0)
