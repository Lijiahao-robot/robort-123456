# 在HumanoidG1Env.__init__中替换
self.dcm_preview = DCMKajitaPreview(com_height=0.92, preview_N=15, dt=self.model.opt.timestep)

# step函数内部：
def step(self, raw_action):
    smooth_act = self.lpf.step(raw_action)
    desired_joint_acc = (smooth_act - self.data.qpos[7:]) / (self.dt**2)

    # 足部摆动轨迹
    t = self.t_sim
    swing_foot_start = np.array([0.0, 0.08, 0.0])
    swing_foot_end = np.array([0.28, 0.08, 0.0])
    foot_pos, foot_vel = self.foot_traj.generate_foot_trajectory(
        t, t_swing_start=0.0, t_swing_end=0.7,
        foot_start=swing_foot_start, foot_end=swing_foot_end
    )

    # Kajita DCM‑Preview QP求解
    com = self.data.subtree_com[self.model.body("root").id][:3]
    dcom = self.data.subtree_linvel[self.model.body("root").id][:3]
    xi = self.dcm_preview.compute_dcm_state(com[0], dcom[0])
    zmp_ref_seq = np.ones(self.dcm_preview.N)*0.14
    u_zmp = self.dcm_preview.get_current_zmp(xi, zmp_ref_seq)
    ref_zmp = np.array([u_zmp, 0.0])

    # 完整WBC‑QP硬ZMP等式约束
    tau = self.wbc.solve(desired_joint_acc, ref_zmp)
    self.data.ctrl[:] = tau
    mujoco.mj_step(self.model, self.data)
    # ...后续不变
