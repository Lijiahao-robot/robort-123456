import numpy as np
import cvxpy as cp
from scipy.linalg import pinv

class WBC_QP:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.nu = model.nu
        self.nq = model.nq
        self.nv = model.nv

        # 权重：姿态任务 > 关节位置任务 > 最小化控制输入
        self.W_ori = 100.0
        self.W_joint = 10.0
        self.W_reg = 0.1

        # 关节力矩上下限（人形机器人典型限制）
        self.tau_max = 20.0

    def compute_jacobian(self, body_id):
        """计算刚体雅可比矩阵"""
        J = np.zeros((6, self.nv))
        mujoco.mj_jacBody(self.model, self.data, J[:3], J[3:], body_id)
        return J

    def solve(self, desired_base_vel, desired_joint_vel):
        """
        desired_base_vel: 基座6维期望速度 [dx,dy,dz,wx,wy,wz]
        desired_joint_vel: 关节期望速度 (nu,)
        return: 关节力矩 tau
        """
        # 优化变量：关节力矩 tau (nu,)
        tau = cp.Variable(self.nu)

        # 1. 基座姿态任务：J_base * qdot ≈ desired_base_vel
        J_base = self.compute_jacobian(self.model.body("root").id)
        qdot = self.data.qvel
        task_ori = J_base @ qdot - desired_base_vel

        # 2. 关节位置任务：关节速度跟踪
        task_joint = qdot[6:] - desired_joint_vel

        # 目标函数：最小化加权任务误差 + 控制正则
        cost = (
            self.W_ori * cp.sum_squares(task_ori)
            + self.W_joint * cp.sum_squares(task_joint)
            + self.W_reg * cp.sum_squares(tau)
        )

        # 约束：力矩限幅
        constraints = [
            tau >= -self.tau_max,
            tau <= self.tau_max
        ]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.ECOS, verbose=False)

        if prob.status != cp.OPTIMAL:
            return np.zeros(self.nu)
        return tau.value
