
import numpy as np
import cvxpy as cp
from scipy.linalg import pinv
import mujoco


class WBC_QP:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.nu = model.nu
        self.nq = model.nq
        self.nv = model.nv

        # 任务权重
        self.W_ori = 100.0
        self.W_joint = 10.0
        self.W_reg = 0.1

        self.tau_max = 20.0

        # 脚body id，根据你的模型修改！
        self.left_foot_id = self.model.body("left_foot").id
        self.right_foot_id = self.model.body("right_foot").id

    def compute_jacobian(self, body_id):
        J = np.zeros((6, self.nv))
        mujoco.mj_jacBody(self.model, self.data, J[:3], J[3:], body_id)
        return J

    def solve(self, desired_base_vel, desired_joint_vel):
        tau = cp.Variable(self.nu)

        # 1.基座姿态任务
        J_base = self.compute_jacobian(self.model.body("root").id)
        qdot = self.data.qvel
        task_ori = J_base @ qdot - desired_base_vel

        # 2.关节跟踪任务
        task_joint = qdot[6:] - desired_joint_vel

        # 3.足部接触约束：脚的加速度尽量为0（接触地面时）
        J_lf = self.compute_jacobian(self.left_foot_id)
        J_rf = self.compute_jacobian(self.right_foot_id)

        # 脚的加速度：J * qddot + dJ*qdot ≈ 0
        # 动力学：M qddot + C qdot + G = B tau
        # 简化QP形式：把脚加速度作为等式约束
        M = np.zeros((self.nv, self.nv))
        mujoco.mjM(self.model, self.data, M)
        C = np.zeros(self.nv)
        mujoco.mjC(self.model, self.data, C)
        G = np.zeros(self.nv)
        mujoco.mjG(self.model, self.data, G)
        B = np.zeros((self.nv, self.nu))
        mujoco.mjB(self.model, self.data, B)

        # 求qddot = M^{-1}(B tau - C qdot - G)
        Minv = pinv(M)
        qddot_expr = Minv @ (B @ tau - C - G)

        # 脚加速度约束：J_lf @ qddot_expr + J_lf_dot @ qdot = 0
        # 简化原型：接触时脚加速度为0
        foot_acc_constraint = []
        # 检测是否接触地面
        contact_left = False
        contact_right = False
        for c in self.data.contact:
            if c.geom1 == self.model.geom("left_foot_geom").id or c.geom2 == self.model.geom("left_foot_geom").id:
                contact_left = True
            if c.geom1 == self.model.geom("right_foot_geom").id or c.geom2 == self.model.geom("right_foot_geom").id:
                contact_right = True

        if contact_left:
            foot_acc_constraint.append(J_lf @ qddot_expr == 0)
        if contact_right:
            foot_acc_constraint.append(J_rf @ qddot_expr == 0)

        cost = (
            self.W_ori * cp.sum_squares(task_ori)
            + self.W_joint * cp.sum_squares(task_joint)
            + self.W_reg * cp.sum_squares(tau)
        )

        constraints = [
            tau >= -self.tau_max,
            tau <= self.tau_max,
        ] + foot_acc_constraint

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.ECOS, verbose=False)

        if prob.status != cp.OPTIMAL:
            return np.zeros(self.nu)
        return tau.value
