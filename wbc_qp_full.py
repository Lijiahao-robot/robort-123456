import numpy as np
import mujoco
import cvxpy as cp

class WBC_QP_Full:
    def __init__(self, model, data, com_height=0.8, g=9.81):
        self.model = model
        self.data = data
        self.nq = model.nq
        self.nv = model.nv
        self.nu = model.nu
        self.z = com_height
        self.g = g

        # 任务权重
        self.w_joint = 1.0
        self.w_vel_reg = 0.01

    def compute_zmp_jacobian(self):
        """
        ZMP雅可比：J_zmp，映射关节加速度 -> ZMP误差
        zmp = com - (z/g)*com_ddot
        com_ddot = J_com @ qddot + dJ_com @ qdot
        """
        J_com = np.zeros((3, self.nv))
        mujoco.mj_jacSubtreeCom(self.model, self.data, J_com, self.model.body("root").id)
        dJ_com = np.zeros((3, self.nv))
        mujoco.mj_jacSubtreeComDot(self.model, self.data, dJ_com, self.model.body("root").id)

        com_ddot_bias = dJ_com @ self.data.qvel
        # ZMP动力学：zmp = com - z/g * com_ddot
        J_zmp = - (self.z / self.g) * J_com[:2, :]
        zmp_bias = self.data.subtree_com[self.model.body("root").id][:2] - (self.z / self.g)*com_ddot_bias[:2]
        return J_zmp, zmp_bias

    def solve(self, desired_joint_acc, ref_zmp):
        """
        WBC‑QP，硬等式约束：J_zmp @ qddot = ref_zmp - zmp_bias
        目标：最小化关节加速度跟踪误差 + 正则；等式约束：ZMP跟踪
        :param desired_joint_acc: 期望关节加速度
        :param ref_zmp: 参考ZMP [x,y]
        :return: tau 关节力矩
        """
        nv = self.nv
        qddot = cp.Variable(nv)
        tau = cp.Variable(nv)

        J_zmp, zmp_bias = self.compute_zmp_jacobian()

        # 代价：关节加速度跟踪 + 力矩正则
        cost = self.w_joint * cp.sum_squares(qddot - desired_joint_acc)
        cost += self.w_vel_reg * cp.sum_squares(tau)

        # 动力学等式：M qddot + C qdot + G = tau
        M = np.zeros((nv, nv))
        C = np.zeros(nv)
        G = np.zeros(nv)
        mujoco.mj_massMatrix(self.model, self.data, M)
        mujoco.mj_bias(self.model, self.data, C)
        dyn_constraint = M @ qddot + C + G == tau

        # ZMP硬等式约束：J_zmp @ qddot = ref_zmp - zmp_bias
        zmp_constraint = J_zmp @ qddot == (ref_zmp - zmp_bias)

        # 边界约束
        tau_max = 120.0
        constraints = [
            dyn_constraint,
            zmp_constraint,
            tau <= tau_max,
            tau >= -tau_max
        ]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        if prob.status != cp.OPTIMAL:
            tau_sol = np.zeros(nv)
        else:
            tau_sol = tau.value
        return tau_sol
