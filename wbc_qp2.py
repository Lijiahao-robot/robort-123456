import numpy as np
import mujoco
from cvxpy import Variable, Minimize, Problem, quad_form

class WBC_QP:
    def __init__(self, model, data,
                 w_joint=1.0, w_zmp=0.5, w_vel=0.1):
        self.model = model
        self.data = data
        self.nq = model.nq
        self.nv = model.nv
        self.nu = model.nu

        # 权重：关节跟踪、ZMP跟踪、速度正则
        self.w_joint = w_joint
        self.w_zmp = w_zmp
        self.w_vel = w_vel

    def solve(self, desired_joint_vel, ref_zmp, real_zmp):
        """
        :param desired_joint_vel: 期望关节速度
        :param ref_zmp: 参考ZMP [x,y]
        :param real_zmp: 真实ZMP [x,y]，None则跳过ZMP代价
        :return: tau 关节力矩
        """
        nv = self.nv
        tau = Variable(nv)

        cost = 0
        # 1.关节速度跟踪代价
        J_joint = np.eye(nv)
        err_joint = desired_joint_vel - self.data.qvel
        cost += self.w_joint * quad_form(J_joint @ tau - err_joint, np.eye(nv))

        # 2.ZMP跟踪代价（新增）
        if real_zmp is not None:
            zmp_err = ref_zmp - real_zmp
            cost += self.w_zmp * quad_form(zmp_err, np.eye(2))

        # 3.力矩正则，防止力矩过大
        cost += self.w_vel * quad_form(tau, np.eye(nv))

        # 简单约束：力矩上下限
        tau_max = 100.0
        constraints = [
            tau <= tau_max,
            tau >= -tau_max
        ]

        prob = Problem(Minimize(cost), constraints)
        prob.solve(solver=cvxpy.OSQP, verbose=False)

        if prob.status != "optimal":
            tau_sol = np.zeros(nv)
        else:
            tau_sol = tau.value

        return tau_sol
