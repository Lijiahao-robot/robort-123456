import numpy as np
import cvxpy as cp

class DCMKajitaPreview:
    def __init__(self, com_height=0.92, g=9.81, preview_N=15, dt=0.01):
        self.z = com_height
        self.g = g
        self.omega = np.sqrt(self.g / self.z)
        self.N = preview_N
        self.dt = dt

        # 离散DCM状态空间
        A = np.array([
            [1 + self.omega*self.dt, self.dt],
            [self.omega**2 * self.dt, 1]
        ])
        B = np.array([
            [-self.omega*self.dt],
            [-self.omega**2 * self.dt]
        ])
        self.A = A
        self.B = B

        # 权重：DCM跟踪，ZMP平滑
        self.Q_xi = 100.0
        self.R_u = 1.0

    def solve_preview(self, xi_current, zmp_ref_seq):
        """
        :param xi_current: 当前DCM状态 [DCM, dDCM] (2,)
        :param zmp_ref_seq: 未来N步ZMP参考 (N,)
        :return: u_opt:最优ZMP序列 (N,)
        """
        N = self.N
        # 决策变量：未来N步ZMP u[0..N‑1]
        u = cp.Variable(N)
        xi = cp.Variable((2, N+1))

        cost = 0
        # 初始条件
        xi[:,0] == xi_current

        # 动力学约束
        for k in range(N):
            xi[:,k+1] == self.A @ xi[:,k] + self.B @ u[k]

        # 代价：DCM跟踪误差 + ZMP变化平滑
        for k in range(N):
            cost += self.Q_xi * cp.square(xi[0,k+1] - zmp_ref_seq[k])
            if k > 0:
                cost += self.R_u * cp.square(u[k] - u[k-1])

        # ZMP约束：必须落在支撑脚支撑多边形内（简化：[-0.15,0.15]）
        constraints = [
            u >= -0.15,
            u <= 0.15
        ]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        if prob.status != cp.OPTIMAL:
            u_opt = zmp_ref_seq.copy()
        else:
            u_opt = u.value
        return u_opt

    def get_current_zmp(self, xi_current, zmp_ref_seq):
        u_opt = self.solve_preview(xi_current, zmp_ref_seq)
        return u_opt[0]

    def compute_dcm_state(self, com, dcom):
        """由CoM位置速度计算DCM状态 xi=[DCM, dDCM]"""
        dcm = com + dcom / self.omega
        ddcm = self.omega * (dcm - com)
        return np.array([dcm, ddcm])
