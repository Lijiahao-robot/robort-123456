import numpy as np
import cvxpy as cp

class LIPM_MPC:
    def __init__(self, com_height=0.92, dt=0.01, horizon=20):
        self.h = com_height
        self.g = 9.81
        self.omega = np.sqrt(self.g / self.h)
        self.dt = dt
        self.N = horizon

        # LIPM离散状态方程 x = [com_x; dcom_x]
        A = np.array([[1, self.dt],
                      [self.omega**2 * self.dt, 1]])
        B = np.array([[0],
                      [-self.omega**2 * self.dt]])
        self.A = A
        self.B = B

    def solve(self, x0, zmp_ref_seq, support_polygon):
        """
        x0: [com_x, dcom_x] 当前质心状态
        zmp_ref_seq: 参考ZMP序列 N步
        support_polygon: (xmin,xmax,ymin,ymax)
        return: com_x_seq, dcom_x_seq, zmp_opt_seq
        """
        N = self.N
        x = cp.Variable((2, N+1))
        u = cp.Variable((1, N))

        cost = 0
        constraints = []
        # 初始条件
        constraints.append(x[:,0] == x0)

        for k in range(N):
            # 状态转移
            constraints.append(x[:,k+1] == self.A @ x[:,k] + self.B @ u[:,k])
            # ZMP约束：zmp = com_x - (1/omega^2)*dcom_x_dot
            zmp_k = x[0,k] - (1.0 / (self.omega**2)) * (x[1,k+1] - x[1,k])/self.dt
            constraints.append(zmp_k >= support_polygon[0])
            constraints.append(zmp_k <= support_polygon[1])
            # 代价：跟踪参考ZMP，控制输入惩罚
            cost += cp.square(zmp_k - zmp_ref_seq[k])
            cost += 0.1 * cp.square(u[:,k])

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP)

        com_x_seq = x.value[0,:]
        dcom_x_seq = x.value[1,:]
        zmp_opt_seq = []
        for k in range(N):
            zmp_k = com_x_seq[k] - (1/(self.omega**2))*(dcom_x_seq[k+1]-dcom_x_seq[k])/self.dt
            zmp_opt_seq.append(zmp_k)
        return com_x_seq, dcom_x_seq, zmp_opt_seq
