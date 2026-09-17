import numpy as np
import cvxpy as cp

class LIPM_MPC_2D:
    def __init__(self, com_height=0.92, dt=0.01, horizon=20):
        self.h = com_height
        self.g = 9.81
        self.omega = np.sqrt(self.g / self.h)
        self.dt = dt
        self.N = horizon

        # 单轴离散LIPM
        A = np.array([[1, self.dt],
                      [self.omega**2 * self.dt, 1]])
        B = np.array([[0],
                      [-self.omega**2 * self.dt]])
        self.A = A
        self.B = B

    def solve(self, x0, y0, zmp_ref_x_seq, zmp_ref_y_seq, support_poly):
        """
        x0: [com_x, dcom_x]
        y0: [com_y, dcom_y]
        zmp_ref_x_seq: N步X参考ZMP
        zmp_ref_y_seq: N步Y参考ZMP
        support_poly: (xmin,xmax,ymin,ymax)
        return: com_x_seq, dcom_x_seq, com_y_seq, dcom_y_seq, zmp_opt_x, zmp_opt_y
        """
        N = self.N
        # X方向变量
        x = cp.Variable((2, N+1))
        ux = cp.Variable((1, N))
        # Y方向变量
        y = cp.Variable((2, N+1))
        uy = cp.Variable((1, N))

        cost = 0
        constraints = []

        # X初始条件
        constraints.append(x[:,0] == x0)
        # Y初始条件
        constraints.append(y[:,0] == y0)

        for k in range(N):
            # X动力学
            constraints.append(x[:,k+1] == self.A @ x[:,k] + self.B @ ux[:,k])
            # Y动力学
            constraints.append(y[:,k+1] == self.A @ y[:,k] + self.B @ uy[:,k])

            # ZMP计算：zmp = com - (1/ω²)*ddcom
            zmp_x_k = x[0,k] - (1.0/(self.omega**2)) * (x[1,k+1]-x[1,k])/self.dt
            zmp_y_k = y[0,k] - (1.0/(self.omega**2)) * (y[1,k+1]-y[1,k])/self.dt

            # ZMP硬约束：必须在支撑多边形
            constraints.append(zmp_x_k >= support_poly[0])
            constraints.append(zmp_x_k <= support_poly[1])
            constraints.append(zmp_y_k >= support_poly[2])
            constraints.append(zmp_y_k <= support_poly[3])

            # 代价：跟踪参考ZMP + 控制输入惩罚
            cost += cp.square(zmp_x_k - zmp_ref_x_seq[k])
            cost += cp.square(zmp_y_k - zmp_ref_y_seq[k])
            cost += 0.1 * cp.square(ux[:,k])
            cost += 0.1 * cp.square(uy[:,k])

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        com_x_seq = x.value[0,:]
        dcom_x_seq = x.value[1,:]
        com_y_seq = y.value[0,:]
        dcom_y_seq = y.value[1,:]

        zmp_opt_x = []
        zmp_opt_y = []
        for k in range(N):
            zx = com_x_seq[k] - (1/(self.omega**2))*(dcom_x_seq[k+1]-dcom_x_seq[k])/self.dt
            zy = com_y_seq[k] - (1/(self.omega**2))*(dcom_y_seq[k+1]-dcom_y_seq[k])/self.dt
            zmp_opt_x.append(zx)
            zmp_opt_y.append(zy)

        return com_x_seq, dcom_x_seq, com_y_seq, dcom_y_seq, zmp_opt_x, zmp_opt_y
