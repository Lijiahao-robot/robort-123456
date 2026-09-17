import numpy as np

class KalmanFilter:
    def __init__(self, x0, P0, A, B, Q, R):
        self.x = x0
        self.P = P0
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

    def predict(self, u):
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        S = self.A @ self.P @ self.A.T + self.R
        K = self.P @ self.A.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.A @ self.x)
        self.P = (np.eye(self.x.shape[0]) - K @ self.A) @ self.P


class EKF:
    """EKF 非线性系统，用于机器人定位"""
    def __init__(self, x0, P0, Q, R):
        self.x = x0
        self.P = P0
        self.Q = Q
        self.R = R

    def predict(self, f, F_jacob, u):
        """f:状态转移函数；F_jacob:雅可比矩阵"""
        self.x = f(self.x, u)
        self.P = F_jacob @ self.P @ F_jacob.T + self.Q

    def update(self, z, h, H_jacob):
        """h:观测函数；H_jacob观测雅可比"""
        S = H_jacob @ self.P @ H_jacob.T + self.R
        K = self.P @ H_jacob.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - h(self.x))
        self.P = (np.eye(self.x.shape[0]) - K @ H_jacob) @ self.P


# 示例：一维位置速度卡尔曼
if __name__ == "__main__":
    dt = 0.1
    A = np.array([[1, dt],[0,1]])
    B = np.array([[0.5*dt**2],[dt]])
    Q = np.diag([0.1,0.1])
    R = np.array([[1.0]])
    kf = KalmanFilter(x0=np.array([[0],[0]]), P0=np.eye(2), A=A, B=B, Q=Q, R=R)
    # 模拟噪声观测
    for _ in range(50):
        kf.predict(u=np.array([[0.1]]))
        z_obs = np.array([[np.random.normal(0,1)]])
        kf.update(z_obs)
        print(f"est pos:{kf.x[0,0]:.2f}")
