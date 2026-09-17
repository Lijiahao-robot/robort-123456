import numpy as np
from scipy.optimize import minimize

def forward_kinematics(theta, L):
    """2连杆机械臂正运动学"""
    x = L[0]*np.cos(theta[0]) + L[1]*np.cos(theta[0]+theta[1])
    y = L[0]*np.sin(theta[0]) + L[1]*np.sin(theta[0]+theta[1])
    return np.array([x,y])


def numerical_ik(target_pos, L, init_theta):
    """数值逆运动学，最小化末端距离误差"""
    def cost(theta):
        end = forward_kinematics(theta, L)
        return np.sum((end-target_pos)**2)
    res = minimize(cost, init_theta, method="L‑BFGS‑B")
    return res.x


if __name__ == "__main__":
    L = [1.0,1.0]
    target = np.array([1.2, 0.8])
    theta_guess = [0.1, 0.1]
    sol_theta = numerical_ik(target, L, theta_guess)
    print("IK解角度：", sol_theta)
    print("末端位置：", forward_kinematics(sol_theta, L))
