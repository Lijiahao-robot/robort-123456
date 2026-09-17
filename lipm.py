import numpy as np

class LIPM:
    def __init__(self, com_height=0.8, g=9.81):
        """
        :param com_height: 质心高度 m
        :param g: 重力加速度
        """
        self.z = com_height
        self.g = g
        self.omega = np.sqrt(self.g / self.z)  # LIPM特征频率

    def solve_cm_from_zmp(self, zmp_ref, x0, dx0, t_span, dt):
        """
        LIPM解析解：已知ZMP参考，求解质心位置 x(t), dx(t)
        方程：
        \ddot x = \omega^2 (x - zmp)
        :param zmp_ref: 参考ZMP序列，shape(N,)
        :param x0: 质心初始位置
        :param dx0: 质心初始速度
        :param t_span: 时间序列
        :param dt: 时间步
        :return: cm_x, dcm_x 质心位置、速度序列
        """
        N = len(t_span)
        cm_x = np.zeros(N)
        dcm_x = np.zeros(N)
        cm_x[0] = x0
        dcm_x[0] = dx0

        for i in range(N-1):
            zmp = zmp_ref[i]
            # 欧拉积分
            ddx = self.omega**2 * (cm_x[i] - zmp)
            dcm_x[i+1] = dcm_x[i] + ddx * dt
            cm_x[i+1] = cm_x[i] + dcm_x[i] * dt
        return cm_x, dcm_x

    def forward_integrate(self, x0, dx0, zmp, t_total, dt):
        """
        单段LIPM向前积分，给定固定ZMP，输出质心轨迹
        """
        t = np.arange(0, t_total, dt)
        zmp_arr = np.full_like(t, zmp)
        cm_x, dcm_x = self.solve_cm_from_zmp(zmp_arr, x0, dx0, t, dt)
        return t, cm_x, dcm_x
