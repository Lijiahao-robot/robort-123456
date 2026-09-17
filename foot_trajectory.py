import numpy as np

class FootTrajectoryGenerator:
    def __init__(self, step_height=0.08, dt=0.01):
        self.step_height = step_height
        self.dt = dt

    def quintic_poly(self, t, t_start, t_end, p0, v0, a0, p1, v1, a1):
        """五次多项式插值，位置/速度/加速度边界条件"""
        tau = (t - t_start) / (t_end - t_start)
        if tau < 0:
            return p0
        if tau > 1:
            return p1
        a = 10*tau**3 - 15*tau**4 + 6*tau**5
        b = t_end-t_start * (tau**3 - 3*tau**4 + 2*tau**5)
        c = (t_end-t_start)**2 * (0.5*tau**3 - 1.5*tau**4 + tau**5)
        d = -10*tau**3 + 15*tau**4 -6*tau**5
        e = t_end-t_start * (-2*tau**3 + 3*tau**4 - tau**5)
        f = (t_end-t_start)**2 * (-0.5*tau**3 + tau**4 -0.5*tau**5)
        pos = p0 + v0*b + a0*c + p1*a + v1*e + a1*f
        return pos

    def generate_foot_trajectory(self, t, t_swing_start, t_swing_end,
                                 foot_start, foot_end):
        """
        :param t: 当前时间
        :param t_swing_start: 摆动开始时刻
        :param t_swing_end: 摆动结束时刻
        :param foot_start: 起始脚位置 [x,y,z]
        :param foot_end: 目标脚位置 [x,y,z]
        :return: foot_pos, foot_vel
        """
        if t < t_swing_start:
            return foot_start, np.zeros(3)
        if t >= t_swing_end:
            return foot_end, np.zeros(3)

        # 中间最高点
        mid_z = max(foot_start[2], foot_end[2]) + self.step_height
        # X/Y 从start到end；Z先抬到mid_z再落回
        px = self.quintic_poly(t, t_swing_start, t_swing_end,
                               foot_start[0],0,0, foot_end[0],0,0)
        py = self.quintic_poly(t, t_swing_start, t_swing_end,
                               foot_start[1],0,0, foot_end[1],0,0)
        pz = self.quintic_poly(t, t_swing_start, t_swing_end,
                               foot_start[2],0,0, mid_z,0,0)
        pz = self.quintic_poly(t, (t_swing_start+t_swing_end)/2, t_swing_end,
                               pz,0,0, foot_end[2],0,0)
        foot_pos = np.array([px, py, pz])
        foot_vel = np.zeros(3)
        return foot_pos, foot_vel
