import numpy as np

class DCM:
    def __init__(self, com_height=0.8, g=9.81):
        self.z = com_height
        self.g = g
        self.omega = np.sqrt(self.g / self.z)

    def compute_dcm(self, com_x, dcom_x):
        """
        DCM = x + dx/omega
        :param com_x:质心x位置
        :param dcom_x:质心速度
        :return: dcm_x
        """
        return com_x + dcom_x / self.omega

    def dcm_integrate(self, zmp, dcm0, dt):
        """
        DCM动力学： d(DCM)/dt = omega*(DCM‑zmp)
        给定ZMP，积分一步得到下一时刻DCM
        """
        d_dcm = self.omega * (dcm0 - zmp)
        dcm_next = dcm0 + d_dcm * dt
        return dcm_next

    def get_com_from_dcm(self, dcm):
        """由DCM反推质心位置、速度"""
        return dcm, self.omega * (dcm - dcm)

    def step_gait_switch(self, zmp_support, dcm_prev, dt):
        """步态切换：单脚支撑切换，DCM状态更新，用于步态规划衔接"""
        dcm_new = self.dcm_integrate(zmp_support, dcm_prev, dt)
        com_new = dcm_new
        dcom_new = self.omega*(dcm_new - zmp_support)
        return dcm_new, com_new, dcom_new
