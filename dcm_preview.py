import numpy as np

class DCMPreviewController:
    def __init__(self, com_height=0.8, g=9.81, preview_horizon=10, dt=0.01):
        self.z = com_height
        self.g = g
        self.omega = np.sqrt(self.g / self.z)
        self.preview_N = preview_horizon
        self.dt = dt

    def dcm_dynamics(self, dcm_prev, zmp):
        """DCM离散动力学 d(DCM)/dt = omega*(DCM‑zmp)"""
        dcm_next = dcm_prev + self.omega * (dcm_prev - zmp) * self.dt
        return dcm_next

    def preview_control(self, dcm_current, zmp_ref_seq):
        """
        N‑step preview：输入当前DCM，未来N步ZMP参考，输出控制ZMP
        :param dcm_current: 当前DCM
        :param zmp_ref_seq: 未来N步ZMP参考 [N]
        :return: u_zmp: 当前控制ZMP
        """
        N = self.preview_N
        dcm_pred = np.zeros(N+1)
        dcm_pred[0] = dcm_current
        for i in range(N):
            dcm_pred[i+1] = self.dcm_dynamics(dcm_pred[i], zmp_ref_seq[i])
        # 简单预览反馈：让预测终点DCM逼近目标ZMP
        target_dcm = zmp_ref_seq[-1]
        u_zmp = zmp_ref_seq[0] + 0.3 * (target_dcm - dcm_pred[-1])
        return u_zmp

    def compute_com_from_dcm(self, dcm):
        com_x = dcm
        dcom_x = self.omega * (dcm - com_x)
        return com_x, dcom_x
