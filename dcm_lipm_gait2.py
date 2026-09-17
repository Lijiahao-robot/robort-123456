import numpy as np
from motion.zmp_gait_generator import ZMPGaitGenerator
from motion.lipm import LIPM
from motion.dcm import DCM

class DCMLIPMGaitPlanner:
    def __init__(self, com_height=0.8, step_length=0.25, step_width=0.14,
                 T_support=0.7, T_double=0.2, dt=0.01):
        self.dt = dt
        self.lipm = LIPM(com_height=com_height)
        self.dcm = DCM(com_height=com_height)
        self.zmp_gait = ZMPGaitGenerator(step_length, step_width, T_support, T_double)

        self.cm_x = 0.0
        self.dcm_x = 0.0
        self.cm_y = 0.0
        self.dcm_y = 0.0
        self.last_phase = 0.0

    def reset(self):
        self.zmp_gait.reset()
        self.cm_x = 0.0
        self.dcm_x = 0.0
        self.cm_y = 0.0
        self.dcm_y = 0.0
        self.last_phase = 0.0

    def plan_one_step(self):
        T_cycle = self.zmp_gait.T_cycle
        t_list = np.arange(0, T_cycle, self.dt)
        N = len(t_list)
        ref_zmp_x = np.zeros(N)
        ref_zmp_y = np.zeros(N)
        dcm_x_seq = np.zeros(N)
        dcm_y_seq = np.zeros(N)
        cm_x_seq = np.zeros(N)
        cm_y_seq = np.zeros(N)

        for i, t in enumerate(t_list):
            self.zmp_gait.phase = t
            zmp = self.zmp_gait.get_reference_zmp()
            ref_zmp_x[i] = zmp[0]
            ref_zmp_y[i] = zmp[1]

        # DCM积分 + 落地时刻重置逻辑
        for i in range(N-1):
            # 判断：单脚支撑结束，脚落地，触发DCM重置
            phase_now = self.zmp_gait.phase
            phase_next = phase_now + self.dt
            # 跨越周期边界：脚落地，重置DCM初始条件
            if phase_next >= self.zmp_gait.T_cycle and self.last_phase < self.zmp_gait.T_cycle:
                # 落地重置：DCM初始化为新支撑脚ZMP位置
                self.dcm_x = ref_zmp_x[i]
                self.dcm_y = ref_zmp_y[i]

            self.dcm_x = self.dcm.dcm_integrate(ref_zmp_x[i], self.dcm_x, self.dt)
            self.dcm_y = self.dcm.dcm_integrate(ref_zmp_y[i], self.dcm_y, self.dt)
            dcm_x_seq[i] = self.dcm_x
            dcm_y_seq[i] = self.dcm_y
            cm_x_seq[i] = self.dcm_x - self.dcm_x / self.dcm.omega
            cm_y_seq[i] = self.dcm_y - self.dcm_y / self.dcm.omega
            self.last_phase = phase_now

        return t_list, ref_zmp_x, ref_zmp_y, cm_x_seq, cm_y_seq, dcm_x_seq, dcm_y_seq
