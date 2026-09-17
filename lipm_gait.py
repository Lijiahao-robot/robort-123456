import numpy as np
from motion.zmp_gait_generator import ZMPGaitGenerator
from motion.lipm import LIPM

class LIPMGaitPlanner:
    def __init__(self,
                 com_height=0.8,
                 step_length=0.25,
                 step_width=0.14,
                 T_support=0.7,
                 T_double=0.2,
                 dt=0.01):
        self.dt = dt
        self.lipm_x = LIPM(com_height=com_height)
        self.lipm_y = LIPM(com_height=com_height)
        self.zmp_gait = ZMPGaitGenerator(
            step_length=step_length,
            step_width=step_width,
            T_support=T_support,
            T_double=T_double
        )
        # 初始质心状态
        self.cm_x = 0.0
        self.dcm_x = 0.0
        self.cm_y = 0.0
        self.dcm_y = 0.0

    def reset(self):
        self.zmp_gait.reset()
        self.cm_x = 0.0
        self.dcm_x = 0.0
        self.cm_y = 0.0
        self.dcm_y = 0.0

    def plan_one_step(self):
        """
        规划一个完整步态周期：输出
        ref_zmp_x, ref_zmp_y, cm_x, cm_y, dcm_x, dcm_y
        """
        T_cycle = self.zmp_gait.T_cycle
        t_list = np.arange(0, T_cycle, self.dt)
        N = len(t_list)

        ref_zmp_x = np.zeros(N)
        ref_zmp_y = np.zeros(N)

        for i, t in enumerate(t_list):
            self.zmp_gait.phase = t
            zmp = self.zmp_gait.get_reference_zmp()
            ref_zmp_x[i] = zmp[0]
            ref_zmp_y[i] = zmp[1]

        # LIPM求解质心轨迹
        cm_x, dcm_x = self.lipm_x.solve_cm_from_zmp(
            ref_zmp_x, self.cm_x, self.dcm_x, t_list, self.dt
        )
        cm_y, dcm_y = self.lipm_y.solve_cm_from_zmp(
            ref_zmp_y, self.cm_y, self.dcm_y, t_list, self.dt
        )

        # 更新下一段初始条件
        self.cm_x = cm_x[-1]
        self.dcm_x = dcm_x[-1]
        self.cm_y = cm_y[-1]
        self.dcm_y = dcm_y[-1]

        return t_list, ref_zmp_x, ref_zmp_y, cm_x, cm_y, dcm_x, dcm_y
