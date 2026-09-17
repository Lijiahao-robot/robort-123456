import numpy as np

class ZMPGaitGenerator:
    def __init__(self,
                 step_length=0.3,    # 单步步长(m)
                 step_width=0.15,    # 左右脚间距(m)
                 T_support=0.8,      # 单脚支撑时间
                 T_double=0.2,       # 双脚支撑时间
                 start_x=0.0,
                 start_y=0.0):
        self.step_length = step_length
        self.step_width = step_width
        self.T_support = T_support
        self.T_double = T_double
        self.T_cycle = T_support + T_double

        # 脚的基准位置
        self.left_foot_y =  start_y + step_width/2
        self.right_foot_y = start_y - step_width/2

        self.phase = 0.0  # 0~T_cycle
        self.step_count = 0

    def reset(self):
        self.phase = 0.0
        self.step_count = 0

    def update(self, dt):
        self.phase += dt
        if self.phase >= self.T_cycle:
            self.phase -= self.T_cycle
            self.step_count += 1

    def get_reference_zmp(self):
        """
        返回参考ZMP [zmp_x, zmp_y]
        阶段划分：
        0~T_double：双脚支撑，ZMP在两脚中间
        T_double ~ T_cycle：单脚支撑，ZMP在支撑脚位置
        """
        t = self.phase
        cycle = self.T_cycle
        Td = self.T_double
        Ts = self.T_support

        # 行走方向x：每周期向前走一个步长
        base_x = self.step_count * self.step_length

        if t < Td:
            # 双脚支撑阶段：ZMP在两脚中间
            zmp_x = base_x
            zmp_y = 0.0
        else:
            # 单脚支撑：交替左脚、右脚支撑
            t_support = t - Td
            if self.step_count % 2 == 0:
                # 左脚支撑
                zmp_x = base_x + self.step_length * (t_support / Ts)
                zmp_y = self.left_foot_y
            else:
                # 右脚支撑
                zmp_x = base_x + self.step_length * (t_support / Ts)
                zmp_y = self.right_foot_y

        return np.array([zmp_x, zmp_y])
