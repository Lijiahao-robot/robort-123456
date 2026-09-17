import numpy as np

class GaitPlanner:
    def __init__(self, step_length=0.28, step_freq=1.2, step_height=0.09, lateral_offset=0.08):
        self.step_length = step_length      # 前进步长(m)
        self.step_freq = step_freq          # 步频 Hz
        self.step_height = step_height      # 抬脚高度
        self.lateral_offset = lateral_offset # 左右脚横向偏移
        self.T_step = 1.0 / step_freq        # 单步周期
        self.T_swing = 0.5 * self.T_step    # 摆动相时间
        self.T_double = 0.5 * self.T_step   # 双支撑相时间

        self.step_count = 0
        self.current_foot_pos_left = np.array([0.0,  self.lateral_offset, 0.0])
        self.current_foot_pos_right = np.array([0.0, -self.lateral_offset, 0.0])

    def reset(self):
        """重置步态规划，回到初始位置"""
        self.step_count = 0
        self.current_foot_pos_left = np.array([0.0,  self.lateral_offset, 0.0])
        self.current_foot_pos_right = np.array([0.0, -self.lateral_offset, 0.0])

    def generate_next_step(self):
        """生成下一步摆动脚目标位置，交替左脚、右脚迈步"""
        step_idx = self.step_count
        dx = self.step_length

        if step_idx % 2 == 0:
            # 右脚摆动
            swing_foot = "right"
            target_foot = self.current_foot_pos_right.copy()
            target_foot[0] += dx
            self.current_foot_pos_right = target_foot
        else:
            # 左脚摆动
            swing_foot = "left"
            target_foot = self.current_foot_pos_left.copy()
            target_foot[0] += dx
            self.current_foot_pos_left = target_foot

        self.step_count += 1
        return swing_foot, target_foot

    def get_gait_timing(self, t_sim):
        """返回当前时刻：摆动开始时间、摆动结束时间、当前步态相位0~1"""
        cycle = np.floor(t_sim / self.T_step)
        t_in_cycle = t_sim - cycle * self.T_step
        t_swing_start = cycle * self.T_step
        t_swing_end = t_swing_start + self.T_swing
        phase = np.clip((t_in_cycle) / self.T_step, 0.0, 1.0)
        return t_swing_start, t_swing_end, phase

    def get_foot_positions(self):
        """返回当前左右脚期望位置"""
        return self.current_foot_pos_left.copy(), self.current_foot_pos_right.copy()
