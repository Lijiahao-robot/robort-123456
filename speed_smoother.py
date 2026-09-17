import numpy as np

class SpeedSmoother:
    def __init__(self, alpha=0.08):
        self.alpha = alpha
        self.cur_step_len = 0.28
        self.cur_step_freq = 1.2

    def set_target(self, target_step_len, target_step_freq):
        """设置目标步长、步频，不直接赋值，平滑过渡"""
        self.cur_step_len = self.alpha * target_step_len + (1-self.alpha)*self.cur_step_len
        self.cur_step_freq = self.alpha * target_step_freq + (1-self.alpha)*self.cur_step_freq

    def get_current_params(self):
        return self.cur_step_len, self.cur_step_freq
