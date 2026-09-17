import numpy as np

class DisturbanceInjector:
    def __init__(self):
        self.disturb_active = False
        self.disturb_start_time = 0.0
        self.disturb_duration = 0.0
        self.force_vec = np.zeros(3)

    def apply_impulse_disturbance(self, t_sim, impulse_force, duration=0.2):
        """瞬时脉冲扰动"""
        self.disturb_active = True
        self.disturb_start_time = t_sim
        self.disturb_duration = duration
        self.force_vec = impulse_force.copy()

    def apply_constant_disturbance(self, t_sim, const_force, duration=3.0):
        """持续恒定外力扰动"""
        self.disturb_active = True
        self.disturb_start_time = t_sim
        self.disturb_duration = duration
        self.force_vec = const_force.copy()

    def get_disturb_force(self, t_sim):
        if not self.disturb_active:
            return np.zeros(3)
        elapsed = t_sim - self.disturb_start_time
        if elapsed > self.disturb_duration:
            self.disturb_active = False
            return np.zeros(3)
        return self.force_vec

    def reset(self):
        self.disturb_active = False
        self.force_vec = np.zeros(3)
