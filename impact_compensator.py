import numpy as np

class ImpactCompensator:
    def __init__(self, impact_threshold=15.0, compensation_duration=0.15):
        self.impact_threshold = impact_threshold
        self.compensation_duration = compensation_duration
        self.in_impact = False
        self.impact_start_time = 0.0

    def detect_landing(self, swing_foot_force, t_sim):
        """检测摆动脚落地：接触力超过阈值，触发冲击补偿"""
        if swing_foot_force > self.impact_threshold and not self.in_impact:
            self.in_impact = True
            self.impact_start_time = t_sim
            return True
        return False

    def update(self, t_sim):
        """更新冲击补偿状态"""
        if self.in_impact:
            elapsed = t_sim - self.impact_start_time
            if elapsed > self.compensation_duration:
                self.in_impact = False

    def get_impact_regularization_weight(self):
        """落地时增大加速度惩罚权重，抑制冲击；正常步态返回小权重"""
        if self.in_impact:
            return 15.0
        else:
            return 1.0
