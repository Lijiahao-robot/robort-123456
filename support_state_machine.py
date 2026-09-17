import numpy as np

class SupportStateMachine:
    def __init__(self, force_threshold=20.0):
        self.force_threshold = force_threshold
        self.state = "DOUBLE_SUPPORT" # DOUBLE_SUPPORT / SINGLE_SUPPORT_LEFT / SINGLE_SUPPORT_RIGHT
        self.swing_foot = None
        self.support_foot = None

    def update(self, left_foot_force, right_foot_force):
        """
        :param left_foot_force: 左脚竖直接触力
        :param right_foot_force: 右脚竖直接触力
        """
        left_contact = left_foot_force > self.force_threshold
        right_contact = right_foot_force > self.force_threshold

        if left_contact and right_contact:
            self.state = "DOUBLE_SUPPORT"
            self.swing_foot = None
            self.support_foot = "both"
        elif left_contact and not right_contact:
            self.state = "SINGLE_SUPPORT_LEFT"
            self.swing_foot = "right"
            self.support_foot = "left"
        elif right_contact and not left_contact:
            self.state = "SINGLE_SUPPORT_RIGHT"
            self.swing_foot = "left"
            self.support_foot = "right"
        else:
            # 双脚离地，摔倒状态
            self.state = "NO_SUPPORT"

    def get_support_polygon(self):
        """返回当前支撑多边形ZMP约束边界"""
        if self.state == "DOUBLE_SUPPORT":
            return (-0.20, 0.20, -0.15, 0.15)
        elif self.state == "SINGLE_SUPPORT_LEFT":
            return (-0.12, 0.12, -0.10, 0.10)
        elif self.state == "SINGLE_SUPPORT_RIGHT":
            return (-0.12, 0.12, -0.10, 0.10)
        else:
            return (-0.15,0.15,-0.15,0.15)

    def is_swing_phase(self):
        return self.state in ["SINGLE_SUPPORT_LEFT", "SINGLE_SUPPORT_RIGHT"]
