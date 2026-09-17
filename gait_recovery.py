import numpy as np

class GaitRecovery:
    def __init__(self, margin_threshold=-0.02, tilt_threshold=0.3):
        self.margin_threshold = margin_threshold
        self.tilt_threshold = tilt_threshold
        self.recovery_flag = False
        self.recovery_timer = 0.0

    def check_need_recovery(self, zmp_margin, torso_tilt):
        """判断是否需要步态恢复"""
        if zmp_margin < self.margin_threshold or abs(torso_tilt) > self.tilt_threshold:
            return True
        return False

    def reset_gait_for_recovery(self, gait_planner, dcm_preview, com_pos):
        """
        扰动后恢复：
        1.重置步态规划，减小步长；
        2.重置DCM状态，把DCM拉回支撑脚中心；
        """
        gait_planner.reset()
        gait_planner.step_length = 0.12  # 恢复模式减小步长
        # 重置DCM，DCM初始值等于CoM位置
        dcm_preview.xi_current = np.array([com_pos[0], 0.0])
        self.recovery_flag = True
        self.recovery_timer = 0.0

    def update_recovery_timer(self, dt):
        self.recovery_timer += dt
        if self.recovery_timer > 2.0:
            # 恢复完成，切回正常步态
            self.recovery_flag = False
            return False
        return True
