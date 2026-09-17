
import numpy as np

class HumanoidIK:
    def __init__(self, thigh_len=0.4, shank_len=0.4, com_height=0.8):
        """
        双腿人形解析IK
        :param thigh_len: 大腿长度 m
        :param shank_len: 小腿长度 m
        :param com_height: 标称质心高度
        """
        self.l_thigh = thigh_len
        self.l_shank = shank_len
        self.com_height = com_height

    def leg_ik(self, foot_pos, com_pos):
        """
        单腿逆运动学：脚位置 foot_pos [x,y,z]，质心com_pos [x,y,z]
        返回：hip_yaw, hip_pitch, knee_pitch, ankle_pitch
        """
        # 髋部位置：com减去躯干偏移，简化
        hip_pos = com_pos.copy()
        hip_pos[2] -= 0.1
        # 脚相对于髋的矢量
        dx = foot_pos[0] - hip_pos[0]
        dy = foot_pos[1] - hip_pos[1]
        dz = foot_pos[2] - hip_pos[2]

        # 平面内求解膝关节
        L = np.sqrt(dx**2 + dz**2)
        if L > self.l_thigh + self.l_shank:
            L = self.l_thigh + self.l_shank - 1e-4
        if L < abs(self.l_thigh - self.l_shank):
            L = abs(self.l_thigh - self.l_shank) + 1e-4

        # 余弦定理求膝角
        cos_knee = (self.l_thigh**2 + self.l_shank**2 - L**2) / (2 * self.l_thigh * self.l_shank)
        cos_knee = np.clip(cos_knee, -0.99, 0.99)
        knee_pitch = np.pi - np.arccos(cos_knee)

        # 髋俯仰角
        alpha = np.arctan2(dz, dx)
        beta = np.arccos((self.l_thigh**2 + L**2 - self.l_shank**2)/(2*self.l_thigh*L))
        hip_pitch = alpha + beta

        hip_yaw = np.arctan2(dy, dx)
        ankle_pitch = -(hip_pitch + knee_pitch)

        return hip_yaw, hip_pitch, knee_pitch, ankle_pitch

    def compute_joint_from_com(self, com_target, left_foot_pos, right_foot_pos):
        """
        输入：质心目标com_target [x,y,z]，左右脚世界坐标
        返回：dict 关节角度
        """
        l_hip_yaw, l_hip_pitch, l_knee, l_ankle = self.leg_ik(left_foot_pos, com_target)
        r_hip_yaw, r_hip_pitch, r_knee, r_ankle = self.leg_ik(right_foot_pos, com_target)

        joint_angles = {
            "left_hip_yaw": l_hip_yaw,
            "left_hip_pitch": l_hip_pitch,
            "left_knee": l_knee,
            "left_ankle": l_ankle,
            "right_hip_yaw": r_hip_yaw,
            "right_hip_pitch": r_hip_pitch,
            "right_knee": r_knee,
            "right_ankle": r_ankle,
        }
        return joint_angles
