import numpy as np

class HumanoidIK_G1:
    def __init__(self, thigh_len=0.42, shank_len=0.42, com_height=0.92):
        self.l_thigh = thigh_len
        self.l_shank = shank_len
        self.com_height = com_height

    def leg_ik(self, foot_pos, com_pos):
        hip_pos = com_pos.copy()
        hip_pos[2] -= 0.12
        dx = foot_pos[0] - hip_pos[0]
        dy = foot_pos[1] - hip_pos[1]
        dz = foot_pos[2] - hip_pos[2]

        L = np.sqrt(dx**2 + dz**2)
        L = np.clip(L, abs(self.l_thigh-self.l_shank)+1e-4, self.l_thigh+self.l_shank-1e-4)
        cos_knee = (self.l_thigh**2 + self.l_shank**2 - L**2)/(2*self.l_thigh*self.l_shank)
        cos_knee = np.clip(cos_knee, -0.99, 0.99)
        knee_pitch = np.pi - np.arccos(cos_knee)

        alpha = np.arctan2(dz, dx)
        beta = np.arccos((self.l_thigh**2 + L**2 - self.l_shank**2)/(2*self.l_thigh*L))
        hip_pitch = alpha + beta
        hip_yaw = np.arctan2(dy, dx)
        ankle_pitch = -(hip_pitch + knee_pitch)
        return hip_yaw, hip_pitch, knee_pitch, ankle_pitch

    def compute_joint_from_com(self, com_target, left_foot_pos, right_foot_pos):
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
