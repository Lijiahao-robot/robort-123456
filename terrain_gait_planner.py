import numpy as np

class TerrainGaitPlanner:
    def __init__(self, step_length=0.28, step_freq=1.2, step_height=0.09, lateral_offset=0.08):
        self.step_length = step_length
        self.step_freq = step_freq
        self.step_height = step_height
        self.lateral_offset = lateral_offset
        self.T_step = 1.0 / step_freq
        self.T_swing = 0.5 * self.T_step
        self.T_double = 0.5 * self.T_step

        self.step_count = 0
        self.current_foot_pos_left = np.array([0.0,  self.lateral_offset, 0.0])
        self.current_foot_pos_right = np.array([0.0, -self.lateral_offset, 0.0])

        self.mode = "forward"
        self.lateral_step = 0.08
        self.turn_omega = 0.3

        # 地形参数
        self.terrain_type = "flat"   # flat / slope / step
        self.slope_angle_deg = 0.0   # 斜坡角度(°)
        self.step_height_m = 0.05    # 台阶高度(m)

    def set_terrain(self, terrain_type, slope_angle_deg=0.0, step_height_m=0.05):
        self.terrain_type = terrain_type
        self.slope_angle_deg = slope_angle_deg
        self.step_height_m = step_height_m

    def set_mode(self, mode, lateral_step=0.08, turn_omega=0.3):
        self.mode = mode
        self.lateral_step = lateral_step
        self.turn_omega = turn_omega

    def reset(self):
        self.step_count = 0
        self.current_foot_pos_left = np.array([0.0,  self.lateral_offset, 0.0])
        self.current_foot_pos_right = np.array([0.0, -self.lateral_offset, 0.0])

    def _get_foot_z_offset(self, x_pos):
        """根据地形计算脚Z偏移"""
        if self.terrain_type == "flat":
            return 0.0
        elif self.terrain_type == "slope":
            # 斜坡：z = x * tan(坡度)
            slope_rad = np.deg2rad(self.slope_angle_deg)
            return x_pos * np.tan(slope_rad)
        elif self.terrain_type == "step":
            # 台阶：每走N步抬升一次
            if self.step_count % 2 == 0:
                return self.step_height_m
            else:
                return 0.0
        return 0.0

    def generate_next_step(self):
        step_idx = self.step_count
        dx = 0.0
        dy = 0.0

        if self.mode == "forward":
            dx = self.step_length
        elif self.mode == "lateral":
            dy = self.lateral_step
        elif self.mode == "turn_in_place":
            dtheta = self.turn_omega * self.T_step
            if step_idx % 2 == 0:
                dy = -0.06
            else:
                dy = +0.06

        if step_idx % 2 == 0:
            swing_foot = "right"
            target_foot = self.current_foot_pos_right.copy()
            target_foot[0] += dx
            target_foot[1] += dy
            target_foot[2] = self._get_foot_z_offset(target_foot[0])
            self.current_foot_pos_right = target_foot
        else:
            swing_foot = "left"
            target_foot = self.current_foot_pos_left.copy()
            target_foot[0] += dx
            target_foot[1] += dy
            target_foot[2] = self._get_foot_z_offset(target_foot[0])
            self.current_foot_pos_left = target_foot

        self.step_count += 1
        return swing_foot, target_foot

    def get_gait_timing(self, t_sim):
        cycle = np.floor(t_sim / self.T_step)
        t_in_cycle = t_sim - cycle * self.T_step
        t_swing_start = cycle * self.T_step
        t_swing_end = t_swing_start + self.T_swing
        phase = np.clip((t_in_cycle) / self.T_step, 0.0, 1.0)
        return t_swing_start, t_swing_end, phase

    def get_foot_positions(self):
        return self.current_foot_pos_left.copy(), self.current_foot_pos_right.copy()

    def get_lipm_com_height(self):
        """斜坡修正LIPM质心高度"""
        if self.terrain_type == "slope":
            base_h = 0.92
            slope_rad = np.deg2rad(self.slope_angle_deg)
            return base_h / np.cos(slope_rad)
        return 0.92
