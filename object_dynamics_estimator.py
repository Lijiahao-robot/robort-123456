
import numpy as np

class ObjectDynamicsEstimator:
    def __init__(self):
        self.object_mass = 0.0
        self.is_grasping = False
        self.g = 9.81
        # 低通滤波
        self.alpha = 0.05

    def reset(self):
        self.object_mass = 0.0
        self.is_grasping = False

    def update(self, arm_force_wrench, grasp_flag):
        """
        arm_force_wrench：手臂六维力传感器 [fx,fy,fz,tx,ty,tz]
        grasp_flag：是否抓取物体
        """
        self.is_grasping = grasp_flag
        if not self.is_grasping:
            self.object_mass = 0.0
            return

        # 重力方向力估计质量 m = Fz / g
        fz = arm_force_wrench[2]
        mass_obs = fz / self.g
        # 指数低通滤波平滑估计
        self.object_mass = self.alpha * mass_obs + (1-self.alpha)*self.object_mass

    def get_object_mass(self):
        return self.object_mass

    def get_object_gravity_force(self):
        """返回物体重力矢量，用于WBC动力学补偿"""
        return np.array([0,0, self.object_mass * self.g])
