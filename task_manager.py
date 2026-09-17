import numpy as np

class TaskManager:
    def __init__(self):
        self.task_mode = "walk" # walk / grasp_stand / lift_object
        # 抓取任务目标
        self.arm_target_pos = np.array([0.3,0.0,0.8])
        self.object_height = 0.0

    def set_task_mode(self, mode, arm_target=None, obj_height=0.0):
        self.task_mode = mode
        if arm_target is not None:
            self.arm_target_pos = arm_target
        self.object_height = obj_height

    def get_desired_joint_target(self, gait_ik_joint):
        """
        gait_ik_joint：行走IK输出关节角度
        返回：任务模式下完整关节目标；抓取模式覆盖手臂关节
        """
        joint_target = gait_ik_joint.copy()
        if self.task_mode == "grasp_stand":
            # 原地抓取：手臂运动，下肢保持站立
            joint_target[12:18] = self.arm_target_pos # 左手臂关节
            joint_target[18:24] = self.arm_target_pos # 右手臂关节
        elif self.task_mode == "lift_object":
            # 抬物体：手臂抬升
            joint_target[12:18] = np.array([0.2,0.1,self.object_height+0.6])
            joint_target[18:24] = np.array([0.2,-0.1,self.object_height+0.6])
        return joint_target

    def is_walking_task(self):
        return self.task_mode == "walk"
