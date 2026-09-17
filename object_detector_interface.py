import numpy as np

class ObjectDetectorInterface:
    def __init__(self, sim_mode=True):
        self.sim_mode = sim_mode
        self.object_world_pos = np.array([0.4, 0.0, 0.0])

    def update_sim(self, sim_object_pos):
        """仿真模式：直接读取MuJoCo物体位置"""
        self.object_world_pos = sim_object_pos.copy()

    def update_real(self, detection_msg):
        """真实相机：接收ROS检测消息，物体世界坐标"""
        self.object_world_pos = np.array([detection_msg.x, detection_msg.y, detection_msg.z])

    def get_grasp_target_pose(self):
        """自动生成抓取目标：物体上方0.1m处"""
        grasp_pos = self.object_world_pos.copy()
        grasp_pos[2] += 0.1
        return grasp_pos
