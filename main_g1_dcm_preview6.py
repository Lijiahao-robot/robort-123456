# 导入新模块
from motion.terrain_gait_planner import TerrainGaitPlanner
from motion.speed_smoother import SpeedSmoother
from utils.sim_visualizer import SimVisualizer

class HumanoidG1Env:
    def __init__(self, model_path):
        # ...原有初始化...
        # 替换为地形步态规划器
        self.gait_planner = TerrainGaitPlanner()
        self.speed_smoother = SpeedSmoother(alpha=0.08)
        self.visualizer = SimVisualizer(history_len=300)

    def step(self, raw_action):
        # 1. 速度平滑：运行时动态更新步长步频
        target_len, target_freq = self.speed_smoother.get_current_params()
        self.gait_planner.step_length = target_len
        self.gait_planner.step_freq = target_freq

        # 2. 地形参数（可运行时修改）
        terrain_type, slope_deg = self.gait_planner.get_terrain_params()

        # ...原有逻辑：支撑相、冲击补偿、DCM‑Preview、WBC‑QP...

        # 3. 更新可视化
        if real_zmp is not None:
            self.visualizer.update(com_xy=com[:2], zmp_xy=real_zmp, support_poly=support_polygon)

        # 4. 如果开启ROS，发布状态
        if self.use_ros:
            self.ros_if.publish_state(self.data.qpos, com[:2], real_zmp)

        return obs_next, reward, done, real_zmp, ref_zmp, real_com
