
# 导入新模块
from motion.force_sensor_feedback import ForceSensorFeedback
from controllers.lipm_mpc import LIPM_MPC
from motion.task_manager import TaskManager
from utils.domain_randomizer import DomainRandomizer

class HumanoidG1Env:
    def __init__(self, model_path, sim_mode=True, use_mpc=True, enable_domain_rand=True):
        # 原有初始化
        self.sim_mode = sim_mode
        self.use_mpc = use_mpc

        # 新模块初始化
        self.force_sensor = ForceSensorFeedback(sim_mode=sim_mode)
        self.mpc_controller = LIPM_MPC(com_height=0.92, dt=0.01, horizon=20)
        self.task_manager = TaskManager()
        self.domain_rand = DomainRandomizer(enable=enable_domain_rand)

        # 模型随机化
        self.domain_rand.randomize_model(self.model)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.domain_rand.randomize_model(self.model)
        # ...原有reset逻辑...
        return obs

    def step(self, raw_action):
        # 1. 力传感器更新
        if self.sim_mode:
            self.force_sensor.update_sim(self.data, left_geom_id=10, right_geom_id=11)
        else:
            # 真实机器人：ROS订阅力传感器数据，调用 update_real
            pass
        left_contact, right_contact = self.force_sensor.get_foot_contact_state()

        # 2. 域随机化随机外力
        self.domain_rand.apply_random_external_force(self.data)

        # 3. 任务管理器判断任务模式
        if not self.task_manager.is_walking_task():
            # 抓取/抬物体任务：停止步态，原地双支撑
            self.gait_planner.reset()
        else:
            # 正常行走步态
            pass

        # 4. 切换控制器：MPC替代DCM‑Preview
        com_state = np.array([self.data.subtree_com[0], self.data.subtree_vel[0]])
        if self.use_mpc:
            com_x_seq, dcom_x_seq, zmp_opt_seq = self.mpc_controller.solve(
                x0=com_state,
                zmp_ref_seq=zmp_ref_seq,
                support_polygon=support_polygon
            )
            ref_com = com_x_seq[0]
        else:
            # 回退DCM‑Preview
            pass

        # 5. IK得到下肢关节，交给任务管理器生成全身关节目标
        gait_ik_joint = self.ik.solve(com_target=ref_com, foot_targets=foot_targets)
        full_joint_target = self.task_manager.get_desired_joint_target(gait_ik_joint)

        # 6. WBC‑QP跟踪全身关节目标 + ZMP约束
        tau = self.wbc.solve(desired_joint_acc=full_joint_target, ref_zmp=ref_zmp, impact_weight=impact_weight)

        # ...后续可视化、ROS发布、日志记录...
        return obs_next, reward, done, real_zmp, ref_zmp, real_com
