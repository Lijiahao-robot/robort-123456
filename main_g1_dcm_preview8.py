# 导入新模块
from controllers.lipm_mpc_2d import LIPM_MPC_2D
from motion.object_dynamics_estimator import ObjectDynamicsEstimator
from vision.object_detector_interface import ObjectDetectorInterface

class HumanoidG1Env:
    def __init__(self, model_path, sim_mode=True, use_mpc_2d=True, enable_domain_rand=True):
        # 原有初始化
        self.use_mpc_2d = use_mpc_2d
        self.mpc_2d = LIPM_MPC_2D(com_height=0.92, dt=0.01, horizon=20)
        self.obj_estimator = ObjectDynamicsEstimator()
        self.vision_detector = ObjectDetectorInterface(sim_mode=sim_mode)

    def step(self, raw_action):
        # 1.视觉更新
        if self.sim_mode:
            self.vision_detector.update_sim(sim_object_pos=self.data.body("object").xpos)
        grasp_target = self.vision_detector.get_grasp_target_pose()
        self.task_manager.set_task_mode("grasp_stand", arm_target=grasp_target)

        # 2.物体动力学估计
        arm_wrench = self.get_arm_force()
        self.obj_estimator.update(arm_wrench, grasp_flag=self.task_manager.is_grasping)
        obj_gravity = self.obj_estimator.get_object_gravity_force()

        # 3.二维MPC求解X+Y
        com_x = self.data.subtree_com[0]
        dcom_x = self.data.subtree_vel[0]
        com_y = self.data.subtree_com[1]
        dcom_y = self.data.subtree_vel[1]
        x0 = np.array([com_x, dcom_x])
        y0 = np.array([com_y, dcom_y])

        com_x_seq, dcom_x_seq, com_y_seq, dcom_y_seq, zmp_x_opt, zmp_y_opt = self.mpc_2d.solve(
            x0=x0, y0=y0,
            zmp_ref_x_seq=zmp_ref_x_seq,
            zmp_ref_y_seq=zmp_ref_y_seq,
            support_poly=support_polygon
        )
        ref_com = np.array([com_x_seq[0], com_y_seq[0]])

        # 4.WBC‑QP，传入物体重力外部力补偿
        tau = self.wbc.solve(desired_joint_acc=full_joint_target,
                             ref_zmp=np.array([zmp_x_opt[0], zmp_y_opt[0]]),
                             impact_weight=impact_weight,
                             external_force=obj_gravity)

        # ...后续逻辑
        return obs_next, reward, done, real_zmp, ref_zmp, real_com
