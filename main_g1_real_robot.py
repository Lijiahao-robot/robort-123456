
from ros2.g1_hardware_interface import G1HardwareInterface
from controllers.lipm_mpc_2d import LIPM_MPC_2D
from motion.object_dynamics_estimator import ObjectDynamicsEstimator

class G1RealRobotController:
    def __init__(self):
        self.hw_if = G1HardwareInterface()
        self.mpc = LIPM_MPC_2D()
        self.obj_estimator = ObjectDynamicsEstimator()

    def run_loop(self):
        while True:
            joint_pos, joint_vel, l_ft, r_ft, arm_ft = self.hw_if.get_robot_state()
            # 力传感器更新
            self.force_sensor.update_real(l_ft, r_ft)
            # 物体质量估计
            self.obj_estimator.update(arm_ft, grasp_flag=True)
            obj_gravity = self.obj_estimator.get_object_gravity_force()
            # 二维MPC求解
            # ...MPC求解得到参考CoM...
            # WBC‑QP，加入物体重力补偿
            joint_target = self.wbc.solve(..., external_force=obj_gravity)
            # 下发给真实硬件
            self.hw_if.send_joint_cmd(joint_target, kp=200, kd=5)
