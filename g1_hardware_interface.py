import rclpy
from rclpy.node import Node
import numpy as np
from unitree_go2_msgs.msg import LowCmd, LowState
from sensor_msgs.msg import JointState, WrenchStamped

class G1HardwareInterface(Node):
    def __init__(self):
        super().__init__("g1_hardware_interface_node")
        # 订阅：机器人状态、力传感器
        self.sub_low_state = self.create_subscription(LowState, "/g1/low_state", self.cb_low_state, 10)
        self.sub_foot_ft_left = self.create_subscription(WrenchStamped, "/g1/ft/left_foot", self.cb_ft_left,10)
        self.sub_foot_ft_right = self.create_subscription(WrenchStamped, "/g1/ft/right_foot", self.cb_ft_right,10)
        self.sub_arm_ft = self.create_subscription(WrenchStamped, "/g1/ft/right_arm", self.cb_arm_ft,10)

        # 发布：关节控制指令
        self.pub_low_cmd = self.create_publisher(LowCmd, "/g1/low_cmd", 10)

        self.joint_pos = np.zeros(28)
        self.joint_vel = np.zeros(28)
        self.left_foot_wrench = np.zeros(6)
        self.right_foot_wrench = np.zeros(6)
        self.arm_wrench = np.zeros(6)

    def cb_low_state(self, msg: LowState):
        """读取G1关节位置速度"""
        for i in range(28):
            self.joint_pos[i] = msg.motor_state[i].q
            self.joint_vel[i] = msg.motor_state[i].dq

    def cb_ft_left(self, msg: WrenchStamped):
        self.left_foot_wrench = np.array([
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        ])

    def cb_ft_right(self, msg: WrenchStamped):
        self.right_foot_wrench = np.array([
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        ])

    def cb_arm_ft(self, msg: WrenchStamped):
        self.arm_wrench = np.array([
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        ])

    def send_joint_cmd(self, joint_target_pos, kp, kd):
        """下发关节位置PD指令到G1硬件"""
        cmd = LowCmd()
        for i in range(28):
            cmd.motor_cmd[i].q = joint_target_pos[i]
            cmd.motor_cmd[i].kp = kp
            cmd.motor_cmd[i].kd = kd
            cmd.motor_cmd[i].tau = 0.0
        self.pub_low_cmd.publish(cmd)

    def get_robot_state(self):
        return self.joint_pos, self.joint_vel, self.left_foot_wrench, self.right_foot_wrench, self.arm_wrench

def main(args=None):
    rclpy.init(args=args)
    node = G1HardwareInterface()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
