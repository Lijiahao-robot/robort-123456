
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import PoseStamped

class G1RosInterface:
    def __init__(self):
        rospy.init_node("g1_humanoid_sim_node", anonymous=True)
        # 发布
        self.pub_robot_state = rospy.Publisher("/g1/robot_state", Float32MultiArray, queue_size=10)
        self.pub_zmp = rospy.Publisher("/g1/zmp", Float32MultiArray, queue_size=10)
        self.pub_com = rospy.Publisher("/g1/com", Float32MultiArray, queue_size=10)
        # 订阅
        self.sub_gait_cmd = rospy.Subscriber("/g1/gait_cmd", Float32MultiArray, self.cb_gait_cmd)
        self.sub_terrain_cmd = rospy.Subscriber("/g1/terrain_cmd", Float32MultiArray, self.cb_terrain_cmd)

        self.target_step_len = 0.28
        self.target_step_freq = 1.2
        self.terrain_type = "flat"
        self.slope_deg = 0.0

    def cb_gait_cmd(self, msg):
        """接收步态指令 [step_len, step_freq, mode]"""
        data = msg.data
        self.target_step_len = data[0]
        self.target_step_freq = data[1]

    def cb_terrain_cmd(self, msg):
        """接收地形指令 [terrain_type_id, slope_deg, step_height]"""
        data = msg.data
        terrain_id = int(data[0])
        if terrain_id == 0:
            self.terrain_type = "flat"
        elif terrain_id == 1:
            self.terrain_type = "slope"
            self.slope_deg = data[1]
        elif terrain_id == 2:
            self.terrain_type = "step"

    def publish_state(self, joint_pos, com_xy, zmp_xy):
        msg_state = Float32MultiArray()
        msg_state.data = joint_pos.tolist()
        self.pub_robot_state.publish(msg_state)

        msg_zmp = Float32MultiArray()
        msg_zmp.data = zmp_xy.tolist()
        self.pub_zmp.publish(msg_zmp)

        msg_com = Float32MultiArray()
        msg_com.data = com_xy.tolist()
        self.pub_com.publish(msg_com)

    def get_target_gait_params(self):
        return self.target_step_len, self.target_step_freq

    def get_terrain_params(self):
        return self.terrain_type, self.slope_deg

if __name__ == "__main__":
    ros_if = G1RosInterface()
    rospy.spin()
