import numpy as np

class ForceSensorFeedback:
    def __init__(self, sim_mode=True):
        self.sim_mode = sim_mode
        # 脚端力阈值
        self.contact_threshold = 15.0
        self.left_foot_force = np.zeros(3)
        self.right_foot_force = np.zeros(3)

    def update_sim(self, mj_data, left_geom_id, right_geom_id):
        """仿真模式：从MuJoCo获取脚接触力"""
        l_force = 0.0
        r_force = 0.0
        for i in range(mj_data.ncon):
            con = mj_data.contact[i]
            if con.geom1 == left_geom_id or con.geom2 == left_geom_id:
                l_force += np.linalg.norm(mj_data.contact_force(i))
            if con.geom1 == right_geom_id or con.geom2 == right_geom_id:
                r_force += np.linalg.norm(mj_data.contact_force(i))
        self.left_foot_force[2] = l_force
        self.right_foot_force[2] = r_force

    def update_real(self, left_ft_wrench, right_ft_wrench):
        """真实机器人模式：接收六维力传感器数据，z向为法向力"""
        self.left_foot_force = np.array(left_ft_wrench[:3])
        self.right_foot_force = np.array(right_ft_wrench[:3])

    def get_foot_contact_state(self):
        """返回左右脚是否接触地面"""
        left_contact = self.left_foot_force[2] > self.contact_threshold
        right_contact = self.right_foot_force[2] > self.contact_threshold
        return left_contact, right_contact

    def get_foot_normal_force(self):
        return self.left_foot_force[2], self.right_foot_force[2]
