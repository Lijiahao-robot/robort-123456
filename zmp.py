import numpy as np
import mujoco

class ZMPCalculator:
    def __init__(self, model, data, ground_z=0.0):
        self.model = model
        self.data = data
        self.ground_z = ground_z

    def compute_zmp(self):
        """
        计算ZMP (x,y)
        返回：zmp_x, zmp_y；无接触返回 None
        原理：ZMP = (M_x / F_z, M_y / F_z)
        M_x/M_y：绕世界坐标系x/y轴的地面力矩
        F_z：竖直方向地面合力
        """
        # 累计所有接触的力与力矩
        total_force = np.zeros(3)
        total_moment = np.zeros(3)

        for contact in self.data.contact:
            # 只处理地面接触
            if contact.dist > 1e-6:
                continue
            # 接触力：世界坐标系
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, contact.id, force)
            f = force[:3]
            m = force[3:]

            # 接触点世界位置
            pos = contact.pos

            # 力矩 = r × f + m
            r = pos - np.array([0, 0, self.ground_z])
            total_force += f
            total_moment += np.cross(r, f) + m

        Fz = total_force[2]
        if abs(Fz) < 1e-4:
            return None

        Mx = total_moment[0]
        My = total_moment[1]

        zmp_x = - My / Fz
        zmp_y =   Mx / Fz
        return np.array([zmp_x, zmp_y])
