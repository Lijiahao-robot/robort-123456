import numpy as np

class DomainRandomizer:
    def __init__(self, enable=True):
        self.enable = enable
        # 随机化参数范围
        self.mass_scale_range = (0.9, 1.1)
        self.friction_range = (0.4, 0.8)
        self.joint_damp_range = (0.8, 1.2)
        self.external_force_std = 5.0

    def randomize_model(self, mj_model):
        """随机化MuJoCo模型参数"""
        if not self.enable:
            return
        # 质量随机
        mass_scale = np.random.uniform(*self.mass_scale_range)
        mj_model.body_mass[:] *= mass_scale
        # 摩擦随机
        friction = np.random.uniform(*self.friction_range)
        mj_model.geom_friction[:,0] = friction
        # 关节阻尼随机
        damp_scale = np.random.uniform(*self.joint_damp_range)
        mj_model.dof_damping[:] *= damp_scale

    def apply_random_external_force(self, mj_data):
        """施加随机外部扰动力"""
        if not self.enable:
            return
        force = np.random.normal(0, self.external_force_std, size=3)
        mj_data.xfrc_applied[1,:3] = force
