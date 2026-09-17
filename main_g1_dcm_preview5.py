# 在__init__增加
from motion.impact_compensator import ImpactCompensator
self.impact_comp = ImpactCompensator()

# step函数内部：
# 获取摆动脚接触力
l_force, r_force = self._get_foot_contact_force()
# 判断摆动脚，检测落地
if self.support_sm.swing_foot == "right":
    swing_foot_force = r_force
else:
    swing_foot_force = l_force

self.impact_comp.detect_landing(swing_foot_force, self.t_sim)
self.impact_comp.update(self.t_sim)
impact_weight = self.impact_comp.get_impact_regularization_weight()

# 传给WBC‑QP，使用impact_weight作为加速度惩罚权重
tau = self.wbc.solve(desired_joint_acc, ref_zmp, impact_weight=impact_weight)
