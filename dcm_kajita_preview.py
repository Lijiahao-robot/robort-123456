# 在DCMKajitaPreview类
def __init__(self, com_height=0.92, g=9.81, preview_N=15, dt=0.01):
    # ...原有代码
    self.zmp_bound = (-0.15,0.15) # 动态边界

def solve_preview(self, xi_current, zmp_ref_seq):
    # ...原有
    z_min,z_max = self.zmp_bound
    constraints = [
        u >= z_min,
        u <= z_max
    ]

