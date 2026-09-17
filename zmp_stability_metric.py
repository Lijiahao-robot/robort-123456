import numpy as np

class ZMPStabilityMetric:
    def compute_margin(self, zmp_xy, support_polygon):
        """
        :param zmp_xy: np.array([x,y]) 实测ZMP
        :param support_polygon: (xmin,xmax,ymin,ymax)
        :return: margin, is_stable
        """
        xmin, xmax, ymin, ymax = support_polygon
        dx_left = zmp_xy[0] - xmin
        dx_right = xmax - zmp_xy[0]
        dy_down = zmp_xy[1] - ymin
        dy_up = ymax - zmp_xy[1]

        margin = min(dx_left, dx_right, dy_down, dy_up)
        is_stable = margin > 0.0
        return margin, is_stable

    def compute_stability_score(self, margin):
        """归一化稳定分数 0~1，越大越稳定"""
        max_margin = 0.12
        score = np.clip(margin / max_margin, 0.0, 1.0)
        return score
