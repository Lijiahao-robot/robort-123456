import matplotlib.pyplot as plt
import numpy as np

class SimVisualizer:
    def __init__(self, history_len=300):
        self.history_len = history_len
        self.com_history = []
        self.zmp_history = []
        self.support_polygons = []

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6,6))

    def update(self, com_xy, zmp_xy, support_poly):
        """
        com_xy: np.array([x,y])
        zmp_xy: np.array([x,y])
        support_poly: (xmin,xmax,ymin,ymax)
        """
        self.com_history.append(com_xy)
        self.zmp_history.append(zmp_xy)
        self.support_polygons.append(support_poly)

        if len(self.com_history) > self.history_len:
            self.com_history.pop(0)
            self.zmp_history.pop(0)
            self.support_polygons.pop(0)

        self._draw()

    def _draw(self):
        self.ax.clear()
        # 绘制支撑多边形
        xmin,xmax,ymin,ymax = self.support_polygons[-1]
        rect_x = [xmin,xmax,xmax,xmin,xmin]
        rect_y = [ymin,ymin,ymax,ymax,ymin]
        self.ax.plot(rect_x, rect_y, 'g-', label="support polygon")

        com_arr = np.array(self.com_history)
        zmp_arr = np.array(self.zmp_history)
        self.ax.plot(com_arr[:,0], com_arr[:,1], 'b-', label="CoM")
        self.ax.plot(zmp_arr[:,0], zmp_arr[:,1], 'r--', label="ZMP")

        self.ax.set_xlim(-0.6, 0.6)
        self.ax.set_ylim(-0.4, 0.4)
        self.ax.set_xlabel("X(m)")
        self.ax.set_ylabel("Y(m)")
        self.ax.set_aspect("equal")
        self.ax.legend()
        self.ax.grid(True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save_figure(self, save_path="sim_plot.png"):
        self.fig.savefig(save_path, dpi=150)

    def close(self):
        plt.ioff()
        plt.close(self.fig)
