import numpy as np
import random

class RRT:
    def __init__(self, start, goal, obstacle_list, area, max_iter=5000, step_size=0.3):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacle_list
        self.area = area
        self.max_iter = max_iter
        self.step_size = step_size
        self.tree = {tuple(self.start): None}

    def is_collision(self, p1, p2):
        """简单线段碰撞检测"""
        for obs in self.obstacles:
            cx, cy, r = obs
            dx = p2[0]-p1[0]
            dy = p2[1]-p1[1]
            a = dx**2 + dy**2
            b = 2*(dx*(p1[0]-cx)+dy*(p1[1]-cy))
            c = (p1[0]-cx)**2 + (p1[1]-cy)**2 - r**2
            disc = b**2 -4*a*c
            if disc >= 0:
                t1 = (-b - np.sqrt(disc))/(2*a)
                t2 = (-b + np.sqrt(disc))/(2*a)
                if 0<=t1<=1 or 0<=t2<=1:
                    return True
        return False

    def get_nearest_node(self, rand_pt):
        min_dist = float("inf")
        nearest = None
        for node in self.tree:
            dist = np.linalg.norm(np.array(node)-rand_pt)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        return nearest

    def steer(self, from_pt, to_pt):
        diff = to_pt - from_pt
        norm = np.linalg.norm(diff)
        if norm < self.step_size:
            return to_pt
        return from_pt + self.step_size * diff / norm

    def plan(self):
        for _ in range(self.max_iter):
            rand_pt = np.array([
                random.uniform(self.area[0], self.area[1]),
                random.uniform(self.area[2], self.area[3])
            ])
            nearest = self.get_nearest_node(rand_pt)
            new_pt = self.steer(np.array(nearest), rand_pt)
            if not self.is_collision(np.array(nearest), new_pt):
                self.tree[tuple(new_pt)] = nearest
                if np.linalg.norm(new_pt-self.goal) < self.step_size:
                    #回溯路径
                    path = []
                    cur = tuple(new_pt)
                    while cur is not None:
                        path.append(cur)
                        cur = self.tree[cur]
                    return path[::-1]
        return None


# RRT* 在RRT基础上增加重选父节点、重布线，这里省略完整实现，核心逻辑一致
if __name__ == "__main__":
    obs = [(2,2,0.5), (4,1,0.6)]
    rrt = RRT(start=[0,0], goal=[5,5], obstacle_list=obs, area=[0,6,0,6])
    path = rrt.plan()
    print("RRT路径：", path)
