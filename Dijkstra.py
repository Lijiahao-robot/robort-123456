import heapq

def dijkstra(grid, start, goal):
    rows, cols = grid.shape
    dist = np.full((rows, cols), np.inf)
    dist[start] = 0
    heap = []
    heapq.heappush(heap, (0, start))
    parent = {}
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]

    while heap:
        d, (x,y) = heapq.heappop(heap)
        if (x,y) == goal:
            break
        for dx, dy in dirs:
            nx, ny = x+dx, y+dy
            if 0<=nx<rows and 0<=ny<cols and grid[nx,ny]==0:
                if dist[nx,ny] > d + 1:
                    dist[nx,ny] = d+1
                    parent[(nx,ny)] = (x,y)
                    heapq.heappush(heap, (dist[nx,ny], (nx,ny)))
    # 回溯路径
    path = []
    cur = goal
    while cur in parent:
        path.append(cur)
        cur = parent[cur]
    path.append(start)
    return path[::-1]


def a_star(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    g_score = np.full((rows, cols), np.inf)
    g_score[start] = 0
    f_score = np.full((rows, cols), np.inf)
    f_score[start] = np.linalg.norm(np.array(start)-np.array(goal))
    parent = {}
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]

    while open_set:
        _, curr = heapq.heappop(open_set)
        if curr == goal:
            path = []
            while curr in parent:
                path.append(curr)
                curr = parent[curr]
            path.append(start)
            return path[::-1]
        for dx, dy in dirs:
            nx, ny = curr[0]+dx, curr[1]+dy
            if 0<=nx<rows and 0<=ny<cols and grid[nx,ny]==0:
                tentative_g = g_score[curr] + 1
                if tentative_g < g_score[nx,ny]:
                    parent[(nx,ny)] = curr
                    g_score[nx,ny] = tentative_g
                    f_score[nx,ny] = tentative_g + np.linalg.norm(np.array((nx,ny))-np.array(goal))
                    heapq.heappush(open_set, (f_score[nx,ny], (nx,ny)))
    return []


if __name__ == "__main__":
    # 0可行，1障碍物
    grid_map = np.array([
        [0,0,0,0,0],
        [0,1,1,0,0],
        [0,0,0,0,0],
        [0,1,0,1,0],
        [0,0,0,0,0]
    ])
    s = (0,0)
    g = (4,4)
    path_a = a_star(grid_map, s, g)
    print("A*路径：", path_a)
