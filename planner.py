import numpy as np, math, random
from scipy.spatial import cKDTree
from maze2d import *

# class Node:
#     def __init__(self, pos, parent=None):
#         self.pos = np.array(pos, dtype=np.float32)
#         self.parent = parent
#         self.cost = 0.0 if parent is None else parent.cost + np.linalg.norm(self.pos - parent.pos)

class InformedRRTStar:
    def __init__(
        self,
        env: Maze2DEnv, 
        start: tuple, 
        goal: tuple,
        coord_order: str="rc",  # "rc" means (row, col) = (y,x). If you use (x,y), set "xy".
    ) -> None:
        """
        Informed RRT*: after first solution, sample uniformly from the prolate hyperspheroid
        (ellipse in 2D) that can still improve the best path cost.

        Assumes Euclidean path length cost and free space in [0,h) x [0,w).

        coord_order:
        - "rc": points are (row, col) and env.h, env.w correspond to (rows, cols)  ✅ for your Maze2DEnv
        - "xy": points are (x, y)
        
        :param env: Obstacle ccupancy grid
        :type env: Maze2DEnv
        :param start: path start
        :type start: tuple
        :param goal: path end
        :type goal: tuple
        :param step_size: maximum size to step 
        :type step_size: float
        :param radius: kdtree radius?
        :type radius: float
        :param max_iter: total iterations to find and refine path
        :type max_iter: int
        :param goal_thresh: distance from goal to consider having reached it
        :type goal_thresh: float
        :param rebuild_every: num iters between rewiring all nodes
        :type rebuild_every: int
        :param goal_sample_rate: rate to try building path straight to goal
        :type goal_sample_rate: float
        :param coord_order: coordinate order
        :type coord_order: str
        :return: tuple of nodes lowk could just be one
        :rtype: tuple
        
        """

        self.env = env
        self.start = np.array(start, dtype=np.float32)
        self.goal  = np.array(goal,  dtype=np.float32)
        self.collision_step_size = 0.5 # non-proportional units?
        self.coord_order = coord_order
        
        # precompute ellipse
        self.c_min = float(np.linalg.norm(self.goal - self.start))
        self.ellipse_center = 0.5 * (self.start + self.goal)
        a1 = (self.goal - self.start) / (self.c_min + 1e-9)  
        self.C = InformedRRTStar._rotation_to_align_with_a1(a1) # rotate unit disk x-axis to a1
        

    def collision_free(self, p1, p2):
        dist = np.linalg.norm(p2 - p1)
        steps = max(2, int(dist / self.collision_step_size))
        
        ts = np.linspace(0, 1, steps).reshape(-1, 1)
        points = p1 + ts * (p2 - p1)  # shape (steps, 2)
        
        rows = points[:, 0]
        cols = points[:, 1]
        
        # Bounds check
        if np.any((rows < 0) | (cols < 0) | (rows >= self.env.h) | (cols >= self.env.w)):
            return False
        
        r = np.floor(rows).astype(int)
        c = np.floor(cols).astype(int)
        
        return not np.any(self.env.maze[r, c] > 0)   
    

    # --- Informed sampling helpers (2D) ---
    @staticmethod
    def _sample_unit_ball_2d() -> np.ndarray:
        # Uniform in unit disk: radius sqrt(u), angle 2πv
        u = random.random()
        v = random.random()
        r = math.sqrt(u)
        th = 2.0 * math.pi * v
        return np.array([r * math.cos(th), r * math.sin(th)], dtype=np.float32)

    @staticmethod
    def _rotation_to_align_with_a1(a1):
        # Rotate x-axis to point along a1 (2D)
        theta = math.atan2(float(a1[1]), float(a1[0]))
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, -s],
                         [s,  c]], dtype=np.float32)


    def sample_free_uniform(self):
        # Sample in correct coordinate convention
        if self.coord_order == "rc":
            # (row, col) in [0,h) x [0,w)
            return np.array([random.uniform(0, self.env.h), random.uniform(0, self.env.w)], dtype=np.float32)
        else:
            # (x, y) in [0,w) x [0,h)
            return np.array([random.uniform(0, self.env.w), random.uniform(0, self.env.h)], dtype=np.float32)

    def sample_informed(self, c_best):
        # If no solution yet, fall back to uniform sampling
        if not np.isfinite(c_best):
            return self.sample_free_uniform()

        # Lower bound straight-line distance between foci
        if c_best <= self.c_min + 1e-6:
            # Already at (near) straight line optimum; just bias to goal/uniform
            return self.sample_free_uniform()

        # Ellipse parameters (Informed RRT*)
        a = c_best / 2.0                                   # semi-major axis length
        b = math.sqrt(max(c_best * c_best - self.c_min * self.c_min, 0.0)) / 2.0  # semi-minor axis length

        L = np.array([[a, 0.0],
                      [0.0, b]], dtype=np.float32)

        x_ball = InformedRRTStar._sample_unit_ball_2d()
        x = (self.C @ (L @ x_ball)) + self.ellipse_center
        return x.astype(np.float32)

    def calculate_path(
        self,
        step_size: float=2.0, 
        radius: float=10.0, 
        max_iter: int=5000, 
        goal_thresh: float=0.5,
        rebuild_every: int=1, 
        goal_sample_rate: float=0.1,
    ) -> tuple:
        """_summary_

        Args:
            step_size (float, optional): _description_. Defaults to 2.0.
            radius (float, optional): _description_. Defaults to 10.0.
            max_iter (int, optional): _description_. Defaults to 5000.
            goal_thresh (float, optional): _description_. Defaults to 0.5.
            rebuild_every (int, optional): _description_. Defaults to 1.
            goal_sample_rate (float, optional): _description_. Defaults to 0.1.

        Returns:
            tuple: (path, all_sampled), where path is an ndarray of each
            point in the final path and all_sampled is all points 
            generated and tested by the algorithm. Note all_sampled
            is not necessary for the algorithm, it's only for 
            visualization.
        """
        
        positions = np.empty((max_iter + 1, 2), dtype=np.float32)
        parents = np.empty((max_iter + 1), dtype=np.int32)
        costs = np.empty((max_iter + 1), dtype=np.float32)
        
        positions[0] = self.start
        parents[0] = -1 # -1 indicates root
        costs[0] = 0        
        
        cur_node_idx = 1
        
        tree = cKDTree(positions[:cur_node_idx])

        best_goal_node = None
        best_cost = float("inf")

        for it in range(max_iter):
            # Sample (goal bias + informed after first solution)
            if random.random() < goal_sample_rate:
                rnd = self.goal.copy()
            else:
                rnd = self.sample_informed(best_cost)
                # reject samples in walls / out of bounds
                if self.env._is_wall(rnd):
                    continue

            # Nearest
            _, idx = tree.query(rnd, k=1)
            nearest = positions[idx]

            direction = rnd - nearest
            dist = np.linalg.norm(direction)
            if dist <= 1e-6:
                continue

            new_pos = nearest + (direction / dist) * min(step_size, dist)

            if self.env._is_wall(new_pos) or not self.collision_free(nearest, new_pos):
                continue

            # Choose best parent among neighbors (RRT* step)
            # the best parent may have changed when capping the step size
            neighbor_idxs = tree.query_ball_point(new_pos, r=radius) # TODO decide radius ------------------------------------------

            best_parent = idx
            best_parent_cost = costs[idx] + np.linalg.norm(new_pos - nearest)

            for j in neighbor_idxs:
                if not self.collision_free(positions[j], new_pos):
                    continue
                c = costs[j] + np.linalg.norm(new_pos - positions[j])
                if c < best_parent_cost:
                    best_parent = j
                    best_parent_cost = c

            # Add node
            positions[cur_node_idx] = new_pos
            parents[cur_node_idx] = best_parent
            costs[cur_node_idx] = best_parent_cost

            # Rewire neighbors through new node
            for j in neighbor_idxs:
                if j == cur_node_idx:
                    continue
                
                if not self.collision_free(positions[cur_node_idx], positions[j]):
                    continue
                
                dx = positions[j, 0] - new_pos[0]
                dy = positions[j, 1] - new_pos[1]
                d = math.sqrt(dx*dx + dy*dy)
                
                new_cost = costs[cur_node_idx] + d
                if new_cost < costs[j]:
                    parents[j] = cur_node_idx
                    costs[j] = new_cost

            # Rebuild KD-tree
            if (cur_node_idx % rebuild_every) == 0:
                tree = cKDTree(positions[:cur_node_idx])

            # Goal check / update best
            if np.linalg.norm(positions[cur_node_idx] - self.goal) < goal_thresh:
                if costs[cur_node_idx] < best_cost:
                    best_goal_node = cur_node_idx
                    best_cost = costs[cur_node_idx]
                    
            cur_node_idx += 1

        node_positions = positions[:cur_node_idx]

        # return nodes
        if best_goal_node is not None:
            path = []
            cur_node = best_goal_node
            while cur_node >= 0:
                path.append(positions[cur_node])
                cur_node = parents[cur_node]
            path = np.array(path[::-1], dtype=np.float32)
            return path, node_positions

        return None, node_positions
