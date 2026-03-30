import math
import numpy as np, random
from scipy.spatial import cKDTree
from maze2d import *

class InformedRRTStar:
    def __init__(
        self,
        env: Maze2DEnv, 
        start: tuple, 
        goal: tuple,
        smallest_obs_width: float = 1.0,
        coord_order: str="rc",  # "rc" means (row, col) = (y,x). If you use (x,y), set "xy".
    ) -> None:

        self.env = env
        self.start = np.array(start, dtype=np.float32)
        self.goal  = np.array(goal,  dtype=np.float32)
        self.coord_order = coord_order
        
        # precompute collision linspace
        self.collision_step_size = smallest_obs_width
        max_steps = int(self.env.maze.shape[0] / self.collision_step_size)
        self.max_ts = np.linspace(0, 1, max_steps).reshape(-1, 1)
        
        # precompute ellipse
        self.c_min = float(self.dist2d(self.goal, self.start))
        self.ellipse_center = 0.5 * (self.start + self.goal)
        a1 = (self.goal - self.start) / (self.c_min + 1e-9)  
        self.C = InformedRRTStar._rotation_to_align_with_a1(a1) # rotate unit disk x-axis to a1
        
        # store most recent calculate_path runs
        self.last_positions = None
        self.last_parents = None
        self.last_costs = None
        
    
    @staticmethod
    def _sample_unit_ball_2d() -> np.ndarray:
        # Uniform in unit disk: radius sqrt(u), angle 2πv
        u = random.random()
        v = random.random()
        r = math.sqrt(u)
        th = 2.0 * math.pi * v
        return np.array([r * math.cos(th), r * math.sin(th)], dtype=np.float32)

    @staticmethod
    def _rotation_to_align_with_a1(a1: np.ndarray) -> np.ndarray:
        # Rotate x-axis to point along a1 (2D)
        theta = math.atan2(float(a1[1]), float(a1[0]))
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, -s],
                         [s,  c]], dtype=np.float32)

    @staticmethod
    def dist2d(a: np.ndarray, b: np.ndarray) -> np.float32:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.sqrt(dx*dx + dy*dy)

    def collision_free(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.sqrt(dx*dx + dy*dy)
        steps = max(2, int(dist / self.collision_step_size))
        
        h, w = self.env.h, self.env.w
        maze = self.env.maze
        
        for i in range(steps + 1):
            t = i / steps
            r = p1[0] + t * dx
            c = p1[1] + t * dy
            if r < 0 or c < 0 or r >= h or c >= w:
                return False
            if maze[int(r), int(c)] > 0:
                return False
        return True
        
    # faster than the above in >~50 nodes between p1 and p2
    def collision_free_vec(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        dist = self.dist2d(p1, p2)
        steps = max(2, int(dist / self.collision_step_size))
        ts = self.max_ts[:steps]
        
        points = p1 + ts * (p2 - p1)  # shape (steps, 2)
        rows = points[:, 0]
        cols = points[:, 1]
        
        # Bounds check
        if np.any((rows < 0) | (cols < 0) | (rows >= self.env.h) | (cols >= self.env.w)):
            return False
        
        r = np.floor(rows).astype(int)
        c = np.floor(cols).astype(int)
        
        return not np.any(self.env.maze[r, c] > 0)   

    def sample_free_uniform(self) -> np.ndarray:
        # Sample in correct coordinate convention
        if self.coord_order == "rc":
            # (row, col) in [0,h) x [0,w)
            return np.array([random.uniform(0, self.env.h), random.uniform(0, self.env.w)], dtype=np.float32)
        else:
            # (x, y) in [0,w) x [0,h)
            return np.array([random.uniform(0, self.env.w), random.uniform(0, self.env.h)], dtype=np.float32)

    def sample_informed(self, c_best: float) -> np.ndarray:
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
        max_step_size: float=2.0,
        radius: float=10.0, 
        max_iter: int=5000, 
        goal_thresh: float=0.5,
        rebuild_every: int=1, 
        goal_sample_rate: float=0.1,
        path_only: bool = False,
    ) -> tuple:
        """_summary_

        Args:
            step_size (float, optional): _description_. Defaults to 2.0.
            radius (float, optional): _description_. Defaults to 10.0.
            max_iter (int, optional): _description_. Defaults to 5000.
            goal_thresh (float, optional): _description_. Defaults to 0.5.
            rebuild_every (int, optional): _description_. Defaults to 1.
            goal_sample_rate (float, optional): _description_. Defaults to 0.1.

        Returns: the path
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
            dist = self.dist2d(rnd, nearest)
            if dist <= 1e-6:
                continue

            new_pos = nearest + (direction / dist) * min(max_step_size, dist)

            if self.env._is_wall(new_pos) or not self.collision_free(nearest, new_pos):
                continue

            # Choose best parent among neighbors (RRT* step)
            # the best parent may have changed when capping the step size
            neighbor_idxs = tree.query_ball_point(new_pos, r=radius)

            best_parent = idx
            best_parent_cost = costs[idx] + self.dist2d(new_pos, nearest)

            for j in neighbor_idxs:
                if not self.collision_free(positions[j], new_pos):
                    continue
                c = costs[j] + self.dist2d(new_pos, positions[j])
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
                
                new_cost = costs[cur_node_idx] + self.dist2d(positions[j], new_pos)
                if new_cost < costs[j]:
                    parents[j] = cur_node_idx
                    costs[j] = new_cost

            # Rebuild KD-tree
            if (cur_node_idx % rebuild_every) == 0:
                tree = cKDTree(positions[:cur_node_idx])

            # Goal check / update best
            if self.dist2d(positions[cur_node_idx], self.goal) < goal_thresh:
                if costs[cur_node_idx] < best_cost:
                    best_goal_node = cur_node_idx
                    best_cost = costs[cur_node_idx]
                    
            cur_node_idx += 1
        
        # store metadata
        if not path_only:
            self.last_positions = positions[:cur_node_idx]
            self.last_parents = parents
            self.last_costs = costs
        
        # create and return path if it's found
        if best_goal_node is not None:
            path = []
            cur_node = best_goal_node
            while cur_node >= 0:
                path.append(positions[cur_node])
                cur_node = parents[cur_node]
            path = np.array(path[::-1], dtype=np.float32)
            return path

        return None
