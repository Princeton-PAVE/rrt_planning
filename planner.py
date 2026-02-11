import numpy as np, math, random
from scipy.spatial import cKDTree
from maze2d import *

class Node:
    def __init__(self, pos, parent=None):
        self.pos = np.array(pos, dtype=np.float32)
        self.parent = parent
        self.cost = 0.0 if parent is None else parent.cost + np.linalg.norm(self.pos - parent.pos)

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
        self.coord_order = coord_order

    def collision_free(self, p1, p2):
        dist = np.linalg.norm(p2 - p1)
        steps = max(2, int(dist / 0.5))  # tune resolution; 0.05 was extremely fine for 100x100
        for t in np.linspace(0, 1, steps):
            p = p1 + t * (p2 - p1)
            if self.env._is_wall(p):
                return False
        return True

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
        c_min = float(np.linalg.norm(self.goal - self.start))
        if c_best <= c_min + 1e-6:
            # Already at (near) straight line optimum; just bias to goal/uniform
            return self.sample_free_uniform()

        # Ellipse parameters (Informed RRT*)
        center = 0.5 * (self.start + self.goal)
        a1 = (self.goal - self.start) / (c_min + 1e-9)               # major axis direction
        C = InformedRRTStar._rotation_to_align_with_a1(a1)                 # rotate unit disk x-axis to a1

        a = c_best / 2.0                                   # semi-major axis length
        b = math.sqrt(max(c_best * c_best - c_min * c_min, 0.0)) / 2.0  # semi-minor axis length

        L = np.array([[a, 0.0],
                      [0.0, b]], dtype=np.float32)

        x_ball = InformedRRTStar._sample_unit_ball_2d()
        x = (C @ (L @ x_ball)) + center
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
        # --- Main loop ---
        nodes = [Node(self.start)]
        positions = np.array([self.start], dtype=np.float32)
        tree = cKDTree(positions)

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
            nearest = nodes[int(idx)]

            direction = rnd - nearest.pos
            dist = float(np.linalg.norm(direction))
            if dist == 0.0:
                continue

            new_pos = nearest.pos + (direction / dist) * min(step_size, dist)

            if self.env._is_wall(new_pos) or not self.collision_free(nearest.pos, new_pos):
                continue

            # Choose best parent among neighbors (RRT* step)
            neighbor_idxs = tree.query_ball_point(new_pos, r=radius)

            best_parent = nearest
            best_parent_cost = nearest.cost + np.linalg.norm(new_pos - nearest.pos)

            for j in neighbor_idxs:
                n = nodes[j]
                if not self.collision_free(n.pos, new_pos):
                    continue
                c = n.cost + np.linalg.norm(new_pos - n.pos)
                if c < best_parent_cost:
                    best_parent = n
                    best_parent_cost = c

            new_node = Node(new_pos, best_parent)
            new_node.cost = float(best_parent_cost)

            # Add node
            nodes.append(new_node)
            positions = np.vstack([positions, new_node.pos])

            # Rewire neighbors through new node
            for j in neighbor_idxs:
                n = nodes[j]
                if n is new_node:
                    continue
                if not self.collision_free(new_node.pos, n.pos):
                    continue
                new_cost = new_node.cost + np.linalg.norm(n.pos - new_node.pos)
                if new_cost < n.cost:
                    n.parent = new_node
                    n.cost = float(new_cost)

            # Rebuild KD-tree
            if (len(nodes) % rebuild_every) == 0:
                tree = cKDTree(positions)

            # Goal check / update best
            if np.linalg.norm(new_node.pos - self.goal) < goal_thresh:
                if new_node.cost < best_cost:
                    best_goal_node = new_node
                    best_cost = float(new_node.cost)

        node_positions = positions

        if best_goal_node is not None:
            path = []
            cur = best_goal_node
            while cur is not None:
                path.append(cur.pos)
                cur = cur.parent
            path = np.array(path[::-1], dtype=np.float32)
            # return path, nodes
            return path, node_positions

        return None, node_positions
