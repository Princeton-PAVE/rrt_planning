"""
#RRT* implementation
#RRT* works better than A* in this continous setting in my opinion
#Even if I tried to make A* more fine grained, it would've still followed coarse
#horizontal/vertical plans ... I think...?
import numpy as np, math, random
import time
from scipy.spatial import cKDTree

class Node:
    def __init__(self, pos, parent=None):
        self.pos = np.array(pos, dtype=np.float32)
        self.parent = parent #for backtracking to get path
        self.cost = 0.0 if parent is None else parent.cost + np.linalg.norm(pos - parent.pos)



def rrt_star(env, start, goal, step_size=2.0, radius=10.0, max_iter=5000, goal_thresh=0.5, rebuild_every=1):
    
    def collision_free(p1, p2):
        dist = np.linalg.norm(p2 - p1)
        steps = max(2, int(dist / 0.05))
        for t in np.linspace(0, 1, steps):
            p = p1 + t * (p2 - p1)
            if env._is_wall(p):
                return False
        return True

    nodes = [Node(np.array(start, dtype=np.float32))]
    positions = np.array([nodes[0].pos], dtype=np.float32)

    tree = cKDTree(positions)  # initial tree

    best_goal_node = None
    best_cost = float("inf")

    for it in range(max_iter):
        # sample
        if random.random() < 0.1:
            rnd = np.array(goal, dtype=np.float32)
        else:
            rnd = np.array([random.uniform(0, env.w), random.uniform(0, env.h)], dtype=np.float32)
            if env._is_wall(rnd):
                continue

        # nearest via KD-tree
        _, idx = tree.query(rnd, k=1)
        nearest = nodes[int(idx)]

        direction = rnd - nearest.pos
        dist = np.linalg.norm(direction)
        if dist == 0:
            continue

        new_pos = nearest.pos + (direction / dist) * min(step_size, dist)

        if env._is_wall(new_pos) or not collision_free(nearest.pos, new_pos):
            continue

        new_node = Node(new_pos, nearest)
        new_node.cost = nearest.cost + np.linalg.norm(new_pos - nearest.pos)

        # radius neighbors via KD-tree
        neighbor_idxs = tree.query_ball_point(new_pos, r=radius)

        # rewire (only over neighbors, not all nodes)
        for j in neighbor_idxs:
            n = nodes[j]
            if n is nearest:
                continue
            if not collision_free(n.pos, new_pos):
                continue
            new_cost = new_node.cost + np.linalg.norm(n.pos - new_pos)
            if new_cost < n.cost:
                n.parent = new_node
                n.cost = new_cost

        # add node
        nodes.append(new_node)
        positions = np.vstack([positions, new_node.pos])

        # rebuild KD-tree occasionally (amortized fast)
        if (len(nodes) % rebuild_every) == 0:
            tree = cKDTree(positions)

        # goal check
        if np.linalg.norm(new_node.pos - goal) < goal_thresh:
            if new_node.cost < best_cost:
                best_goal_node = new_node
                best_cost = new_node.cost

    node_positions = positions

    if best_goal_node is not None:
        path = []
        cur = best_goal_node
        while cur is not None:
            path.append(cur.pos)
            cur = cur.parent
        path = np.array(path[::-1], dtype=np.float32)
        return path, node_positions

    return None, node_positions
"""
import numpy as np, math, random
from scipy.spatial import cKDTree

class Node:
    def __init__(self, pos, parent=None):
        self.pos = np.array(pos, dtype=np.float32)
        self.parent = parent
        self.cost = 0.0 if parent is None else parent.cost + np.linalg.norm(self.pos - parent.pos)

def rrt_star(
    env, start, goal,
    step_size=2.0, radius=10.0, max_iter=5000, goal_thresh=0.5,
    rebuild_every=1, goal_sample_rate=0.1,
    coord_order="rc",  # "rc" means (row, col) = (y,x). If you use (x,y), set "xy".
):
    """
    Informed RRT*: after first solution, sample uniformly from the prolate hyperspheroid
    (ellipse in 2D) that can still improve the best path cost.

    Assumes Euclidean path length cost and free space in [0,h) x [0,w).

    coord_order:
      - "rc": points are (row, col) and env.h, env.w correspond to (rows, cols)  ✅ for your Maze2DEnv
      - "xy": points are (x, y)
    """

    start = np.array(start, dtype=np.float32)
    goal  = np.array(goal,  dtype=np.float32)

    def collision_free(p1, p2):
        dist = np.linalg.norm(p2 - p1)
        steps = max(2, int(dist / 0.5))  # tune resolution; 0.05 was extremely fine for 100x100
        for t in np.linspace(0, 1, steps):
            p = p1 + t * (p2 - p1)
            if env._is_wall(p):
                return False
        return True

    # --- Informed sampling helpers (2D) ---
    def _sample_unit_ball_2d():
        # Uniform in unit disk: radius sqrt(u), angle 2πv
        u = random.random()
        v = random.random()
        r = math.sqrt(u)
        th = 2.0 * math.pi * v
        return np.array([r * math.cos(th), r * math.sin(th)], dtype=np.float32)

    def _rotation_to_align_with_a1(a1):
        # Rotate x-axis to point along a1 (2D)
        theta = math.atan2(float(a1[1]), float(a1[0]))
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, -s],
                         [s,  c]], dtype=np.float32)

    def sample_free_uniform():
        # Sample in correct coordinate convention
        if coord_order == "rc":
            # (row, col) in [0,h) x [0,w)
            return np.array([random.uniform(0, env.h), random.uniform(0, env.w)], dtype=np.float32)
        else:
            # (x, y) in [0,w) x [0,h)
            return np.array([random.uniform(0, env.w), random.uniform(0, env.h)], dtype=np.float32)

    def sample_informed(c_best):
        # If no solution yet, fall back to uniform sampling
        if not np.isfinite(c_best):
            return sample_free_uniform()

        # Lower bound straight-line distance between foci
        c_min = float(np.linalg.norm(goal - start))
        if c_best <= c_min + 1e-6:
            # Already at (near) straight line optimum; just bias to goal/uniform
            return sample_free_uniform()

        # Ellipse parameters (Informed RRT*)
        center = 0.5 * (start + goal)
        a1 = (goal - start) / (c_min + 1e-9)               # major axis direction
        C = _rotation_to_align_with_a1(a1)                 # rotate unit disk x-axis to a1

        a = c_best / 2.0                                   # semi-major axis length
        b = math.sqrt(max(c_best * c_best - c_min * c_min, 0.0)) / 2.0  # semi-minor axis length

        L = np.array([[a, 0.0],
                      [0.0, b]], dtype=np.float32)

        x_ball = _sample_unit_ball_2d()
        x = (C @ (L @ x_ball)) + center
        return x.astype(np.float32)

    # --- Main loop ---
    nodes = [Node(start)]
    positions = np.array([start], dtype=np.float32)
    tree = cKDTree(positions)

    best_goal_node = None
    best_cost = float("inf")

    for it in range(max_iter):
        # Sample (goal bias + informed after first solution)
        if random.random() < goal_sample_rate:
            rnd = goal.copy()
        else:
            rnd = sample_informed(best_cost)
            # reject samples in walls / out of bounds
            if env._is_wall(rnd):
                continue

        # Nearest
        _, idx = tree.query(rnd, k=1)
        nearest = nodes[int(idx)]

        direction = rnd - nearest.pos
        dist = float(np.linalg.norm(direction))
        if dist == 0.0:
            continue

        new_pos = nearest.pos + (direction / dist) * min(step_size, dist)

        if env._is_wall(new_pos) or not collision_free(nearest.pos, new_pos):
            continue

        # Choose best parent among neighbors (RRT* step)
        neighbor_idxs = tree.query_ball_point(new_pos, r=radius)

        best_parent = nearest
        best_parent_cost = nearest.cost + np.linalg.norm(new_pos - nearest.pos)

        for j in neighbor_idxs:
            n = nodes[j]
            if not collision_free(n.pos, new_pos):
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
            if not collision_free(new_node.pos, n.pos):
                continue
            new_cost = new_node.cost + np.linalg.norm(n.pos - new_node.pos)
            if new_cost < n.cost:
                n.parent = new_node
                n.cost = float(new_cost)

        # Rebuild KD-tree
        if (len(nodes) % rebuild_every) == 0:
            tree = cKDTree(positions)

        # Goal check / update best
        if np.linalg.norm(new_node.pos - goal) < goal_thresh:
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
        return path, node_positions

    return None, node_positions
