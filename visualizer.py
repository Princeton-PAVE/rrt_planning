import cv2
import numpy as np
from scipy.spatial import cKDTree

WINDOW_NAME = "bev"

class BevEnv:
    def __init__(self, occ):
        self.occ = occ
        self.h, self.w = occ.shape
    def _is_wall(self, p):
        r = int(round(p[0])); c = int(round(p[1]))
        if r < 0 or r >= self.h or c < 0 or c >= self.w:
            return True
        return self.occ[r, c] == 0

        
        
def init_visualizer():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

def visualize_plan(
    maze_uint8: np.ndarray,     
    start_rc, goal_rc,           
    nodes_rc=None,             
    path_rc=None,             
    scale=1,
    node_stride=2,           
    node_radius=2,          
    edge_stride=3,          
    max_edge_len=25.0,           
    branch_alpha=0.75, 
    vis_controls=None          
):
    maze = maze_uint8.astype(np.uint8)
    if maze.max() <= 1:
        maze = (maze * 255).astype(np.uint8)

    base = cv2.cvtColor(maze, cv2.COLOR_GRAY2BGR)
    H, W = base.shape[:2]

    def rc_to_xy(rc):
        return int(round(rc[1])), int(round(rc[0]))  # (x,y)

    overlay = base.copy()


    #check this
    if vis_controls is not None and len(vis_controls) > 1:
        p = np.asarray(vis_controls, dtype=np.float32)
        poly = np.stack([p[:, 1], p[:, 0]], axis=1)
        poly = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [poly], False, (0, 255, 0), thickness=4, lineType=cv2.LINE_AA)
        
        
    if nodes_rc is not None and len(nodes_rc) > 1:
        pts = np.asarray(nodes_rc, dtype=np.float32)

        # optional: subsample for speed
        pts_draw = pts[::node_stride]

        # 1) Prominent nodes (thicker dots)
        for r, c in pts_draw:
            x, y = int(round(c)), int(round(r))
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(overlay, (x, y), node_radius, (255, 0, 255), -1, lineType=cv2.LINE_AA)


        pts_edge = pts[::edge_stride]
        tree = cKDTree(pts)  # all nodes
        for p in pts_edge:
            # query 2 because nearest is itself
            dists, idxs = tree.query(p, k=2)
            if np.isscalar(dists) or len(dists) < 2:
                continue
            d = float(dists[1])
            if d > max_edge_len:
                continue
            q = pts[int(idxs[1])]

            x1, y1 = int(round(p[1])), int(round(p[0]))
            x2, y2 = int(round(q[1])), int(round(q[0]))
            if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 255), 2, lineType=cv2.LINE_AA)

   
    out = cv2.addWeighted(overlay, branch_alpha, base, 1.0 - branch_alpha, 0.0)

    if path_rc is not None and len(path_rc) > 1:
        p = np.asarray(path_rc, dtype=np.float32)
        poly = np.stack([p[:, 1], p[:, 0]], axis=1)
        poly = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [poly], False, (0, 255, 0), thickness=4, lineType=cv2.LINE_AA)

 
    sx, sy = rc_to_xy(start_rc)
    gx, gy = rc_to_xy(goal_rc)
    cv2.circle(out, (sx, sy), 7, (255, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.circle(out, (gx, gy), 7, (0, 255, 255), -1, lineType=cv2.LINE_AA)

    if scale != 1:
        out = cv2.resize(out, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)

    return out