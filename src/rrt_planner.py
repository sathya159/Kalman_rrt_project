# src/rrt_planner.py
import numpy as np

class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent

class RRTPlanner:
    def __init__(self, env, max_iter=1000, epsilon=2, map_size=(100, 100)):
        self.env = env
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.map_size = map_size

    def plan(self, start, goal_x_threshold):
        """
        Build the RRT and return a path (as an array of [x,y] points)
        when a node with x > goal_x_threshold is reached.
        """
        tree = [Node(start[0], start[1], parent=None)]
        for i in range(self.max_iter):
            rand_point = np.array([
                np.random.uniform(0, self.map_size[0]),
                np.random.uniform(0, self.map_size[1])
            ])
            nearest = min(tree, key=lambda node: np.linalg.norm(np.array([node.x, node.y]) - rand_point))
            direction = rand_point - np.array([nearest.x, nearest.y])
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            direction = direction / norm
            new_point = np.array([nearest.x, nearest.y]) + self.epsilon * direction
            if self.env.collision_check_point(new_point):
                continue
            if self.env.collision_check_segment(np.array([nearest.x, nearest.y]), new_point):
                continue
            new_node = Node(new_point[0], new_point[1], parent=nearest)
            tree.append(new_node)
            if new_node.x > goal_x_threshold:
                return self.reconstruct_path(new_node), len(tree)
        print("Maximum iterations reached without reaching the goal!")
        return None, None

    def reconstruct_path(self, goal_node):
        """Reconstruct the path from the start to goal node."""
        path = []
        current = goal_node
        while current is not None:
            path.append([current.x, current.y])
            current = current.parent
        return np.array(path[::-1])