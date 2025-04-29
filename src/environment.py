# src/environment.py
import numpy as np

class Environment:
    def __init__(self, start_state, goal_region, obstacles):
        """
        start_state: [x, y]
        goal_region: [x1, y1, x2, y2, x3, y3, x4, y4]
        obstacles: list of obstacles (each obstacle: [x1, y1, x2, y2, x3, y3, x4, y4])
        """
        self.start_state = np.array(start_state)
        self.goal_region = np.array(goal_region)
        self.goal_x = np.array([goal_region[0], goal_region[2], goal_region[4], goal_region[6]])
        self.goal_y = np.array([goal_region[1], goal_region[3], goal_region[5], goal_region[7]])
        self.obstacles = np.array(obstacles)

    def collision_check_point(self, p):
        """Return True if point p ([x,y]) is in collision with any obstacle."""
        for obs in self.obstacles:
            x1, y1, x2, y2, x3, y3, x4, y4 = obs
            if (p[0] >= x1 and p[0] <= x2) and (p[1] >= y1 and p[1] <= y3):
                return True
        return False

    def collision_check_segment(self, p1, p2):
        """
        Check if the segment from point p1 to p2 collides with any obstacle.
        Returns True if a collision is detected.
        """
        collision_found = False
        a = p2[0] - p1[0]
        b = p2[1] - p1[1]
        m = None if a == 0 else b / a

        for obs in self.obstacles:
            x1, y1, x2, y2, x3, y3, x4, y4 = obs
            if m is not None and a != 0:
                y_at_x1 = m * (x1 - p1[0]) + p1[1]
                t_y_at_x1 = (x1 - p1[0]) / a
                y_at_x2 = m * (x2 - p1[0]) + p1[1]
                t_y_at_x2 = (x2 - p1[0]) / a
                if m != 0:
                    x_at_y1 = (y1 - p1[1]) / m + p1[0]
                    t_x_at_y1 = (y1 - p1[1]) / b if b != 0 else 0
                    x_at_y3 = (y3 - p1[1]) / m + p1[0]
                    t_x_at_y3 = (y3 - p1[1]) / b if b != 0 else 0
                else:
                    x_at_y1 = p1[0]
                    t_x_at_y1 = 0
                    x_at_y3 = p1[0]
                    t_x_at_y3 = 0

                if ((y_at_x1 >= y1 and y_at_x1 <= y3 and 0 <= t_y_at_x1 <= 1) or
                    (y_at_x2 >= y1 and y_at_x2 <= y3 and 0 <= t_y_at_x2 <= 1) or
                    (x_at_y1 >= x1 and x_at_y1 <= x2 and 0 <= t_x_at_y1 <= 1) or
                    (x_at_y3 >= x1 and x_at_y3 <= x2 and 0 <= t_x_at_y3 <= 1)):
                    collision_found = True
                    break
            else:
                # For vertical segments:
                if p1[0] >= x1 and p1[0] <= x2:
                    ymin = min(p1[1], p2[1])
                    ymax = max(p1[1], p2[1])
                    if ymax >= y1 and ymin <= y3:
                        collision_found = True
                        break
        return collision_found