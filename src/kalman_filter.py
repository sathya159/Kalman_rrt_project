# src/kalman_filter.py
import numpy as np
from matplotlib.patches import Ellipse

class KalmanFilterRRT:
    def __init__(self, env, sensing_range=5, delta_t=1):
        self.env = env
        self.sensing_range = sensing_range
        self.delta_t = delta_t
        self.A = np.array([[1, 0, delta_t, 0],
                           [0, 1, 0, delta_t],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.Q = np.eye(4)

    def run(self, path, draw_ellipse=False, ax=None):
        """
        Run the Kalman filter along the given path.
        If draw_ellipse is True and an axis (ax) is provided, draw uncertainty ellipses.
        Returns the last covariance matrix Pk.
        """
        Pk_prev = np.zeros((4, 4))
        Pk = None
        for point in path:
            currentX, currentY = point
            # Simulate availability of lidar measurements along x and y directions
            lidar_x = self.env.collision_check_segment(
                np.array([currentX, currentY]),
                np.array([currentX + self.sensing_range, currentY])
            ) or self.env.collision_check_segment(
                np.array([currentX, currentY]),
                np.array([currentX - self.sensing_range, currentY])
            )
            lidar_y = self.env.collision_check_segment(
                np.array([currentX, currentY]),
                np.array([currentX, currentY + self.sensing_range])
            ) or self.env.collision_check_segment(
                np.array([currentX, currentY]),
                np.array([currentX, currentY - self.sensing_range])
            )

            # Build measurement matrix H
            H = np.array([[0, 0, 1, 0],
                          [0, 0, 0, 1]])
            measurement_rows = 2
            if lidar_x:
                H = np.vstack([H, np.array([1, 0, 0, 0])])
                measurement_rows += 1
            if lidar_y:
                H = np.vstack([H, np.array([0, 1, 0, 0])])
                measurement_rows += 1
            R = np.eye(measurement_rows)

            try:
                inv_term = np.linalg.inv(self.A @ Pk_prev @ self.A.T + self.Q)
                temp = H.T @ np.linalg.inv(R) @ H
                Pk = np.linalg.inv(inv_term + temp)
            except np.linalg.LinAlgError:
                Pk = np.eye(4)

            if draw_ellipse and ax is not None:
                width = 2 * np.sqrt(Pk[0, 0])
                height = 2 * np.sqrt(Pk[1, 1])
                ell = Ellipse((currentX, currentY), width, height, angle=0,
                              edgecolor='red', facecolor='none')
                ax.add_patch(ell)
            Pk_prev = Pk

        return Pk