# src/plotting.py
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

def save_figure_with_timestamp(fig, folder='plots', prefix='plot'):
    """
    Save the figure into the given folder without adding timestamp to filename.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filename = f"{prefix}.png"  # No timestamp in filename
    filepath = os.path.join(folder, filename)
    fig.savefig(filepath)
    print(f"Plot saved at: {filepath}")

def plot_data(env, path, figure_title, show=True):
    """
    Create and return a plot of the environment with obstacles, start, goal, and path.
    """
    fig, ax = plt.subplots()
    ax.set_title(figure_title)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')

    # Plot start state
    ax.plot(env.start_state[0], env.start_state[1], 'r.', markersize=20)

    # Plot goal region
    goal_coords = np.column_stack((env.goal_x, env.goal_y))
    goal_poly = Polygon(goal_coords, color='green', alpha=0.5)
    ax.add_patch(goal_poly)

    # Plot obstacles
    for obs in env.obstacles:
        obs_coords = np.array([
            [obs[0], obs[1]],
            [obs[2], obs[3]],
            [obs[4], obs[5]],
            [obs[6], obs[7]]
        ])
        obs_poly = Polygon(obs_coords, color='blue', alpha=0.5)
        ax.add_patch(obs_poly)

    # Plot the path
    if path is not None and len(path) > 0:
        ax.plot(path[:, 0], path[:, 1], '-o', color='black')

    if show:
        plt.show()

    return fig, ax