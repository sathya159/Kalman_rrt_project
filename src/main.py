# src/main.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime
from environment import Environment
from rrt_planner import RRTPlanner
from kalman_filter import KalmanFilterRRT
from plotting import plot_data, save_figure_with_timestamp

print("Current working directory:", os.getcwd())
sys.stdout.flush()

def main():
    # Define environment parameters
    start_state = [5, 50]
    goal_region = [90, 0, 100, 0, 100, 100, 90, 100]
    obstacles = [
        [5, 10, 15, 10, 15, 20, 5, 20],
        [10, 40, 20, 40, 20, 50, 10, 50],
        [20, 70, 30, 70, 30, 80, 20, 80],
        [30, 20, 40, 20, 40, 30, 30, 30],
        [40, 50, 50, 50, 50, 60, 40, 60],
        [50, 5, 60, 5, 60, 15, 50, 15],
        [55, 80, 65, 80, 65, 90, 55, 90],
        [60, 40, 70, 40, 70, 50, 60, 50],
        [70, 20, 80, 20, 80, 30, 70, 30],
        [75, 65, 85, 65, 85, 75, 75, 75]
    ]
    
    # Create environment, planner, Kalman filter
    env = Environment(start_state, goal_region, obstacles)
    planner = RRTPlanner(env)
    kf = KalmanFilterRRT(env)
    
    total_trials = 100
    min_path_length = float('inf')
    min_uncertainty = float('inf')
    max_uncertainty = 0
    
    shortest_path = None
    min_uncertainty_path = None
    max_uncertainty_path = None

    # Create unique folder for this run
    run_folder = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_folder = 'plots'
    save_folder = os.path.join(base_folder, run_folder)
    os.makedirs(save_folder, exist_ok=True)
    
    # Run RRT trials
    for trial in range(total_trials):
        path, tree_length = planner.plan(start_state, goal_x_threshold=90)
        if path is None:
            continue
        Pk = kf.run(path, draw_ellipse=False)
        if tree_length < min_path_length:
            min_path_length = tree_length
            shortest_path = {'path': path, 'path_length': tree_length, 'Pk': Pk}
        area_of_ellipse = np.pi * Pk[0, 0] * Pk[1, 1]
        if area_of_ellipse < min_uncertainty:
            min_uncertainty = area_of_ellipse
            min_uncertainty_path = {'path': path, 'path_length': tree_length, 'Pk': Pk}
        if area_of_ellipse > max_uncertainty:
            max_uncertainty = area_of_ellipse
            max_uncertainty_path = {'path': path, 'path_length': tree_length, 'Pk': Pk}

    # --- Save 4 plots ---
    
    # 1. Shortest Path (normal)
    fig_short, ax_short = plot_data(env, shortest_path['path'], "Shortest Path", show=False)
    save_figure_with_timestamp(fig_short, folder=save_folder, prefix='shortest_path')

    # 2. Shortest Path (with ellipses)
    fig_short_ellipses, ax_short_ellipses = plot_data(env, shortest_path['path'], "Shortest Path (With Ellipses)", show=False)
    kf.run(shortest_path['path'], draw_ellipse=True, ax=ax_short_ellipses)
    plt.draw()
    save_figure_with_timestamp(fig_short_ellipses, folder=save_folder, prefix='shortest_path_with_ellipses')

    # 3. Min Uncertainty Path (with ellipses)
    fig_min_ellipses, ax_min_ellipses = plot_data(env, min_uncertainty_path['path'], "Minimum Uncertainty Path (With Ellipses)", show=False)
    kf.run(min_uncertainty_path['path'], draw_ellipse=True, ax=ax_min_ellipses)
    plt.draw()
    save_figure_with_timestamp(fig_min_ellipses, folder=save_folder, prefix='min_uncertainty_path_with_ellipses')

    # 4. Max Uncertainty Path (with ellipses)
    fig_max_ellipses, ax_max_ellipses = plot_data(env, max_uncertainty_path['path'], "Maximum Uncertainty Path (With Ellipses)", show=False)
    kf.run(max_uncertainty_path['path'], draw_ellipse=True, ax=ax_max_ellipses)
    plt.draw()
    save_figure_with_timestamp(fig_max_ellipses, folder=save_folder, prefix='max_uncertainty_path_with_ellipses')

    print(f"All plots saved successfully inside: {save_folder}")
    sys.stdout.flush()

if __name__ == '__main__':
    main()