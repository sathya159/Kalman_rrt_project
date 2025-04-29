# clear_cache.py
import os
import shutil

def clear_pycache(folder_path='/Users/sathyaprasadreddypatlolla/Downloads/kalman_rrt_project/src/__pycache__'):
    """
    Recursively delete all __pycache__ folders.
    """
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                pycache_path = os.path.join(root, dir_name)
                print(f"Deleting {pycache_path}")
                shutil.rmtree(pycache_path)

if __name__ == "__main__":
    clear_pycache()