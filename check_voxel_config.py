# check_voxel_config.py
import numpy as np
import pickle

# Load a sample point cloud
pts = pickle.load(open('data/pandaset/005/lidar/07.pkl', 'rb'))
if hasattr(pts, 'values'):  # DataFrame
    pts = pts[pts['d'] == 1]  # Front LiDAR only
    pts = pts[['x', 'y', 'z', 'i']].values

print(f"Point cloud shape: {pts.shape}")
print(f"Point cloud range:")
print(f"  X: [{pts[:, 0].min():.2f}, {pts[:, 0].max():.2f}]")
print(f"  Y: [{pts[:, 1].min():.2f}, {pts[:, 1].max():.2f}]")
print(f"  Z: [{pts[:, 2].min():.2f}, {pts[:, 2].max():.2f}]")

# Check voxel config
pc_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
voxel_size = [0.075, 0.075, 0.2]

grid_size = [
    int((pc_range[3] - pc_range[0]) / voxel_size[0]),
    int((pc_range[4] - pc_range[1]) / voxel_size[1]),
    int((pc_range[5] - pc_range[2]) / voxel_size[2])
]

print(f"\nVoxel grid size: {grid_size}")
print(f"Total possible voxels: {np.prod(grid_size):,}")
print(f"Max voxels config: [120000, 160000]")