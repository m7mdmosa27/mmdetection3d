#!/usr/bin/env python
"""
Diagnostic script for PandaSet BEVFusion training issues

This script verifies:
1. Whether annotations are in world or ego coordinates
2. Whether points are being properly transformed
3. Coordinate system alignment between points, annotations, and camera matrices

Run this from the mmdetection3d directory:
    python diagnose_pandaset_issues.py
"""

import os
import sys
import pickle
import numpy as np
import json

# Configuration - adjust paths as needed
DATA_ROOT = 'data/pandaset/'
TRAIN_PKL = os.path.join(DATA_ROOT, 'pandaset_infos_train.pkl')


def quaternion_to_rotation_matrix(w, x, y, z):
    """Convert quaternion to rotation matrix."""
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ], dtype=np.float32)


def build_transform_matrix(position, heading):
    """Build 4x4 transformation matrix from position and quaternion heading."""
    t = np.array([
        position.get('x', 0.0),
        position.get('y', 0.0),
        position.get('z', 0.0)
    ], dtype=np.float32)
    
    w = heading.get('w', 1.0)
    x = heading.get('x', 0.0)
    y = heading.get('y', 0.0)
    z = heading.get('z', 0.0)
    
    R = quaternion_to_rotation_matrix(w, x, y, z)
    
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def load_lidar_pose(scene_dir, frame_idx):
    """Load LiDAR pose for a frame."""
    poses_path = os.path.join(scene_dir, 'lidar', 'poses.json')
    if not os.path.exists(poses_path):
        return None
    
    with open(poses_path, 'r') as f:
        poses = json.load(f)
    
    if frame_idx >= len(poses):
        return None
    
    pose = poses[frame_idx]
    return build_transform_matrix(
        pose.get('position', {}),
        pose.get('heading', {})
    )


def load_points_from_pkl(pkl_path):
    """Load point cloud from PandaSet pkl file."""
    if not os.path.exists(pkl_path):
        return None
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different formats
    if isinstance(data, np.ndarray):
        return data
    elif hasattr(data, 'values'):  # DataFrame
        cols = ['x', 'y', 'z', 'i'] if 'i' in data.columns else ['x', 'y', 'z', 'intensity']
        available_cols = [c for c in cols if c in data.columns]
        if len(available_cols) >= 3:
            return data[available_cols].values.astype(np.float32)
    return None


def load_annotations(anno_path):
    """Load annotation cuboids."""
    if not os.path.exists(anno_path):
        return None
    
    with open(anno_path, 'rb') as f:
        data = pickle.load(f)
    
    # Check sensor_id
    if hasattr(data, 'columns') and 'cuboids.sensor_id' in data.columns:
        sensor_ids = data['cuboids.sensor_id'].unique()
        print(f"    Annotation sensor_ids: {sensor_ids}")
        if -1 in sensor_ids:
            print("    ⚠️  sensor_id=-1 means annotations are in WORLD coordinates!")
    
    return data


def print_separator(title=""):
    print("\n" + "="*70)
    if title:
        print(f" {title}")
        print("="*70)


def diagnose_single_sample(info, idx):
    """Diagnose a single sample."""
    print_separator(f"Sample {idx}: {info.get('sample_idx', 'N/A')}")
    
    sample_idx = info.get('sample_idx', '')
    if '_' in str(sample_idx):
        scene_id, frame_str = str(sample_idx).split('_')
        frame_idx = int(frame_str)
    else:
        print("  Cannot parse sample_idx")
        return
    
    scene_dir = os.path.join(DATA_ROOT, scene_id)
    
    # 1. Load and analyze LiDAR pose
    print("\n1. LiDAR Pose (world_from_ego):")
    world_from_ego = load_lidar_pose(scene_dir, frame_idx)
    if world_from_ego is not None:
        ego_position = world_from_ego[:3, 3]
        print(f"   Ego position in world: [{ego_position[0]:.2f}, {ego_position[1]:.2f}, {ego_position[2]:.2f}]")
    else:
        print("   ❌ Could not load LiDAR pose")
    
    # 2. Load and analyze point cloud
    print("\n2. Point Cloud Analysis:")
    lidar_path = info.get('lidar_points', {}).get('lidar_path', '')
    if not lidar_path:
        lidar_path = info.get('lidar_path', '')
    
    full_lidar_path = os.path.join(DATA_ROOT, lidar_path) if lidar_path else None
    
    if full_lidar_path and os.path.exists(full_lidar_path):
        points = load_points_from_pkl(full_lidar_path)
        if points is not None:
            pts_mean = points[:, :3].mean(axis=0)
            pts_std = points[:, :3].std(axis=0)
            pts_min = points[:, :3].min(axis=0)
            pts_max = points[:, :3].max(axis=0)
            
            print(f"   Points shape: {points.shape}")
            print(f"   Mean XYZ (WORLD): [{pts_mean[0]:.2f}, {pts_mean[1]:.2f}, {pts_mean[2]:.2f}]")
            print(f"   Std XYZ: [{pts_std[0]:.2f}, {pts_std[1]:.2f}, {pts_std[2]:.2f}]")
            print(f"   Min XYZ: [{pts_min[0]:.2f}, {pts_min[1]:.2f}, {pts_min[2]:.2f}]")
            print(f"   Max XYZ: [{pts_max[0]:.2f}, {pts_max[1]:.2f}, {pts_max[2]:.2f}]")
            
            # Transform to ego and show
            if world_from_ego is not None:
                ego_from_world = np.linalg.inv(world_from_ego)
                pts_xyz = points[:, :3]
                pts_ego = (ego_from_world[:3, :3] @ pts_xyz.T).T + ego_from_world[:3, 3]
                pts_ego_mean = pts_ego.mean(axis=0)
                pts_ego_min = pts_ego.min(axis=0)
                pts_ego_max = pts_ego.max(axis=0)
                
                print(f"\n   After world→ego transform:")
                print(f"   Mean XYZ (EGO): [{pts_ego_mean[0]:.2f}, {pts_ego_mean[1]:.2f}, {pts_ego_mean[2]:.2f}]")
                print(f"   Min XYZ (EGO): [{pts_ego_min[0]:.2f}, {pts_ego_min[1]:.2f}, {pts_ego_min[2]:.2f}]")
                print(f"   Max XYZ (EGO): [{pts_ego_max[0]:.2f}, {pts_ego_max[1]:.2f}, {pts_ego_max[2]:.2f}]")
                
                # Check if within expected range
                pc_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
                in_range = (pts_ego_min[0] >= pc_range[0] and pts_ego_max[0] <= pc_range[3] and
                           pts_ego_min[1] >= pc_range[1] and pts_ego_max[1] <= pc_range[4])
                print(f"\n   Points within BEV range [-54, 54]? {'✓ Yes' if in_range else '❌ No - some points outside!'}")
    else:
        print(f"   ❌ Could not load points from: {full_lidar_path}")
    
    # 3. Load and analyze annotations
    print("\n3. Annotation Analysis:")
    gt_boxes = info.get('gt_boxes', None)
    if gt_boxes is None:
        gt_boxes = info.get('instances', [])
    
    if isinstance(gt_boxes, np.ndarray) and len(gt_boxes) > 0:
        gt_centers = gt_boxes[:, :3]
        gt_mean = gt_centers.mean(axis=0)
        gt_min = gt_centers.min(axis=0)
        gt_max = gt_centers.max(axis=0)
        
        print(f"   Number of GT boxes: {len(gt_boxes)}")
        print(f"   GT center mean: [{gt_mean[0]:.2f}, {gt_mean[1]:.2f}, {gt_mean[2]:.2f}]")
        print(f"   GT center min: [{gt_min[0]:.2f}, {gt_min[1]:.2f}, {gt_min[2]:.2f}]")
        print(f"   GT center max: [{gt_max[0]:.2f}, {gt_max[1]:.2f}, {gt_max[2]:.2f}]")
        
        # Check if GT seems to be in world or ego coordinates
        if world_from_ego is not None:
            ego_pos = world_from_ego[:3, 3]
            
            # If GT is in world coords, it should be close to ego position
            dist_to_ego_world = np.linalg.norm(gt_mean[:2] - ego_pos[:2])
            dist_to_origin = np.linalg.norm(gt_mean[:2])
            
            print(f"\n   Distance analysis:")
            print(f"   - GT center to world origin: {dist_to_origin:.2f}m")
            print(f"   - GT center to ego position (world): {dist_to_ego_world:.2f}m")
            
            if dist_to_ego_world < dist_to_origin and dist_to_origin > 50:
                print("   ⚠️  GT appears to be in WORLD coordinates!")
                print("   ⚠️  GT boxes need to be transformed to EGO frame!")
            elif dist_to_origin < 100:
                print("   ✓ GT appears to be in EGO coordinates (centered near origin)")
    elif isinstance(gt_boxes, list) and len(gt_boxes) > 0:
        print(f"   Number of instances: {len(gt_boxes)}")
        if 'bbox_3d' in gt_boxes[0]:
            centers = np.array([inst['bbox_3d'][:3] for inst in gt_boxes])
            gt_mean = centers.mean(axis=0)
            print(f"   GT center mean: [{gt_mean[0]:.2f}, {gt_mean[1]:.2f}, {gt_mean[2]:.2f}]")
    else:
        print("   No GT boxes found in info")
        
        # Try loading from annotation file directly
        anno_path_options = [
            info.get('anno_path', ''),
            f"{scene_id}/annotations/cuboids/{frame_idx:02d}.pkl"
        ]
        
        for anno_path in anno_path_options:
            full_anno_path = os.path.join(DATA_ROOT, anno_path)
            if os.path.exists(full_anno_path):
                print(f"\n   Loading annotations from: {anno_path}")
                load_annotations(full_anno_path)
                break


def main():
    print_separator("PandaSet BEVFusion Diagnostic Tool")
    
    # Check paths
    if not os.path.exists(DATA_ROOT):
        print(f"❌ Data root not found: {DATA_ROOT}")
        print("   Please adjust DATA_ROOT in this script")
        return
    
    if not os.path.exists(TRAIN_PKL):
        print(f"❌ Training pkl not found: {TRAIN_PKL}")
        return
    
    # Load training info
    print(f"\nLoading {TRAIN_PKL}...")
    with open(TRAIN_PKL, 'rb') as f:
        train_infos = pickle.load(f)
    
    print(f"Loaded {len(train_infos)} samples")
    
    # Diagnose a few samples
    samples_to_check = [0, len(train_infos)//2, len(train_infos)-1]
    
    for idx in samples_to_check:
        if idx < len(train_infos):
            diagnose_single_sample(train_infos[idx], idx)
    
    # Summary
    print_separator("SUMMARY")
    print("""
Based on the analysis above, check:

1. COORDINATE SYSTEM:
   - If GT centers are far from origin (>50m) but points after transform
     are centered around origin → GT is in WORLD coordinates
   - Solution: Transform GT boxes using ego_from_world matrix

2. BEV RANGE:
   - Points (in EGO frame) should be within [-54, 54] for X and Y
   - GT boxes should also be within this range
   - If not, either transform is wrong or range needs adjustment

3. CAMERA MATRICES:
   - lidar2cam should transform EGO frame points to camera frame
   - If points are transformed but GT is not, there's a mismatch

RECOMMENDED FIXES:
1. Modify PandaSetWorldToEgo to also transform GT boxes
2. Or transform GT boxes during info file creation
""")


if __name__ == '__main__':
    main()