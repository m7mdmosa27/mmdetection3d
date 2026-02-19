#!/usr/bin/env python
"""
Diagnose the BEV Pool empty interval issue.

The crash occurs when:
    interval_starts = torch.where(kept)[0].int()
    interval_starts.numel() == 0  # Empty!

This happens when geometric features (geom_feats) result in no valid voxels.

This script traces back to find why geom_feats becomes empty.
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, '.')


def print_separator(title=""):
    print("\n" + "="*70)
    if title:
        print(f" {title}")
        print("="*70)


def simulate_bev_pool_geometry(points, lidar2img, img_shape, xbound, ybound, zbound, dbound):
    """
    Simulate the geometry calculation that happens before BEV pool.
    
    This mimics what happens in depth_lss.py get_geometry() and bev_pool().
    """
    
    # Calculate grid dimensions
    nx = int((xbound[1] - xbound[0]) / xbound[2])
    ny = int((ybound[1] - ybound[0]) / ybound[2])
    nz = int((zbound[1] - zbound[0]) / zbound[2])
    nd = int((dbound[1] - dbound[0]) / dbound[2])
    
    print(f"Grid: nx={nx}, ny={ny}, nz={nz}, nd={nd}")
    
    # For each depth bin, create frustum points
    # In the actual code, this creates a frustum of points for each pixel and depth
    
    # Simplified check: see if points project into valid image coordinates
    # and fall within the BEV bounds
    
    if len(points) == 0:
        print("No points to check!")
        return False
    
    pts = points[:, :3]
    
    # Check if points are in BEV range
    in_x = (pts[:, 0] >= xbound[0]) & (pts[:, 0] <= xbound[1])
    in_y = (pts[:, 1] >= ybound[0]) & (pts[:, 1] <= ybound[1])
    in_z = (pts[:, 2] >= zbound[0]) & (pts[:, 2] <= zbound[1])
    
    in_bev = in_x & in_y & in_z
    n_in_bev = in_bev.sum()
    
    print(f"Points in X range [{xbound[0]}, {xbound[1]}]: {in_x.sum()}/{len(pts)}")
    print(f"Points in Y range [{ybound[0]}, {ybound[1]}]: {in_y.sum()}/{len(pts)}")
    print(f"Points in Z range [{zbound[0]}, {zbound[1]}]: {in_z.sum()}/{len(pts)}")
    print(f"Points in full BEV range: {n_in_bev}/{len(pts)}")
    
    if n_in_bev == 0:
        print("⚠️  NO POINTS IN BEV RANGE - This will cause empty BEV pool!")
        return False
    
    # Check depth range
    # In ego frame, depth is typically distance from origin
    depths = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    in_depth = (depths >= dbound[0]) & (depths <= dbound[1])
    
    print(f"Points in depth range [{dbound[0]}, {dbound[1]}]: {in_depth.sum()}/{len(pts)}")
    
    # Combined valid points
    valid = in_bev & in_depth
    n_valid = valid.sum()
    print(f"Fully valid points (BEV + depth): {n_valid}/{len(pts)}")
    
    return n_valid > 0


def check_camera_projection(points, lidar2img, img_h, img_w):
    """Check if points project into valid image coordinates."""
    
    if len(points) == 0:
        return 0
    
    pts = points[:, :3]
    
    # Add homogeneous coordinate
    pts_homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    
    # Project to image
    pts_img = (lidar2img @ pts_homo.T).T
    
    # Normalize by depth
    depth = pts_img[:, 2]
    valid_depth = depth > 0.1  # Must be in front of camera
    
    pts_2d = pts_img[:, :2] / (depth[:, None] + 1e-6)
    
    # Check if in image bounds
    in_img = (
        valid_depth &
        (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < img_w) &
        (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < img_h)
    )
    
    return in_img.sum()


def main():
    from mmengine import Config
    from mmcv.transforms import Compose
    from mmdet3d.registry import TRANSFORMS, DATASETS
    
    print_separator("BEV Pool Empty Interval Diagnosis")
    
    config_path = 'projects/BEVFusion/configs/bevfusion_pandaset_all_cameras.py'
    cfg = Config.fromfile(config_path)
    
    # Get bounds from config
    view_transform = cfg.model.view_transform
    xbound = view_transform.xbound
    ybound = view_transform.ybound
    zbound = view_transform.zbound
    dbound = view_transform.dbound
    
    print(f"View Transform Bounds:")
    print(f"  xbound: {xbound}")
    print(f"  ybound: {ybound}")
    print(f"  zbound: {zbound}")
    print(f"  dbound: {dbound}")
    
    pc_range = cfg.point_cloud_range
    print(f"\nPoint Cloud Range: {pc_range}")
    
    # Create dataset
    dataset_cfg = cfg.train_dataloader.dataset.copy()
    dataset_cfg['pipeline'] = []
    dataset = DATASETS.build(dataset_cfg)
    
    # Build pipeline up to the point before BEV pool
    pipeline_cfg = [
        dict(type='BEVLoadMultiViewImageFromFiles', to_float32=True, color_type='color', num_views=6),
        dict(type='LoadPandaSetPointsFromPKL', coord_type='LIDAR', load_dim=4, use_dim=4),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
        dict(type='PandaSetWorldToEgo'),
        dict(type='PointsRangeFilter', point_cloud_range=pc_range),
    ]
    
    pipeline = Compose([TRANSFORMS.build(t) for t in pipeline_cfg])
    
    # Test multiple samples
    print_separator("Testing Samples")
    
    problem_samples = []
    
    for idx in range(len(dataset)):
        info = dataset.get_data_info(idx)
        info['ann_info'] = dataset.parse_ann_info(info)
        info['box_type_3d'] = dataset.box_type_3d
        info['box_mode_3d'] = dataset.box_mode_3d
        
        try:
            result = pipeline(info.copy())
            
            if result is None:
                problem_samples.append((idx, "Pipeline returned None"))
                continue
            
            points = result.get('points', None)
            if points is None or len(points.tensor) == 0:
                problem_samples.append((idx, "No points after pipeline"))
                continue
            
            pts = points.tensor.numpy()
            
            # Check if points are in valid range for BEV pool
            in_x = (pts[:, 0] >= xbound[0]) & (pts[:, 0] <= xbound[1])
            in_y = (pts[:, 1] >= ybound[0]) & (pts[:, 1] <= ybound[1])
            in_bev = in_x & in_y
            
            n_valid = in_bev.sum()
            
            if n_valid == 0:
                problem_samples.append((idx, f"No points in BEV range"))
            elif n_valid < 100:
                problem_samples.append((idx, f"Only {n_valid} points in BEV range"))
                
        except Exception as e:
            problem_samples.append((idx, str(e)[:50]))
    
    # Summary
    print_separator("Results")
    
    print(f"Tested: {min(50, len(dataset))} samples")
    print(f"Problem samples: {len(problem_samples)}")
    
    if problem_samples:
        print("\nProblem samples:")
        for idx, reason in problem_samples[:10]:
            print(f"  Sample {idx}: {reason}")
    else:
        print("\n✓ All samples have valid points in BEV range")
    
    # Detailed analysis of first problem sample
    if problem_samples:
        print_separator("Detailed Analysis of First Problem Sample")
        
        idx = problem_samples[0][0]
        info = dataset.get_data_info(idx)
        info['ann_info'] = dataset.parse_ann_info(info)
        info['box_type_3d'] = dataset.box_type_3d
        info['box_mode_3d'] = dataset.box_mode_3d
        
        sample_idx = info.get('sample_idx', 'N/A')
        print(f"Sample ID: {sample_idx}")
        
        # Check each stage
        stages = [
            ('LoadPoints', [dict(type='LoadPandaSetPointsFromPKL', coord_type='LIDAR', load_dim=4, use_dim=4)]),
            ('WorldToEgo', [
                dict(type='LoadPandaSetPointsFromPKL', coord_type='LIDAR', load_dim=4, use_dim=4),
                dict(type='PandaSetWorldToEgo'),
            ]),
            ('PointsRangeFilter', [
                dict(type='LoadPandaSetPointsFromPKL', coord_type='LIDAR', load_dim=4, use_dim=4),
                dict(type='PandaSetWorldToEgo'),
                dict(type='PointsRangeFilter', point_cloud_range=pc_range),
            ]),
        ]
        
        for stage_name, stage_cfg in stages:
            print(f"\n--- {stage_name} ---")
            stage_pipeline = Compose([TRANSFORMS.build(t) for t in stage_cfg])
            
            try:
                result = stage_pipeline(info.copy())
                
                if result is None:
                    print("  Result is None!")
                    continue
                
                points = result.get('points', None)
                if points is None:
                    print("  No points!")
                    continue
                
                pts = points.tensor.numpy()
                print(f"  Points: {len(pts)}")
                
                if len(pts) > 0:
                    print(f"  X range: [{pts[:, 0].min():.1f}, {pts[:, 0].max():.1f}]")
                    print(f"  Y range: [{pts[:, 1].min():.1f}, {pts[:, 1].max():.1f}]")
                    print(f"  Z range: [{pts[:, 2].min():.1f}, {pts[:, 2].max():.1f}]")
                    
                    # Check BEV validity
                    in_x = (pts[:, 0] >= xbound[0]) & (pts[:, 0] <= xbound[1])
                    in_y = (pts[:, 1] >= ybound[0]) & (pts[:, 1] <= ybound[1])
                    print(f"  In X bound: {in_x.sum()}/{len(pts)}")
                    print(f"  In Y bound: {in_y.sum()}/{len(pts)}")
                    
            except Exception as e:
                print(f"  Error: {e}")


if __name__ == '__main__':
    main()