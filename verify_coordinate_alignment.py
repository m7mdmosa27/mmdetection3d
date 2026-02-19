#!/usr/bin/env python
"""
Verify that PandaSetWorldToEgo correctly transforms BOTH points AND GT boxes.

Run this after applying the fix to confirm:
1. Points are in EGO frame (centered around origin)
2. GT boxes are in EGO frame (centered around origin)
3. Points and GT boxes are in the same coordinate system

Usage:
    python verify_coordinate_alignment.py
"""

import os
import sys
import numpy as np
import pickle

# Add mmdetection3d to path if needed
sys.path.insert(0, '.')

def print_separator(title=""):
    print("\n" + "="*70)
    if title:
        print(f" {title}")
        print("="*70)


def main():
    print_separator("Coordinate Alignment Verification")
    
    # Import after path setup
    from mmdet3d.datasets import PandaSetDataset
    from mmdet3d.datasets.transforms import (
        LoadPandaSetPointsFromPKL,
        LoadAnnotations3D,
        PandaSetWorldToEgo
    )
    
    data_root = 'data/pandaset'
    
    # Create minimal pipeline to test the transform
    pipeline = [
        dict(type='LoadPandaSetPointsFromPKL', coord_type='LIDAR', load_dim=4, use_dim=4),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
        dict(type='PandaSetWorldToEgo'),  # This should now transform BOTH
    ]
    
    # Build transforms
    from mmcv.transforms import Compose
    from mmdet3d.registry import TRANSFORMS
    
    transform = Compose([TRANSFORMS.build(t) for t in pipeline])
    
    # Create dataset without pipeline to get raw data
    dataset = PandaSetDataset(
        data_root=data_root,
        ann_file='pandaset_infos_train.pkl',
        pipeline=None,
        data_prefix=dict(pts='', img='', sweeps=''),
        test_mode=False,
        modality=dict(use_lidar=True, use_camera=False),
        box_type_3d='LiDAR',
        filter_empty_gt=False
    )
    
    print(f"Testing on {len(dataset)} samples...")
    
    # Test a few samples
    test_indices = [0, len(dataset)//2, len(dataset)-1]
    
    all_passed = True
    
    for idx in test_indices:
        print_separator(f"Sample {idx}")
        
        # Get raw data
        info = dataset.get_data_info(idx)
        info['ann_info'] = dataset.parse_ann_info(info)
        info['box_type_3d'] = dataset.box_type_3d
        info['box_mode_3d'] = dataset.box_mode_3d
        
        sample_idx = info.get('sample_idx', 'N/A')
        print(f"Sample ID: {sample_idx}")
        
        # Get GT boxes BEFORE transform
        gt_boxes_before = info['ann_info']['gt_bboxes_3d'].tensor.numpy()
        if len(gt_boxes_before) > 0:
            gt_center_before = gt_boxes_before[:, :3].mean(axis=0)
            print(f"\nBEFORE transform (WORLD coords):")
            print(f"  GT center mean: [{gt_center_before[0]:.2f}, {gt_center_before[1]:.2f}, {gt_center_before[2]:.2f}]")
        
        # Apply transform
        result = transform(info.copy())
        
        # Check points
        if 'points' in result:
            pts = result['points'].tensor.numpy()
            pts_mean = pts[:, :3].mean(axis=0)
            pts_std = pts[:, :3].std(axis=0)
            
            print(f"\nAFTER transform (EGO coords):")
            print(f"  Points mean: [{pts_mean[0]:.2f}, {pts_mean[1]:.2f}, {pts_mean[2]:.2f}]")
            print(f"  Points std:  [{pts_std[0]:.2f}, {pts_std[1]:.2f}, {pts_std[2]:.2f}]")
            
            # Points should be roughly centered (mean close to 0 for x,y, small positive for z)
            if abs(pts_mean[0]) < 10 and abs(pts_mean[1]) < 15:
                print("  ✓ Points appear to be in EGO frame")
            else:
                print("  ✗ Points may not be properly transformed!")
                all_passed = False
        
        # Check GT boxes
        if 'gt_bboxes_3d' in result:
            gt_boxes = result['gt_bboxes_3d'].tensor.numpy()
            if len(gt_boxes) > 0:
                gt_center = gt_boxes[:, :3].mean(axis=0)
                gt_min = gt_boxes[:, :3].min(axis=0)
                gt_max = gt_boxes[:, :3].max(axis=0)
                
                print(f"\n  GT boxes ({len(gt_boxes)} objects):")
                print(f"    Center mean: [{gt_center[0]:.2f}, {gt_center[1]:.2f}, {gt_center[2]:.2f}]")
                print(f"    Center min:  [{gt_min[0]:.2f}, {gt_min[1]:.2f}, {gt_min[2]:.2f}]")
                print(f"    Center max:  [{gt_max[0]:.2f}, {gt_max[1]:.2f}, {gt_max[2]:.2f}]")
                
                # Check if GT boxes are within reasonable EGO range
                pc_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
                in_range = (
                    gt_min[0] >= pc_range[0] - 20 and gt_max[0] <= pc_range[3] + 20 and
                    gt_min[1] >= pc_range[1] - 20 and gt_max[1] <= pc_range[4] + 20
                )
                
                if in_range:
                    print("    ✓ GT boxes appear to be in EGO frame")
                else:
                    print("    ✗ GT boxes may not be properly transformed!")
                    all_passed = False
                
                # Check alignment: GT and points should be in similar range
                if 'points' in result:
                    pts_in_range = pts[
                        (pts[:, 0] >= -60) & (pts[:, 0] <= 60) &
                        (pts[:, 1] >= -60) & (pts[:, 1] <= 60)
                    ]
                    if len(pts_in_range) > 0:
                        pts_range_mean = pts_in_range[:, :3].mean(axis=0)
                        
                        # GT center should be somewhat close to points center
                        dist = np.linalg.norm(gt_center[:2] - pts_range_mean[:2])
                        print(f"\n  Alignment check:")
                        print(f"    Distance between GT center and points center: {dist:.2f}m")
                        
                        if dist < 30:  # Allow some difference since GT might be sparse
                            print("    ✓ Points and GT boxes appear aligned")
                        else:
                            print("    ⚠ Points and GT boxes might be misaligned")
            else:
                print("\n  No GT boxes in this sample")
    
    # Summary
    print_separator("SUMMARY")
    
    if all_passed:
        print("""
✓ All checks passed!

The PandaSetWorldToEgo transform appears to be correctly transforming
both points AND GT boxes to EGO frame.

You should now see:
- Training loss decreasing properly
- Higher matched_ious during training
- Non-zero mAP during validation
""")
    else:
        print("""
✗ Some checks failed!

The transform may not be working correctly. Please verify:
1. You replaced the PandaSetWorldToEgo class with the fixed version
2. The transform is registered with force=True to override the old one
3. The pipeline order is correct (LoadAnnotations3D before PandaSetWorldToEgo)
""")


if __name__ == '__main__':
    main()