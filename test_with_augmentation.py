#!/usr/bin/env python
"""
Test data validity with FULL training pipeline including augmentation.

The BEV pool crash only occurs during training because augmentation
(rotation, scaling, flipping) can push geometric features outside
the valid BEV range.

This script runs the full training pipeline multiple times per sample
(since augmentation is random) to find problematic cases.
"""

import os
import sys
import numpy as np
import torch
import random

sys.path.insert(0, '.')

from mmengine import Config
from mmcv.transforms import Compose
from mmdet3d.registry import TRANSFORMS, DATASETS


def print_separator(title=""):
    print("\n" + "="*70)
    if title:
        print(f" {title}")
        print("="*70)


def check_bev_validity(result, xbound, ybound):
    """Check if result has valid points in BEV range."""
    points = result.get('points', None)
    if points is None:
        return False, 0, "No points"
    
    if hasattr(points, 'tensor'):
        pts = points.tensor.numpy()
    else:
        pts = np.array(points)
    
    if len(pts) == 0:
        return False, 0, "Empty points"
    
    in_x = (pts[:, 0] >= xbound[0]) & (pts[:, 0] <= xbound[1])
    in_y = (pts[:, 1] >= ybound[0]) & (pts[:, 1] <= ybound[1])
    in_bev = in_x & in_y
    n_valid = in_bev.sum()
    
    if n_valid == 0:
        return False, 0, "No points in BEV range after augmentation"
    
    return True, n_valid, "OK"


def main():
    print_separator("BEV Pool Crash Test WITH Augmentation")
    
    config_path = 'projects/BEVFusion/configs/bevfusion_pandaset_all_cameras.py'
    cfg = Config.fromfile(config_path)
    
    # Get bounds
    view_transform = cfg.model.view_transform
    xbound = view_transform.xbound
    ybound = view_transform.ybound
    
    print(f"BEV X bound: {xbound}")
    print(f"BEV Y bound: {ybound}")
    
    # Create dataset with FULL training pipeline (including augmentation)
    print_separator("Building Dataset with Full Training Pipeline")
    
    dataset_cfg = cfg.train_dataloader.dataset.copy()
    dataset = DATASETS.build(dataset_cfg)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Pipeline transforms: {len(dataset.pipeline.transforms)}")
    
    # Print pipeline
    print("\nPipeline transforms:")
    for i, t in enumerate(dataset.pipeline.transforms):
        print(f"  {i}: {t.__class__.__name__}")
    
    # Test samples with multiple random augmentations
    print_separator("Testing Samples (Multiple Augmentation Runs)")
    
    num_samples = 100  # Test 100 samples
    runs_per_sample = 5  # Run augmentation 5 times per sample
    
    problem_cases = []
    total_tests = 0
    
    for idx in range(len(dataset)):
        for run in range(runs_per_sample):
            total_tests += 1
            
            try:
                # Set different random seed for each run
                seed = idx * 1000 + run
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                
                # Get sample through full pipeline
                sample = dataset[idx]
                
                if sample is None:
                    problem_cases.append({
                        'idx': idx,
                        'run': run,
                        'reason': 'Pipeline returned None',
                        'seed': seed
                    })
                    continue
                
                # Check points in inputs
                if 'inputs' in sample and 'points' in sample['inputs']:
                    points = sample['inputs']['points']
                    
                    if hasattr(points, 'numpy'):
                        pts = points.numpy()
                    else:
                        pts = np.array(points)
                    
                    if len(pts) == 0:
                        problem_cases.append({
                            'idx': idx,
                            'run': run,
                            'reason': 'Empty points after pipeline',
                            'seed': seed
                        })
                        continue
                    
                    # Check BEV validity
                    in_x = (pts[:, 0] >= xbound[0]) & (pts[:, 0] <= xbound[1])
                    in_y = (pts[:, 1] >= ybound[0]) & (pts[:, 1] <= ybound[1])
                    n_in_bev = (in_x & in_y).sum()
                    
                    if n_in_bev < 10:  # Very few points
                        problem_cases.append({
                            'idx': idx,
                            'run': run,
                            'reason': f'Only {n_in_bev} points in BEV range',
                            'seed': seed,
                            'pts_total': len(pts),
                            'pts_in_bev': n_in_bev
                        })
                        
            except Exception as e:
                error_msg = str(e)
                if 'index -1 is out of bounds' in error_msg or 'interval_starts' in error_msg:
                    problem_cases.append({
                        'idx': idx,
                        'run': run,
                        'reason': 'BEV Pool crash (empty intervals)',
                        'seed': seed,
                        'error': error_msg[:100]
                    })
                else:
                    problem_cases.append({
                        'idx': idx,
                        'run': run,
                        'reason': f'Error: {error_msg[:50]}',
                        'seed': seed
                    })
        
        # Progress
        if (idx + 1) % 20 == 0:
            print(f"  Tested {idx + 1}/{num_samples} samples, {len(problem_cases)} problems found")
    
    # Summary
    print_separator("Results")
    
    print(f"Total tests: {total_tests}")
    print(f"Problem cases: {len(problem_cases)}")
    print(f"Problem rate: {100 * len(problem_cases) / total_tests:.2f}%")
    
    if problem_cases:
        print("\nProblem cases:")
        
        # Group by reason
        reasons = {}
        for case in problem_cases:
            reason = case['reason']
            if reason not in reasons:
                reasons[reason] = []
            reasons[reason].append(case)
        
        for reason, cases in reasons.items():
            print(f"\n  {reason}: {len(cases)} cases")
            for case in cases[:3]:  # Show first 3
                print(f"    - Sample {case['idx']}, run {case['run']}, seed {case['seed']}")
        
        # Detailed analysis of first BEV pool crash
        bev_crashes = [c for c in problem_cases if 'BEV Pool crash' in c['reason']]
        if bev_crashes:
            print_separator("Detailed Analysis of BEV Pool Crash")
            
            crash = bev_crashes[0]
            idx = crash['idx']
            seed = crash['seed']
            
            print(f"Sample {idx}, seed {seed}")
            
            # Reproduce step by step
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # Get raw info
            info = dataset.get_data_info(idx)
            info['ann_info'] = dataset.parse_ann_info(info)
            info['box_type_3d'] = dataset.box_type_3d
            info['box_mode_3d'] = dataset.box_mode_3d
            
            # Run transforms one by one
            result = info.copy()
            for i, transform in enumerate(dataset.pipeline.transforms):
                try:
                    prev_pts = None
                    if 'points' in result and result['points'] is not None:
                        if hasattr(result['points'], 'tensor'):
                            prev_pts = len(result['points'].tensor)
                    
                    result = transform(result)
                    
                    if result is None:
                        print(f"  Transform {i} ({transform.__class__.__name__}): returned None")
                        break
                    
                    curr_pts = None
                    if 'points' in result and result['points'] is not None:
                        if hasattr(result['points'], 'tensor'):
                            curr_pts = len(result['points'].tensor)
                            pts = result['points'].tensor.numpy()
                            in_bev = ((pts[:, 0] >= xbound[0]) & (pts[:, 0] <= xbound[1]) &
                                     (pts[:, 1] >= ybound[0]) & (pts[:, 1] <= ybound[1])).sum()
                            print(f"  Transform {i} ({transform.__class__.__name__}): {curr_pts} pts, {in_bev} in BEV")
                    else:
                        print(f"  Transform {i} ({transform.__class__.__name__}): no points")
                        
                except Exception as e:
                    print(f"  Transform {i} ({transform.__class__.__name__}): ERROR - {str(e)[:60]}")
                    break
    else:
        print("\n✓ No problems found in tested samples")
        print("  The crash may be very rare or require more test runs")
        print("  Consider increasing runs_per_sample or num_samples")


if __name__ == '__main__':
    main()