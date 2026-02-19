#!/usr/bin/env python
"""
Validate and filter training data to prevent BEV pool crashes.

This script can be used to:
1. Pre-validate all training samples with augmentation
2. Create a filtered info file excluding problematic samples
3. Identify augmentation settings that cause issues

The BEV pool crash happens when geometric features fall outside the valid range
after augmentation, resulting in empty interval_starts.
"""

import os
import sys
import pickle
import numpy as np
import torch
import random
from tqdm import tqdm

sys.path.insert(0, '.')


def print_separator(title=""):
    print("\n" + "="*70)
    if title:
        print(f" {title}")
        print("="*70)


def validate_sample_with_augmentation(dataset, idx, xbound, ybound, num_runs=10):
    """
    Validate a single sample with multiple augmentation runs.
    
    Returns:
        (is_valid, failure_rate, details)
    """
    failures = []
    
    for run in range( ):
        seed = idx * 10000 + run
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        try:
            sample = dataset[idx]
            
            if sample is None:
                failures.append({'run': run, 'reason': 'None result'})
                continue
            
            # Check points
            if 'inputs' in sample and 'points' in sample['inputs']:
                points = sample['inputs']['points']
                
                if hasattr(points, 'numpy'):
                    pts = points.numpy()
                elif hasattr(points, 'cpu'):
                    pts = points.cpu().numpy()
                else:
                    pts = np.array(points)
                
                if len(pts) == 0:
                    failures.append({'run': run, 'reason': 'Empty points'})
                    continue
                
                # Check BEV range
                in_bev = (
                    (pts[:, 0] >= xbound[0]) & (pts[:, 0] <= xbound[1]) &
                    (pts[:, 1] >= ybound[0]) & (pts[:, 1] <= ybound[1])
                ).sum()
                
                if in_bev < 100:  # Too few points in BEV
                    failures.append({
                        'run': run, 
                        'reason': f'Only {in_bev} points in BEV',
                        'total_pts': len(pts)
                    })
                    
        except Exception as e:
            error_str = str(e)
            if 'interval_starts' in error_str or 'index -1' in error_str:
                failures.append({'run': run, 'reason': 'BEV pool crash'})
            else:
                failures.append({'run': run, 'reason': f'Error: {error_str[:50]}'})
    
    failure_rate = len(failures) / num_runs
    is_valid = failure_rate < 0.5  # Consider valid if less than 50% failure rate
    
    return is_valid, failure_rate, failures


def main():
    from mmengine import Config
    from mmdet3d.registry import DATASETS
    
    print_separator("Training Data Validation")
    
    config_path = 'projects/BEVFusion/configs/bevfusion_pandaset_all_cameras.py'
    cfg = Config.fromfile(config_path)
    
    # Get bounds
    view_transform = cfg.model.view_transform
    xbound = view_transform.xbound
    ybound = view_transform.ybound
    
    print(f"Config: {config_path}")
    print(f"BEV X bound: {xbound}")
    print(f"BEV Y bound: {ybound}")
    
    # Build dataset with full pipeline
    print_separator("Building Dataset")
    
    dataset_cfg = cfg.train_dataloader.dataset.copy()
    dataset = DATASETS.build(dataset_cfg)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Validate samples
    print_separator("Validating Samples")
    
    num_samples = len(dataset)
    num_runs = 5  # Augmentation runs per sample
    
    valid_indices = []
    invalid_indices = []
    failure_details = {}
    
    for idx in tqdm(range(num_samples), desc="Validating"):
        is_valid, failure_rate, failures = validate_sample_with_augmentation(
            dataset, idx, xbound, ybound, num_runs
        )
        
        if is_valid:
            valid_indices.append(idx)
        else:
            invalid_indices.append(idx)
            failure_details[idx] = {
                'failure_rate': failure_rate,
                'failures': failures
            }
    
    # Summary
    print_separator("Validation Results")
    
    print(f"Total samples: {num_samples}")
    print(f"Valid samples: {len(valid_indices)} ({100*len(valid_indices)/num_samples:.1f}%)")
    print(f"Invalid samples: {len(invalid_indices)} ({100*len(invalid_indices)/num_samples:.1f}%)")
    
    if invalid_indices:
        print(f"\nInvalid sample indices (first 20): {invalid_indices[:20]}")
        
        # Analyze failure reasons
        reasons = {}
        for idx, details in failure_details.items():
            for failure in details['failures']:
                reason = failure['reason']
                if reason not in reasons:
                    reasons[reason] = 0
                reasons[reason] += 1
        
        print("\nFailure reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
        
        # Option to create filtered dataset
        print_separator("Create Filtered Dataset?")
        print("You can create a new info file excluding invalid samples.")
        print("This would prevent BEV pool crashes during training.")
        
        # Save invalid indices for reference
        invalid_file = 'data/pandaset/invalid_sample_indices.pkl'
        with open(invalid_file, 'wb') as f:
            pickle.dump({
                'invalid_indices': invalid_indices,
                'failure_details': failure_details
            }, f)
        print(f"\nSaved invalid sample info to: {invalid_file}")
        
        # Create filtered info file
        create_filtered = input("\nCreate filtered info file? (y/n): ").lower() == 'y'
        
        if create_filtered:
            # Load original info
            info_file = os.path.join(cfg.data_root, cfg.train_dataloader.dataset.ann_file)
            with open(info_file, 'rb') as f:
                infos = pickle.load(f)
            
            # Filter
            filtered_infos = [infos[i] for i in valid_indices]
            
            # Save
            filtered_file = info_file.replace('.pkl', '_filtered.pkl')
            with open(filtered_file, 'wb') as f:
                pickle.dump(filtered_infos, f)
            
            print(f"\nCreated filtered info file: {filtered_file}")
            print(f"Original samples: {len(infos)}")
            print(f"Filtered samples: {len(filtered_infos)}")
            print(f"\nTo use, update your config:")
            print(f"  ann_file='pandaset_infos_train_filtered.pkl'")
    else:
        print("\n✓ All samples are valid!")
        print("  If BEV pool crashes still occur, they may be very rare")
        print("  or caused by specific augmentation combinations.")


if __name__ == '__main__':
    main()