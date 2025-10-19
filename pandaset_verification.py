#!/usr/bin/env python3
"""
PandaSet MMDetection3D Integration Verification Script (FIXED)

This script systematically tests each component of your PandaSet integration.
Key fix: Pre-imports all transforms to ensure registry compatibility.

Usage:
    python pandaset_verification.py --data-root data/pandaset
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Import mmdet3d modules (this is sufficient now that we use custom Compose)
print("Pre-loading MMDetection3D modules...")
from mmdet3d.datasets import PandaSetDataset
from mmdet3d.datasets.transforms import LoadPandaSetPointsFromPKL
print("✓ Modules loaded\n")

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_section(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")

def print_info(msg):
    print(f"  {msg}")


# ==============================================================================
# TEST 1: Verify Info Files Structure
# ==============================================================================
def test_info_files(data_root):
    """Verify .pkl info files were generated correctly."""
    print_section("TEST 1: Info Files Structure")
    
    train_pkl = os.path.join(data_root, 'pandaset_infos_train.pkl')
    val_pkl = os.path.join(data_root, 'pandaset_infos_val.pkl')
    
    errors = []
    
    if not os.path.exists(train_pkl):
        print_error(f"Training info file not found: {train_pkl}")
        errors.append("train_pkl_missing")
    else:
        print_success(f"Found training info file")
    
    if not os.path.exists(val_pkl):
        print_error(f"Validation info file not found: {val_pkl}")
        errors.append("val_pkl_missing")
    else:
        print_success(f"Found validation info file")
    
    if errors:
        return False, errors
    
    print_info("\nLoading info files...")
    train_infos = pickle.load(open(train_pkl, 'rb'))
    val_infos = pickle.load(open(val_pkl, 'rb'))
    
    print_success(f"Train samples: {len(train_infos)}")
    print_success(f"Val samples: {len(val_infos)}")
    
    print_info("\nInspecting first training sample structure:")
    sample = train_infos[0]
    required_keys = ['sample_idx', 'lidar_path', 'anno_path']
    
    for key in required_keys:
        if key in sample:
            print_success(f"  '{key}': {sample[key]}")
        else:
            print_error(f"  Missing required key: '{key}'")
            errors.append(f"missing_key_{key}")
    
    optional_keys = ['img_path', 'calib']
    for key in optional_keys:
        if key in sample:
            print_info(f"  '{key}': Present")
    
    print_info("\nVerifying file paths exist:")
    test_samples = train_infos[:5]
    
    for i, sample in enumerate(test_samples):
        lidar_path = sample['lidar_path']
        anno_path = sample['anno_path']
        
        if not os.path.isabs(lidar_path):
            lidar_path = os.path.join(data_root, lidar_path)
        if not os.path.isabs(anno_path):
            anno_path = os.path.join(data_root, anno_path)
        
        if os.path.exists(lidar_path):
            print_success(f"Sample {i}: LiDAR file exists")
        else:
            print_error(f"Sample {i}: LiDAR file missing: {lidar_path}")
            errors.append(f"lidar_missing_{i}")
        
        if os.path.exists(anno_path):
            print_success(f"Sample {i}: Annotation file exists")
        else:
            print_error(f"Sample {i}: Annotation file missing: {anno_path}")
            errors.append(f"anno_missing_{i}")
    
    return len(errors) == 0, errors


# ==============================================================================
# TEST 2: Test Raw Data Loading
# ==============================================================================
def test_raw_data_loading(data_root, train_infos):
    """Test loading raw .pkl files directly."""
    print_section("TEST 2: Raw Data Loading")
    
    sample = train_infos[0]
    lidar_path = sample['lidar_path']
    anno_path = sample['anno_path']
    
    if not os.path.isabs(lidar_path):
        lidar_path = os.path.join(data_root, lidar_path)
    if not os.path.isabs(anno_path):
        anno_path = os.path.join(data_root, anno_path)
    
    errors = []
    
    print_info(f"Loading LiDAR: {lidar_path}")
    try:
        lidar_df = pd.read_pickle(lidar_path)
        print_success(f"LiDAR DataFrame shape: {lidar_df.shape}")
        print_info(f"  Columns: {list(lidar_df.columns)}")
        
        expected_cols = ['x', 'y', 'z', 'i']
        for col in expected_cols:
            if col in lidar_df.columns:
                print_success(f"  Column '{col}' present")
            else:
                print_error(f"  Missing column '{col}'")
                errors.append(f"lidar_missing_col_{col}")
        
        points = lidar_df[['x', 'y', 'z', 'i']].values.astype(np.float32)
        print_success(f"Point cloud array shape: {points.shape}")
        print_info(f"  Min coords: {points.min(axis=0)}")
        print_info(f"  Max coords: {points.max(axis=0)}")
        
    except Exception as e:
        print_error(f"Failed to load LiDAR: {e}")
        errors.append("lidar_load_failed")
        return False, errors
    
    print_info(f"\nLoading annotations: {anno_path}")
    try:
        anno_df = pd.read_pickle(anno_path)
        print_success(f"Annotation DataFrame shape: {anno_df.shape}")
        print_info(f"  Columns: {list(anno_df.columns)}")
        
        required_anno_cols = ['label', 'position.x', 'position.y', 'position.z',
                              'dimensions.x', 'dimensions.y', 'dimensions.z', 'yaw']
        missing_cols = [col for col in required_anno_cols if col not in anno_df.columns]
        
        if missing_cols:
            print_error(f"  Missing annotation columns: {missing_cols}")
            errors.append("anno_missing_columns")
        else:
            print_success("  All required annotation columns present")
        
        if 'label' in anno_df.columns:
            label_counts = anno_df['label'].value_counts()
            print_info(f"\n  Label distribution in this frame:")
            for label, count in label_counts.items():
                print_info(f"    {label}: {count}")
        
    except Exception as e:
        print_error(f"Failed to load annotations: {e}")
        errors.append("anno_load_failed")
        return False, errors
    
    return len(errors) == 0, errors


# ==============================================================================
# TEST 3: Full Pipeline
# ==============================================================================
def test_full_pipeline(data_root):
    """Test complete data pipeline with all transforms."""
    print_section("TEST 3: Full Data Pipeline")
    
    from mmdet3d.datasets import PandaSetDataset
    
    test_pipeline = [
        dict(type='LoadPandaSetPointsFromPKL', coord_type='LIDAR', load_dim=4, use_dim=4),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
        dict(type='PointsRangeFilter', point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
        dict(type='ObjectRangeFilter', point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
        dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]
    
    errors = []
    
    print_info("Creating dataset with pipeline...")
    try:
        dataset = PandaSetDataset(
            data_root=data_root,
            ann_file='pandaset_infos_train.pkl',
            pipeline=test_pipeline,
            data_prefix=dict(pts='', img='', sweeps=''),
            test_mode=False,
            modality=dict(use_lidar=True, use_camera=False),
            box_type_3d='LiDAR',
            filter_empty_gt=False
        )
        print_success(f"Dataset with pipeline created: {len(dataset)} samples")
    except Exception as e:
        print_error(f"Failed to create dataset with pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False, ["pipeline_dataset_failed"]
    
    print_info("\nLoading sample through full pipeline...")
    try:
        sample = dataset[0]
        print_success("Sample loaded successfully!")
        
        print_info("\n  Sample structure:")
        if isinstance(sample, dict):
            for key in sample.keys():
                print_info(f"    '{key}': {type(sample[key])}")
            
            if 'inputs' in sample:
                print_info(f"\n  Inputs:")
                for key, val in sample['inputs'].items():
                    if hasattr(val, 'shape'):
                        print_info(f"    {key}: shape {val.shape}, dtype {val.dtype}")
                    else:
                        print_info(f"    {key}: {type(val)}")
            
            if 'data_samples' in sample:
                ds = sample['data_samples']
                print_info(f"\n  Data samples type: {type(ds)}")
                if hasattr(ds, 'gt_instances_3d'):
                    gt = ds.gt_instances_3d
                    print_info(f"    GT instances:")
                    print_info(f"      bboxes_3d: {gt.bboxes_3d.tensor.shape}")
                    print_info(f"      labels_3d: {gt.labels_3d.shape}")
                    print_info(f"      Number of objects: {len(gt.labels_3d)}")
        
    except Exception as e:
        print_error(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        errors.append("pipeline_execution_failed")
        return False, errors
    
    print_info("\nTesting multiple samples (first 5)...")
    success_count = 0
    for i in range(min(5, len(dataset))):
        try:
            sample = dataset[i]
            success_count += 1
            n_objects = len(sample['data_samples'].gt_instances_3d.labels_3d)
            print_success(f"  Sample {i}: OK ({n_objects} objects)")
        except Exception as e:
            print_error(f"  Sample {i}: Failed - {e}")
            errors.append(f"sample_{i}_failed")
    
    print_info(f"\nSuccess rate: {success_count}/5")
    
    return len(errors) == 0, errors


# ==============================================================================
# TEST 4: DataLoader
# ==============================================================================
def test_dataloader(data_root):
    """Test DataLoader batching."""
    print_section("TEST 4: DataLoader Batching")
    
    from mmdet3d.datasets import PandaSetDataset
    from torch.utils.data import DataLoader
    
    test_pipeline = [
        dict(type='LoadPandaSetPointsFromPKL', coord_type='LIDAR', load_dim=4, use_dim=4),
        dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
        dict(type='PointsRangeFilter', point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
        dict(type='ObjectRangeFilter', point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
        dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
    ]
    
    errors = []
    
    print_info("Creating DataLoader...")
    try:
        dataset = PandaSetDataset(
            data_root=data_root,
            ann_file='pandaset_infos_train.pkl',
            pipeline=test_pipeline,
            data_prefix=dict(pts='', img='', sweeps=''),
            test_mode=False,
            modality=dict(use_lidar=True, use_camera=False),
            box_type_3d='LiDAR',
            filter_empty_gt=True
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda x: x
        )
        
        print_success(f"DataLoader created (batch_size=2)")
        
    except Exception as e:
        print_error(f"DataLoader creation failed: {e}")
        return False, ["dataloader_creation_failed"]
    
    print_info("\nTesting batch iteration...")
    try:
        for i, batch in enumerate(dataloader):
            print_success(f"  Batch {i}: {len(batch)} samples")
            
            if i == 0:
                for j, sample in enumerate(batch):
                    n_points = sample['inputs']['points'].shape[0]
                    n_boxes = len(sample['data_samples'].gt_instances_3d.labels_3d)
                    print_info(f"    Sample {j}: {n_points} points, {n_boxes} objects")
            
            if i >= 2:
                break
        
        print_success("DataLoader iteration successful!")
        
    except Exception as e:
        print_error(f"DataLoader iteration failed: {e}")
        import traceback
        traceback.print_exc()
        errors.append("dataloader_iteration_failed")
    
    return len(errors) == 0, errors


# ==============================================================================
# Main Runner
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='Verify PandaSet integration')
    parser.add_argument('--data-root', type=str, default='data/pandaset',
                        help='Root directory of PandaSet')
    args = parser.parse_args()
    
    print(f"\n{Colors.BOLD}PandaSet MMDetection3D Integration Verification{Colors.END}")
    print(f"Data root: {args.data_root}\n")
    
    # Store results
    results = {}
    
    # Run tests
    tests = [
        ("Info Files", lambda: test_info_files(args.data_root)),
    ]
    
    train_pkl = os.path.join(args.data_root, 'pandaset_infos_train.pkl')
    if os.path.exists(train_pkl):
        train_infos = pickle.load(open(train_pkl, 'rb'))
        tests.append(("Raw Data Loading", lambda: test_raw_data_loading(args.data_root, train_infos)))
    
    tests.extend([
        ("Full Pipeline", lambda: test_full_pipeline(args.data_root)),
        ("DataLoader", lambda: test_dataloader(args.data_root)),
    ])
    
    for test_name, test_func in tests:
        try:
            passed, errors = test_func()
            results[test_name] = (passed, errors)
        except Exception as e:
            print_error(f"Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = (False, ["test_crashed"])
    
    # Summary
    print_section("SUMMARY")
    
    all_passed = True
    for test_name, (passed, errors) in results.items():
        if passed:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")
            print_info(f"  Errors: {errors}")
            all_passed = False
    
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All tests passed! Ready to start training.{Colors.END}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Some tests failed. Fix issues before training.{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())