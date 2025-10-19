#!/usr/bin/env python3
"""
Directly test parse_ann_info on the dataset instance.
"""

from mmdet3d.datasets import PandaSetDataset
import os
import pickle

print("=" * 70)
print("Direct parse_ann_info Test")
print("=" * 70)

# Create dataset
dataset = PandaSetDataset(
    data_root='data/pandaset',
    ann_file='pandaset_infos_train.pkl',
    pipeline=None,
    data_prefix=dict(pts='', img='', sweeps=''),
    test_mode=False,
    modality=dict(use_lidar=True, use_camera=False),
    box_type_3d='LiDAR'
)

print(f"\nDataset classes: {dataset.metainfo['classes']}")
print(f"Dataset class type: {type(dataset.metainfo['classes'])}")

# Get first sample info
info = dataset.get_data_info(0)
print(f"\nFirst sample info:")
for key, value in info.items():
    if key != 'lidar_points':
        print(f"  {key}: {value}")

# Check the anno_path
anno_path = info.get('anno_path', None)
print(f"\nAnnotation path from info: {anno_path}")
print(f"Exists: {os.path.exists(anno_path) if anno_path else 'N/A'}")

# Load raw annotations
if anno_path and os.path.exists(anno_path):
    import pandas as pd
    annos = pd.read_pickle(anno_path)
    print(f"\nRaw annotation file:")
    print(f"  Total objects: {len(annos)}")
    print(f"  Labels: {annos['label'].value_counts().to_dict()}")
    
    # Check which labels match
    matched = 0
    for _, obj in annos.iterrows():
        label = obj.get('label', None)
        if label in dataset.metainfo['classes']:
            matched += 1
    print(f"  Objects matching our classes: {matched}")

# Now call parse_ann_info
print(f"\nCalling dataset.parse_ann_info(info)...")
ann_info = dataset.parse_ann_info(info)

print(f"\nResult:")
print(f"  gt_bboxes_3d: {ann_info['gt_bboxes_3d']}")
print(f"  gt_bboxes_3d.tensor.shape: {ann_info['gt_bboxes_3d'].tensor.shape}")
print(f"  gt_labels_3d: {ann_info['gt_labels_3d']}")
print(f"  gt_labels_3d.shape: {ann_info['gt_labels_3d'].shape}")

if len(ann_info['gt_labels_3d']) == 0:
    print(f"\n✗ PROBLEM: parse_ann_info returned 0 objects!")
    print(f"\nLet's trace through parse_ann_info manually...")
    
    # Manually trace through the logic
    anno_path_from_info = info.get('anno_path', None)
    print(f"\n  Step 1: Get anno_path from info")
    print(f"    anno_path = {anno_path_from_info}")
    
    if anno_path_from_info is not None and not os.path.isabs(anno_path_from_info):
        anno_path_abs = os.path.join(dataset.data_root, anno_path_from_info)
        print(f"    Made absolute: {anno_path_abs}")
    else:
        anno_path_abs = anno_path_from_info
    
    print(f"    Exists: {os.path.exists(anno_path_abs) if anno_path_abs else False}")
    
    if anno_path_abs and os.path.exists(anno_path_abs):
        print(f"\n  Step 2: Load annotations")
        annos = pd.read_pickle(anno_path_abs)
        print(f"    Loaded {len(annos)} objects")
        
        print(f"\n  Step 3: Check each object")
        matched_count = 0
        for idx, obj in annos.iterrows():
            label = obj.get('label', None)
            
            # Check exact matching
            is_in_classes = label in dataset.metainfo['classes']
            
            if idx < 3:  # Show first 3
                print(f"    Object {idx}: label='{label}', in_classes={is_in_classes}")
            
            if is_in_classes:
                matched_count += 1
        
        print(f"    Total matched: {matched_count}")
        
        if matched_count > 0:
            print(f"\n  ⚠ Objects SHOULD have been found but weren't!")
            print(f"  Something is wrong with the parse_ann_info logic")
else:
    print(f"\n✓ SUCCESS: Found {len(ann_info['gt_labels_3d'])} objects!")

print("\n" + "=" * 70)
