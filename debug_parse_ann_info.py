#!/usr/bin/env python3
"""
Debug why parse_ann_info is returning 0 objects even though annotations exist.
"""

import os
import pickle
import pandas as pd
import numpy as np

# Simulate what the dataset does
data_root = 'data/pandaset'
train_pkl = os.path.join(data_root, 'pandaset_infos_train.pkl')
train_infos = pickle.load(open(train_pkl, 'rb'))

# Define classes (from your updated METAINFO)
classes = ('Car', 'Other Vehicle - Uncommon', 'Construction Signs', 'Cones', 'Pedestrian')

print("=" * 70)
print("Debugging parse_ann_info")
print("=" * 70)
print(f"\nDefined classes: {classes}")

# Test first few samples
for i in range(5):
    info = train_infos[i]
    anno_path = info.get('anno_path', None)
    
    if not os.path.isabs(anno_path):
        anno_path = os.path.join(data_root, anno_path)
    
    print(f"\n{'='*70}")
    print(f"Sample {i}: {info['sample_idx']}")
    print(f"{'='*70}")
    print(f"Anno path: {anno_path}")
    print(f"Exists: {os.path.exists(anno_path)}")
    
    if not os.path.exists(anno_path):
        print("⚠ Annotation file missing!")
        continue
    
    # Load annotations
    annos = pd.read_pickle(anno_path)
    print(f"\nTotal objects in file: {len(annos)}")
    
    if len(annos) == 0:
        print("⚠ No objects in annotation file")
        continue
    
    # Check labels
    print(f"\nLabel distribution:")
    label_counts = annos['label'].value_counts()
    for label, count in label_counts.items():
        in_classes = "✓" if label in classes else "✗"
        print(f"  {in_classes} {label}: {count}")
    
    # Simulate parse_ann_info logic
    print(f"\nSimulating parse_ann_info:")
    matched = 0
    skipped = 0
    
    for idx, obj in annos.iterrows():
        label = obj.get('label', None)
        
        if label not in classes:
            skipped += 1
            continue
        
        matched += 1
        
        # Show first matched object details
        if matched == 1:
            print(f"  First matched object:")
            print(f"    Label: {label}")
            print(f"    Position: ({obj['position.x']:.2f}, {obj['position.y']:.2f}, {obj['position.z']:.2f})")
            print(f"    Dimensions: ({obj['dimensions.x']:.2f}, {obj['dimensions.y']:.2f}, {obj['dimensions.z']:.2f})")
            print(f"    Yaw: {obj['yaw']:.2f}")
    
    print(f"\n  Result: {matched} matched, {skipped} skipped")
    
    if matched == 0:
        print(f"\n⚠ WARNING: 0 objects matched for this sample!")
        print(f"  This means none of the labels in the annotation file match your class list.")

print("\n" + "=" * 70)
print("Analysis complete")
print("=" * 70)

# Check if there's a mismatch between what we expect and what's actually in METAINFO
print("\n" + "=" * 70)
print("Checking actual dataset class usage:")
print("=" * 70)

try:
    from mmdet3d.datasets import PandaSetDataset
    
    # Create a minimal dataset instance
    dataset = PandaSetDataset(
        data_root=data_root,
        ann_file='pandaset_infos_train.pkl',
        pipeline=None,
        data_prefix=dict(pts='', img='', sweeps=''),
        test_mode=False,
        modality=dict(use_lidar=True, use_camera=False),
        box_type_3d='LiDAR'
    )
    
    print(f"\nDataset METAINFO classes: {dataset.metainfo['classes']}")
    print(f"Type: {type(dataset.metainfo['classes'])}")
    
    # Test parse_ann_info on first sample
    info = dataset.get_data_info(0)
    ann_info = dataset.parse_ann_info(info)
    
    print(f"\nFirst sample parse result:")
    print(f"  GT boxes shape: {ann_info['gt_bboxes_3d'].tensor.shape}")
    print(f"  GT labels shape: {ann_info['gt_labels_3d'].shape}")
    print(f"  Number of objects: {len(ann_info['gt_labels_3d'])}")
    
except Exception as e:
    print(f"\n✗ Error loading dataset: {e}")
    import traceback
    traceback.print_exc()
