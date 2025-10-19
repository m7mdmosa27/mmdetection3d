#!/usr/bin/env python3
"""
Force rebuild and test the dataset to ensure changes are applied.
"""

import sys
import importlib

# Force reload of the module
if 'mmdet3d.datasets.pandaset_dataset' in sys.modules:
    del sys.modules['mmdet3d.datasets.pandaset_dataset']

if 'mmdet3d.datasets' in sys.modules:
    importlib.reload(sys.modules['mmdet3d.datasets'])

# Now import fresh
from mmdet3d.datasets import PandaSetDataset

print("=" * 70)
print("Force Rebuild Test")
print("=" * 70)

# Check what classes are registered
print(f"\nPandaSetDataset.METAINFO: {PandaSetDataset.METAINFO}")

# Create instance
dataset = PandaSetDataset(
    data_root='data/pandaset',
    ann_file='pandaset_infos_train.pkl',
    pipeline=None,
    data_prefix=dict(pts='', img='', sweeps=''),  # FIX: Need to provide data_prefix
    test_mode=False,
    modality=dict(use_lidar=True, use_camera=False),
    box_type_3d='LiDAR'
)

print(f"\nDataset instance classes: {dataset.metainfo['classes']}")
print(f"Number of samples: {len(dataset)}")

# Test first sample
info = dataset.get_data_info(0)
print(f"\nFirst sample info keys: {info.keys()}")

ann_info = dataset.parse_ann_info(info)
print(f"\nAnnotation parsing result:")
print(f"  Boxes: {ann_info['gt_bboxes_3d'].tensor.shape}")
print(f"  Labels: {ann_info['gt_labels_3d'].shape}")
print(f"  Count: {len(ann_info['gt_labels_3d'])}")

if len(ann_info['gt_labels_3d']) > 0:
    print(f"\n✓ SUCCESS! Objects detected: {len(ann_info['gt_labels_3d'])}")
    print(f"  Label indices: {ann_info['gt_labels_3d']}")
    print(f"  Mapped to classes: {[dataset.metainfo['classes'][i] for i in ann_info['gt_labels_3d']]}")
else:
    print(f"\n✗ FAILED! Still 0 objects")

print("\n" + "=" * 70)