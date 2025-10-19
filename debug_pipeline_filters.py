#!/usr/bin/env python3
"""
Debug if pipeline filters are removing all objects.
"""

from mmdet3d.datasets import PandaSetDataset
import numpy as np

print("=" * 70)
print("Debugging Pipeline Filters")
print("=" * 70)

# Test with NO filters first
print("\n[1] Testing WITHOUT filters:")
pipeline_no_filter = [
    dict(type='LoadPandaSetPointsFromPKL', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

dataset = PandaSetDataset(
    data_root='data/pandaset',
    ann_file='pandaset_infos_train.pkl',
    pipeline=pipeline_no_filter,
    data_prefix=dict(pts='', img='', sweeps=''),
    test_mode=False,
    modality=dict(use_lidar=True, use_camera=False),
    box_type_3d='LiDAR',
    filter_empty_gt=False
)

sample = dataset[0]
n_objects_no_filter = len(sample['data_samples'].gt_instances_3d.labels_3d)
print(f"  Objects without filters: {n_objects_no_filter}")

if n_objects_no_filter > 0:
    boxes = sample['data_samples'].gt_instances_3d.bboxes_3d.tensor
    print(f"  Box centers (first 5):")
    for i in range(min(5, len(boxes))):
        print(f"    Box {i}: x={boxes[i][0]:.2f}, y={boxes[i][1]:.2f}, z={boxes[i][2]:.2f}")

# Test with PointsRangeFilter only
print("\n[2] Testing WITH PointsRangeFilter only:")
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
pipeline_points_filter = [
    dict(type='LoadPandaSetPointsFromPKL', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

dataset2 = PandaSetDataset(
    data_root='data/pandaset',
    ann_file='pandaset_infos_train.pkl',
    pipeline=pipeline_points_filter,
    data_prefix=dict(pts='', img='', sweeps=''),
    test_mode=False,
    modality=dict(use_lidar=True, use_camera=False),
    box_type_3d='LiDAR',
    filter_empty_gt=False
)

sample2 = dataset2[0]
n_objects_points_filter = len(sample2['data_samples'].gt_instances_3d.labels_3d)
print(f"  Objects with PointsRangeFilter: {n_objects_points_filter}")

# Test with ObjectRangeFilter only
print("\n[3] Testing WITH ObjectRangeFilter only:")
pipeline_object_filter = [
    dict(type='LoadPandaSetPointsFromPKL', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

dataset3 = PandaSetDataset(
    data_root='data/pandaset',
    ann_file='pandaset_infos_train.pkl',
    pipeline=pipeline_object_filter,
    data_prefix=dict(pts='', img='', sweeps=''),
    test_mode=False,
    modality=dict(use_lidar=True, use_camera=False),
    box_type_3d='LiDAR',
    filter_empty_gt=False
)

sample3 = dataset3[0]
n_objects_object_filter = len(sample3['data_samples'].gt_instances_3d.labels_3d)
print(f"  Objects with ObjectRangeFilter: {n_objects_object_filter}")

# Test with BOTH filters (like verification script)
print("\n[4] Testing WITH BOTH filters (like verification script):")
pipeline_both_filters = [
    dict(type='LoadPandaSetPointsFromPKL', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

dataset4 = PandaSetDataset(
    data_root='data/pandaset',
    ann_file='pandaset_infos_train.pkl',
    pipeline=pipeline_both_filters,
    data_prefix=dict(pts='', img='', sweeps=''),
    test_mode=False,
    modality=dict(use_lidar=True, use_camera=False),
    box_type_3d='LiDAR',
    filter_empty_gt=False
)

sample4 = dataset4[0]
n_objects_both_filters = len(sample4['data_samples'].gt_instances_3d.labels_3d)
print(f"  Objects with both filters: {n_objects_both_filters}")

print(f"\n{'='*70}")
print("Summary:")
print(f"{'='*70}")
print(f"No filters:          {n_objects_no_filter} objects")
print(f"PointsRangeFilter:   {n_objects_points_filter} objects")
print(f"ObjectRangeFilter:   {n_objects_object_filter} objects")
print(f"Both filters:        {n_objects_both_filters} objects")

if n_objects_no_filter > 0 and n_objects_both_filters == 0:
    print(f"\n⚠ WARNING: Filters are removing ALL objects!")
    print(f"  Point cloud range: {point_cloud_range}")
    print(f"  This range might be too restrictive for PandaSet data.")
    print(f"\n  Object centers in sample 0:")
    boxes = sample['data_samples'].gt_instances_3d.bboxes_3d.tensor
    for i in range(min(10, len(boxes))):
        x, y, z = boxes[i][:3].cpu().numpy()
        in_range = (point_cloud_range[0] <= x <= point_cloud_range[3] and
                   point_cloud_range[1] <= y <= point_cloud_range[4] and
                   point_cloud_range[2] <= z <= point_cloud_range[5])
        status = "IN" if in_range else "OUT"
        print(f"    Box {i}: ({x:6.2f}, {y:6.2f}, {z:6.2f}) [{status} OF RANGE]")
