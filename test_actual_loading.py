#!/usr/bin/env python3
"""
Test actual data loading with the full pipeline to see if objects are really there.
"""

from mmdet3d.datasets import PandaSetDataset

print("=" * 70)
print("Testing Actual Data Loading with Pipeline")
print("=" * 70)

# Create dataset with simple pipeline
test_pipeline = [
    dict(type='LoadPandaSetPointsFromPKL', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

dataset = PandaSetDataset(
    data_root='data/pandaset',
    ann_file='pandaset_infos_train.pkl',
    pipeline=test_pipeline,
    data_prefix=dict(pts='', img='', sweeps=''),
    test_mode=False,
    modality=dict(use_lidar=True, use_camera=False),
    box_type_3d='LiDAR',
    filter_empty_gt=False
)

print(f"\nDataset created with {len(dataset)} samples")
print(f"Classes: {dataset.metainfo['classes']}")

# Load first 5 samples and check objects
print(f"\n{'='*70}")
print("Loading samples through pipeline:")
print(f"{'='*70}")

for i in range(5):
    try:
        sample = dataset[i]
        
        # Get the data
        points = sample['inputs']['points']
        gt_instances = sample['data_samples'].gt_instances_3d
        
        n_points = points.shape[0]
        n_boxes = len(gt_instances.labels_3d)
        
        print(f"\nSample {i}:")
        print(f"  Points: {n_points}")
        print(f"  Boxes: {n_boxes}")
        
        if n_boxes > 0:
            print(f"  Labels: {gt_instances.labels_3d}")
            print(f"  Boxes shape: {gt_instances.bboxes_3d.tensor.shape}")
            
            # Map label indices to class names
            class_names = [dataset.metainfo['classes'][idx] for idx in gt_instances.labels_3d.cpu().numpy()]
            from collections import Counter
            label_counts = Counter(class_names)
            print(f"  Class distribution:")
            for cls_name, count in label_counts.items():
                print(f"    {cls_name}: {count}")
        else:
            print(f"  ⚠ No objects detected!")
            
    except Exception as e:
        print(f"\n✗ Sample {i} failed: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*70}")
print("If you see objects above, the dataset is working correctly!")
print("The '0' in the initial table is just a display issue.")
print(f"{'='*70}")
