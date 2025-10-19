# configs/_base_/datasets/pandaset.py
#
# PandaSet dataset configuration for MMDetection3D
# Setup for front_camera + forward LiDAR + cuboids (3D detection)
# Using native PandaSetDataset class and .pkl info files

dataset_type = 'PandaSetDataset'
data_root = 'data/pandaset/'

class_names = ('Car', 'Truck', 'Bus', 'Pedestrian', 'Cyclist', 'Motorcycle')

# --------------------------------------------------------------
# Data processing pipelines
# --------------------------------------------------------------

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4
    ),
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations3D'),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.785, 0.785],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.5, 0.5, 0.5]
    ),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='Pack3DDetInputs')
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4
    ),
    dict(type='LoadImageFromFile'),
    dict(type='Pack3DDetInputs')
]

# --------------------------------------------------------------
# DataLoader settings
# --------------------------------------------------------------

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='pandaset_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=dict(classes=class_names),
        modality=dict(use_lidar=True, use_camera=True),
        box_type_3d='LiDAR'
    ),
    sampler=dict(type='DefaultSampler', shuffle=True)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='pandaset_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=dict(classes=class_names),
        modality=dict(use_lidar=True, use_camera=True),
        box_type_3d='LiDAR'
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)

test_dataloader = val_dataloader

# --------------------------------------------------------------
# Evaluators
# --------------------------------------------------------------
val_evaluator = dict(
    type='CocoMetric3D',
    ann_file=data_root + 'pandaset_infos_val.pkl',
    metric='bbox'
)
test_evaluator = val_evaluator
