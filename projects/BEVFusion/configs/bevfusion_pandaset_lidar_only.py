_base_ = [
    './bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

# Ensure the custom dataset class and BEVFusion components are registered
custom_imports = dict(
    imports=[
        'mmdet3d.datasets.pandaset_dataset',
        'projects.BEVFusion.bevfusion'
    ], allow_failed_imports=False)

# Dataset
dataset_type = 'PandaSetDataset'
data_root = 'data/pandaset/'
class_names = ('Car', 'Pedestrian', 'Pedestrian with Object', 'Temporary Construction Barriers', 'Cones')

# Use ego-relative coordinates (like nuScenes)
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
input_modality = dict(use_lidar=True, use_camera=False)  # LiDAR only for now
backend_args = None

# Model configuration
model = dict(
    bbox_head=dict(
        num_classes=len(class_names),
        common_heads=dict(
            center=[2, 2],
            height=[1, 2],
            dim=[3, 2],
            rot=[2, 2],
            vel=[2, 2]
        ),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=[0.075, 0.075],
            code_size=10
        ),
        train_cfg=dict(
            dataset='PandaSet',
            point_cloud_range=point_cloud_range,
            grid_size=[1440, 1440, 41],
            voxel_size=[0.075, 0.075, 0.2],
            out_size_factor=8,
            gaussian_overlap=0.1,
            min_radius=2,
            pos_weight=-1,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1],
            assign_method='hungarian'
        ),
        test_cfg=dict(
            dataset='PandaSet',
            grid_size=[1440, 1440, 41],
            out_size_factor=8,
            voxel_size=[0.075, 0.075],
            pc_range=point_cloud_range[:2],
            nms_type=None
        )
    ),
    pts_voxel_encoder=dict(num_features=4),
    pts_middle_encoder=dict(in_channels=4)
)

# Pipelines - LiDAR only
train_pipeline = [
    dict(
        type='LoadPandaSetPointsFromPKL',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4
    ),
    dict(
        type='PandaSetWorldToEgo'  # Transform world -> ego coordinates
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False
    ),
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.95, 1.05],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5
    ),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'box_type_3d', 'sample_idx', 'lidar_path', 
            'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'num_pts_feats'
        ]
    )
]

test_pipeline = [
    dict(
        type='LoadPandaSetPointsFromPKL',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4
    ),
    dict(
        type='PandaSetWorldToEgo'
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=['box_type_3d', 'sample_idx', 'lidar_path', 'num_pts_feats']
    )
]

# Dataloaders
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts=data_root),
        ann_file='pandaset_infos_train.pkl',
        pipeline=train_pipeline,
        modality=input_modality,
        box_type_3d='LiDAR',
        filter_empty_gt=True)
)

val_dataloader = None
test_dataloader = None

# Schedule/runtime
train_cfg = dict(by_epoch=True, max_epochs=12, val_interval=0)
val_cfg = None
test_cfg = None
val_evaluator = None
test_evaluator = None

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.33333333, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', begin=0, T_max=12, end=12, by_epoch=True, eta_min_ratio=1e-4, convert_to_iter_based=True)
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1)
)

work_dir = 'work_dirs/bevfusion_pandaset_lidar'
