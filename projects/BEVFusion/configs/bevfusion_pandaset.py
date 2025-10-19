_base_ = [
    './bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

# Ensure the custom dataset class and BEVFusion components are registered
custom_imports = dict(
    imports=[
        'mmdet3d.datasets.pandaset_dataset',
        'projects.BEVFusion.bevfusion'
    ], allow_failed_imports=True)

# Dataset
dataset_type = 'PandaSetDataset'
data_root = 'data/pandaset/'
class_names = ('Car', 'Pedestrian', 'Pickup Truck', 'Semi-truck', 'Cyclist')

# Modalities and ranges
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

# Model: enable image branch + fusion, adjust num_classes
model = dict(
    # inherit all components from base; only override class dims and point dims
    bbox_head=dict(
        num_classes=len(class_names),
        # remove velocity prediction and use 8-d code (no vel)
        common_heads=dict(
            center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2]
        ),
        bbox_coder=dict(code_size=8),
        train_cfg=dict(dataset='PandaSet', code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        test_cfg=dict(dataset='PandaSet')
    ),
    pts_voxel_encoder=dict(num_features=4),
    pts_middle_encoder=dict(in_channels=4)
)

# Pipelines (single-view camera for PandaSet)
train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args,
        num_views=1
    ),
    dict(
        type='LoadPandaSetPointsFromPKL',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False
    ),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.5, 0.6],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True
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
    dict(
        type='GridMask',
        use_h=True,
        use_w=True,
        max_epoch=6,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=0.0,
        fixed_prob=True
    ),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix', 'lidar_aug_matrix', 'num_pts_feats'
        ]
    )
]

test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args,
        num_views=1
    ),
    dict(
        type='LoadPandaSetPointsFromPKL',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4
    ),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.5, 0.5],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=['cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar', 'ori_lidar2img', 'box_type_3d', 'sample_idx', 'lidar_path', 'img_path', 'num_pts_feats']
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
        data_prefix=dict(pts=data_root, img=data_root),
        ann_file='pandaset_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=dict(classes=class_names),
        modality=input_modality,
        default_cam_key='FRONT',
        box_type_3d='LiDAR',
        filter_empty_gt=False)
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

# Optional: initialize from nuScenes BEVFusion lidar+cam weights
# load_from can be set to a nuScenes pretrained model if available
# load_from = 'projects/BEVFusion/ckpts/bevfusion_lidar_cam_nuscenes.pth'
