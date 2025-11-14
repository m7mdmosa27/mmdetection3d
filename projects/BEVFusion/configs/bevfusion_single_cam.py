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
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

# ============================================================
# KEY FIX: Adjust camera parameters for SINGLE-VIEW setup
# ============================================================
# nuScenes has 6 cameras, PandaSet has 1 front camera
# We need to adjust the view transform accordingly

model = dict(
    type='BEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False),
    
    # Image backbone (Swin Transformer - can train from scratch)
    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=True,  # Changed to False - no pretrained weights
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
        )
    ),
    
    # Image neck (FPN)
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)
    ),
    
    # CRITICAL: Adjust view transform for single camera
    # nuScenes: 6 cameras × 80 channels = 480 total
    # PandaSet: 1 camera × 80 channels = 80 total
    # We need to adjust feature_size to match single-view BEV grid
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=80,
        image_size=[256, 704],  # PandaSet front camera size
        feature_size=[32, 88],   # Downsampled feature map size (256/8, 704/8)
        xbound=[-54.0, 54.0, 0.3],   # Match point_cloud_range
        ybound=[-54.0, 54.0, 0.3],   # Match point_cloud_range
        zbound=[-10.0, 10.0, 20.0],  # Vertical range
        dbound=[1.0, 60.0, 0.5],     # Depth range
        downsample=2  # Downsample factor for BEV features
    ),
    
    # CRITICAL: Fusion layer must match single-camera output
    # Input channels: [80 (camera BEV), 256 (LiDAR BEV)]
    fusion_layer=dict(
        type='ConvFuser', 
        in_channels=[80, 256],  # Camera: 80, LiDAR: 256
        out_channels=256
    ),
    
    # Detection head configuration
    bbox_head=dict(
        num_classes=len(class_names),
        common_heads=dict(
            center=[2, 2],   # x, y
            height=[1, 2],   # z
            dim=[3, 2],      # dx, dy, dz
            rot=[2, 2],      # sin(yaw), cos(yaw)
            vel=[2, 2]       # vx, vy (zero for PandaSet)
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
    
    # LiDAR encoder configuration
    pts_voxel_encoder=dict(num_features=4),
    pts_middle_encoder=dict(in_channels=4)
)

# ============================================================
# Data pipelines with single-view camera
# ============================================================

train_pipeline = [
    # Load single front camera image
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args,
        num_views=1  # CRITICAL: Only 1 camera
    ),
    # Load LiDAR points from PKL
    dict(
        type='LoadPandaSetPointsFromPKL',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4
    ),
    # Load annotations
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False
    ),
    # Transform world -> ego coordinates
    dict(
        type='PandaSetWorldToEgo'
    ),
    # Image augmentation for single camera
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.5, 0.6],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True
    ),
    # 3D augmentation
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
        max_epoch=20,
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
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix', 'lidar_aug_matrix', 
            'num_pts_feats'
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
        type='PandaSetWorldToEgo'
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
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats','img_aug_matrix', 'lidar_aug_matrix',
        ]
    )
]

# ============================================================
# Dataloaders
# ============================================================

train_dataloader = dict(
    batch_size=4,  # Reduce if OOM
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts=data_root, img=data_root),
        ann_file='pandaset_infos_train.pkl',
        pipeline=train_pipeline,
        modality=input_modality,
        default_cam_key='FRONT',
        box_type_3d='LiDAR',
        filter_empty_gt=True
    )
)

# Validation dataloader
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,  # Important: clear base config
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts=data_root, img=data_root),
        ann_file='pandaset_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        default_cam_key='FRONT',
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        test_mode=True
    )
)

# Test = validation (same split)
test_dataloader = val_dataloader

# ============================================================
# Training schedule
# ============================================================

# Train from scratch - longer schedule needed
# Training config
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=2)

# Validation config
val_cfg = dict()

# Test config
test_cfg = dict()

# Evaluators - use Indoor3DMetric (simple bbox evaluation)
# val_evaluator = dict(
#     _delete_=True,
#     type='KittiMetric',
#     ann_file=data_root + 'pandaset_infos_val.pkl',
#     metric='bbox',
#     pcd_limit_range=point_cloud_range
# )
# test_evaluator = val_evaluator
# # Simple evaluator that just counts detections (no complex metrics)
# val_evaluator = dict(
#     type='BaseMetric'  # Minimal evaluator, just processes results
# )
# test_evaluator = val_evaluator

val_evaluator = dict(
    _delete_=True,
    type='PandaSetMetric',
    ann_file=data_root + 'pandaset_infos_val.pkl',
    iou_thresholds=[0.25, 0.5, 0.7],  # Easy and hard thresholds
    score_threshold=0.25,  # Minimum confidence score
    prefix='pandaset',
    collect_device = 'gpu'
)
test_evaluator = val_evaluator
# val_cfg = dict()
# test_cfg = dict()

# Optimizer - lower LR for training from scratch
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.01),  # Lower LR
    clip_grad=dict(max_norm=35, norm_type=2)
)

# Learning rate schedule for 20 epochs
param_scheduler = [
    # Warmup for first 1000 iterations
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=1000
    ),
    # Cosine annealing for 20 epochs
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=20,
        end=20,
        by_epoch=True,
        eta_min_ratio=1e-4,
        convert_to_iter_based=True
    ),
    # Momentum scheduler
    dict(
        type='CosineAnnealingMomentum',
        eta_min=0.85 / 0.95,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True
    ),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=1,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True
    )
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1)
)

work_dir = 'work_dirs/bevfusion_pandaset_single_cam'

# Don't load pretrained weights - train from scratch
load_from = None

# Auto scaling LR
auto_scale_lr = dict(enable=False, base_batch_size=16)
