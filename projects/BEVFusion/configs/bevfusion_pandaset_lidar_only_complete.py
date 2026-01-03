# BEVFusion LiDAR-Only Configuration for PandaSet
# Train this first, then use as base for camera fusion
# Complete standalone config with all variables

# ============================================================
# Environment and Basics
# ============================================================
default_scope = 'mmdet3d'
launcher = 'none'
work_dir = './work_dirs/bevfusion_pandaset_lidar_only'

# Custom imports
custom_imports = dict(
    imports=[
        'mmdet3d.datasets.pandaset_dataset',
        'projects.BEVFusion.bevfusion'
    ],
    allow_failed_imports=False
)

# ============================================================
# Dataset Configuration
# ============================================================
dataset_type = 'PandaSetDataset'
data_root='../../../opt/dlami/nvme/dataset/'

# Class names (5 PandaSet classes covering 87.6% of annotations)
class_names = (
    'Car',
    'Pedestrian',
    'Pedestrian with Object',
    'Temporary Construction Barriers',
    'Cones'
)

# Metainfo (used by model)
metainfo = dict(classes=class_names)

# Point cloud range (ego-vehicle centered)
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

# Voxel configuration
voxel_size = [0.075, 0.075, 0.2]
grid_size = [1440, 1440, 41]

# Input modality (LiDAR only for this config)
input_modality = dict(use_lidar=True, use_camera=False)

# Backend args
backend_args = None

# Data prefix (kept for compatibility, similar to nuScenes structure)
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    sweeps='sweeps/LIDAR_TOP',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_BACK='samples/CAM_BACK',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT'
)

# ============================================================
# GT Database Sampling (for data augmentation)
# ============================================================
# Note: Need to generate PandaSet GT database first
# For now, configured but can be disabled by setting rate=0.0
db_sampler = dict(
    type='DataBaseSampler',
    data_root=data_root,
    info_path=data_root + 'pandaset_dbinfos_train.pkl',  # Need to generate this
    rate=0.0,  # DISABLED until database is generated
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            Car=5,
            Pedestrian=5,
            **{'Pedestrian with Object': 5},
            **{'Temporary Construction Barriers': 5},
            Cones=5
        )
    ),
    classes=class_names,
    sample_groups=dict(
        Car=2,
        Pedestrian=2,
        **{'Pedestrian with Object': 2},
        **{'Temporary Construction Barriers': 2},
        Cones=2
    ),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=[0, 1, 2, 3],
        backend_args=backend_args
    )
)

# ============================================================
# Model Configuration (LiDAR-Only)
# ============================================================
model = dict(
    type='BEVFusion',
    
    # Data preprocessor
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxelize_cfg=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(120000, 160000),
            voxelize_reduce=True
        ),
        # Image preprocessing (not used but required by preprocessor)
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        pad_size_divisor=32
    ),
    
    # ========== LiDAR Branch ==========
    pts_voxel_encoder=dict(
        type='HardSimpleVFE',
        num_features=4
    ),
    
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder',  # Use BEVFusion's encoder
        in_channels=4,
        sparse_shape=grid_size,
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [1, 1, 0]), (0, 0)),
        block_type='basicblock'
    ),
    
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)
    ),
    
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True
    ),
    
    # ========== Detection Head ==========
    bbox_head=dict(
        type='TransFusionHead',
        num_proposals=200,
        auxiliary=True,
        in_channels=512,  # 256 * 2 (from neck)
        hidden_channel=128,
        num_classes=len(class_names),
        num_decoder_layers=1,
        num_heads=8,
        nms_kernel_size=3,
        bn_momentum=0.1,
        
        # Decoder layer config
        decoder_layer=dict(
            type='TransformerDecoderLayer',
            self_attn_cfg=dict(
                embed_dims=128,
                num_heads=8,
                dropout=0.1
            ),
            cross_attn_cfg=dict(
                embed_dims=128,
                num_heads=8,
                dropout=0.1
            ),
            ffn_cfg=dict(
                embed_dims=128,
                feedforward_channels=256,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)
            ),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(
                input_channel=2,
                num_pos_feats=128
            )
        ),
        
        # Common heads for predictions
        common_heads=dict(
            center=[2, 2],
            height=[1, 2],
            dim=[3, 2],
            rot=[2, 2],
            vel=[2, 2]
        ),
        
        # Bbox coder
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=10
        ),
        
        # Losses
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0
        ),
        loss_bbox=dict(
            type='mmdet.L1Loss',
            reduction='mean',
            loss_weight=0.25
        ),
        loss_heatmap=dict(
            type='mmdet.GaussianFocalLoss',
            reduction='mean',
            loss_weight=1.0
        ),
        
        # Training config
        train_cfg=dict(
            dataset='PandaSet',
            point_cloud_range=point_cloud_range,
            grid_size=grid_size,
            voxel_size=voxel_size,
            out_size_factor=8,
            gaussian_overlap=0.1,
            min_radius=2,
            pos_weight=-1,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            assign_method='hungarian',
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='mmdet.FocalLossCost', weight=0.15, alpha=0.25, gamma=2.0),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25),
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar')
            )
        ),
        
        # Test config
        test_cfg=dict(
            dataset='PandaSet',
            grid_size=grid_size,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            nms_type=None
        )
    ),
    
    # Train and test config (top level)
    train_cfg=dict(pts=None),
    test_cfg=dict(pts=None)
)

# ============================================================
# Data Pipeline
# ============================================================

# Training pipeline
train_pipeline = [
    # Load LiDAR points
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
    
    # Transform points from world to ego coordinates
    # CRITICAL: Boxes are NOT transformed (already in sensor frame)
    dict(
        type='PandaSetWorldToEgo'
    ),
    
    # Ground truth sampling (disabled until database is generated)
    dict(
        type='ObjectSample',
        db_sampler=db_sampler,
        use_ground_plane=False
    ),
    
    # 3D augmentation
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.95, 1.05],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5
    ),
    
    dict(type='BEVFusionRandomFlip3D'),
    
    # Range filters
    dict(
        type='PointsRangeFilter',
        point_cloud_range=point_cloud_range
    ),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=point_cloud_range
    ),
    
    dict(type='PointShuffle'),
    
    # Pack data
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

# Test pipeline
test_pipeline = [
    dict(
        type='LoadPandaSetPointsFromPKL',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4
    ),
    
    dict(type='PandaSetWorldToEgo'),
    
    dict(
        type='PointsRangeFilter',
        point_cloud_range=point_cloud_range
    ),
    
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=['box_type_3d', 'sample_idx', 'lidar_path', 'num_pts_feats']
    )
]

# ============================================================
# Data Loaders
# ============================================================

# Training dataloader
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='pandaset_infos_train.pkl',
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        modality=input_modality,
        box_type_3d='LiDAR',
        filter_empty_gt=True,
        test_mode=False,
        metainfo=metainfo
    )
)

# Validation dataloader
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='pandaset_infos_val.pkl',
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        modality=input_modality,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        test_mode=True,
        metainfo=metainfo
    )
)

# Test dataloader
test_dataloader = val_dataloader

# ============================================================
# Evaluation
# ============================================================

val_evaluator = dict(
    type='PandaSetMetric',
    data_root=data_root,
    ann_file=data_root + 'pandaset_infos_val.pkl',
    iou_thresholds=[0.25, 0.5, 0.7],
    score_threshold=0.25,
    prefix='pandaset',
    collect_device='gpu'
)

test_evaluator = val_evaluator

# ============================================================
# Training Schedule
# ============================================================

# Train for 20 epochs
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=20,
    val_interval=2
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ============================================================
# Optimization
# ============================================================

# Optimizer
lr = 0.0001
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=lr,
        weight_decay=0.01
    ),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# Learning rate schedule
param_scheduler = [
    # Warmup
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=1000
    ),
    # Cosine annealing
    dict(
        type='CosineAnnealingLR',
        T_max=20,
        eta_min_ratio=1e-4,
        begin=0,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True
    ),
    # Momentum schedulers
    dict(
        type='CosineAnnealingMomentum',
        T_max=8,
        eta_min=0.85 / 0.95,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True
    ),
    dict(
        type='CosineAnnealingMomentum',
        T_max=12,
        eta_min=1.0,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True
    )
]

# ============================================================
# Hooks
# ============================================================

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=5,
        save_best='auto'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook')
)

# Custom hooks (empty for now, can add object sampling later)
custom_hooks = []

# Custom hooks
# Note: DisableObjectSampleHook disabled since db_sampler rate=0.0
# Uncomment when using GT database sampling
custom_hooks = [
    dict(
        type='DisableObjectSampleHook',
        disable_after_epoch=15  # Disable GT sampling after epoch 15
    )
]

# ============================================================
# Runtime
# ============================================================

# Environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

# Visualization
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# Logging
log_processor = dict(
    type='LogProcessor',
    window_size=50,
    by_epoch=True
)

log_level = 'INFO'

# Resume and loading
load_from = None
resume = False

# Auto scale learning rate
auto_scale_lr = dict(
    enable=False,
    base_batch_size=16
)

# Random seed
randomness = dict(seed=0, deterministic=False)