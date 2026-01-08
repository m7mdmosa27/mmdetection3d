# BEVFusion LiDAR + All 6 Cameras Configuration for PandaSet
# Uses existing BEVLoadMultiViewImageFromFiles transform

_base_ = ['./bevfusion_pandaset_lidar_only_complete.py']

work_dir = './work_dirs/bevfusion_pandaset_all_cameras'
data_root='data/pandaset/'
# ============================================================
# Camera Configuration - ALL 6 CAMERAS
# ============================================================
NUM_CAMERAS = 6
CAMERA_CHANNELS = 80
TOTAL_CAMERA_CHANNELS = NUM_CAMERAS * CAMERA_CHANNELS  # 480

input_modality = dict(use_lidar=True, use_camera=True)

class_names = (
    'Car',
    'Pedestrian',
    'Pedestrian with Object',
    'Temporary Construction Barriers',
    'Cones'
)

metainfo = dict(classes=class_names)

# ============================================================
# Model with 6 cameras
# ============================================================

model = dict(
    type='BEVFusion',
    
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        pad_size_divisor=32,
        voxelize_cfg=dict(
            max_num_points=10,
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            voxel_size=[0.075, 0.075, 0.2],
            max_voxels=(120000, 160000),
            voxelize_reduce=True
        )
    ),
    
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
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
        )
    ),
    
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
    
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=TOTAL_CAMERA_CHANNELS,
        image_size=[192, 512],
        feature_size=[24, 64],
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2
    ),
    
    # 6 cameras (480) + LiDAR (256) = 736
    fusion_layer=dict(
        type='ConvFuser',
        in_channels=[TOTAL_CAMERA_CHANNELS, 256],
        out_channels=256
    ),
    
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder',
        in_channels=4,
        sparse_shape=[1440, 1440, 41],
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
    
    bbox_head=dict(
        type='TransFusionHead',
        num_proposals=200,
        auxiliary=True,
        in_channels=512,
        hidden_channel=128,
        num_classes=5,
        nms_kernel_size=3,
        bn_momentum=0.1,
        num_decoder_layers=1,
        decoder_layer=dict(
            type='TransformerDecoderLayer',
            self_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            cross_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            ffn_cfg=dict(embed_dims=128, feedforward_channels=256, num_fcs=2, ffn_drop=0.1, act_cfg=dict(type='ReLU', inplace=True)),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128)
        ),
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        num_heads=8,
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=[-54.0, -54.0],
            voxel_size=[0.075, 0.075],
            out_size_factor=8,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            code_size=10
        ),
        loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        train_cfg=dict(
            dataset='PandaSet',
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            grid_size=[1440, 1440, 41],
            voxel_size=[0.075, 0.075, 0.2],
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
        test_cfg=dict(
            dataset='PandaSet',
            grid_size=[1440, 1440, 41],
            out_size_factor=8,
            voxel_size=[0.075, 0.075],
            pc_range=[-54.0, -54.0],
            nms_type=None
        )
    ),
    
    train_cfg=dict(pts=None),
    test_cfg=dict(pts=None)
)

# ============================================================
# Data Pipeline - Use existing BEVLoadMultiViewImageFromFiles
# ============================================================

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

train_pipeline = [
    # DEBUG: Before image loading
    # dict(type='DebugImageCount', stage_name='BEFORE_IMG_LOAD'),
    
    # Use existing transform from BEVFusion
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=None,
        num_views=NUM_CAMERAS
    ),
    
    # DEBUG: After image loading
    # dict(type='DebugImageCount', stage_name='AFTER_IMG_LOAD'),
    
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
    
    dict(type='PandaSetWorldToEgo'),
    
    dict(
        type='ObjectSample',
        db_sampler=dict(
            data_root=data_root,
            info_path=data_root + '/pandaset_dbinfos_train.pkl',
            rate=0.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(Car=5, Pedestrian=5, **{'Pedestrian with Object': 5}, **{'Temporary Construction Barriers': 5}, Cones=5)
            ),
            classes=class_names,
            sample_groups=dict(Car=2, Pedestrian=2, **{'Pedestrian with Object': 2}, **{'Temporary Construction Barriers': 2}, Cones=2),
            points_loader=dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=[0, 1, 2, 3], backend_args=None)
        ),
        use_ground_plane=False
    ),
    
    
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.98, 1.02],  # Reduced from [0.95, 1.05]
        rot_range=[-0.3925, 0.3925],     # Reduced from ±45° to ±22.5°
        translation_std=0.1               # Reduced from 0.5 to 0.1
    ),

    # Also reduce ImageAug3D rotation:
    dict(
        type='ImageAug3D',
        final_dim=[192, 512],
        resize_lim=[0.48, 0.52],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-2.7, 2.7],  # Reduced from [-5.4, 5.4]
        rand_flip=True,
        is_train=True
    ),
    
    dict(type='BEVFusionRandomFlip3D'),
    
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    
    dict(type='GridMask', use_h=True, use_w=True, max_epoch=24, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.0, fixed_prob=True),
    
    dict(type='PointShuffle'),
    
    # DEBUG: Before packing
    # dict(type='DebugImageCount', stage_name='BEFORE_PACK'),
    
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'lidar_aug_matrix', 'num_pts_feats'
        ]
    )
]

test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=None,
        num_views=NUM_CAMERAS
    ),
    
    dict(
        type='LoadPandaSetPointsFromPKL',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4
    ),
    
    dict(type='PandaSetWorldToEgo'),
    
    dict(
        type='ImageAug3D',
        final_dim=[192, 512],
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
            'lidar_path', 'img_path', 'num_pts_feats', 'lidar_aug_matrix'
        ]
    )
]

# ============================================================
# Data Loaders
# ============================================================

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='PandaSetDataset',
        data_root='data/pandaset/',
        ann_file='pandaset_infos_train.pkl',
        pipeline=train_pipeline,
        modality=input_modality,
        box_type_3d='LiDAR',
        filter_empty_gt=True,
        test_mode=False,
        metainfo=metainfo,
        data_prefix=dict(pts='', img=''),
        serialize_data=False,  # CRITICAL: Disable serialization to preserve images dict
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='PandaSetDataset',
        data_root=data_root,
        ann_file='pandaset_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        test_mode=True,
        metainfo=metainfo,
        data_prefix=dict(pts='', img=''),
        serialize_data=False,  # CRITICAL: Disable serialization to preserve images dict
    )
)

test_dataloader = val_dataloader

# ============================================================
# Evaluation
# ============================================================

val_evaluator = dict(
    type='PandaSetMetric',
    ann_file=data_root + 'pandaset_infos_val.pkl',
    iou_thresholds=[0.25, 0.5, 0.7],
    score_threshold=0.1,
    prefix='pandaset',
    collect_device='cpu'
)

test_evaluator = val_evaluator

# ============================================================
# Training Schedule
# ============================================================

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

lr = 0.0001
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', T_max=24, eta_min_ratio=1e-4, begin=0, end=24, by_epoch=True)
]

auto_scale_lr = dict(enable=False, base_batch_size=16)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=5, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook')
)

custom_hooks = [dict(type='DisableObjectSampleHook', disable_after_epoch=15)]