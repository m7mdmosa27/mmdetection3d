# projects/BEVFusion/configs/bevfusion_pandaset.py
#
# BEVFusion model configuration adapted for PandaSet
#
# Note on inheritance:
# - We inherit BEVFusion's nuScenes model config to reuse the model and
#   pipelines. We DO NOT inherit a dataset base here to avoid duplicate
#   top-level keys across bases (MMEngine restriction).

_base_ = [
    '../../../configs/_base_/default_runtime.py',
    # './bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

# Ensure the custom dataset class and BEVFusion model are registered
custom_imports = dict(
    imports=[
        'mmdet3d.datasets.pandaset_dataset',
        'projects.BEVFusion.bevfusion'
    ], allow_failed_imports=True)

# ---------------------------------------------------------------------
# Dataset settings (override nuScenes settings from the model base)
# ---------------------------------------------------------------------
dataset_type = 'PandaSetDataset'
data_root = 'data/pandaset/'
class_names = ('Car', 'Pedestrian', 'Pickup Truck', 'Semi-truck', 'Cyclist')

# ---------------------------------------------------------------------
# Model tweaks for PandaSet classes
# ---------------------------------------------------------------------
model = dict(
    # Use LiDAR-only BEVFusion base to simplify bring-up on PandaSet
    type='BEVFusion',
    bbox_head=dict(num_classes=len(class_names)),
    pts_voxel_encoder=dict(num_features=4),
    pts_middle_encoder=dict(in_channels=4)
)

# If you need PandaSet-specific pipelines, define them here and then
# reference them in dataloaders. Otherwise, the inherited BEVFusion pipelines
# are used; you should ensure they are compatible with your PandaSet data.

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

# Minimal PandaSet pipelines (LiDAR-only to start)
train_pipeline = [
    dict(type='LoadPandaSetPointsFromPKL', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='GlobalRotScaleTrans', rot_range=[-0.785, 0.785],
         scale_ratio_range=[0.95, 1.05], translation_std=[0.5, 0.5, 0.5]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadPandaSetPointsFromPKL', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points'])
]

# Dataloaders for PandaSet infos
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
        metainfo=dict(classes=class_names),
        modality=dict(use_lidar=True, use_camera=False),
        box_type_3d='LiDAR',
        filter_empty_gt=False))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts=data_root),
        ann_file='pandaset_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=dict(classes=class_names),
        modality=dict(use_lidar=True, use_camera=False),
        box_type_3d='LiDAR',
        filter_empty_gt=False))

test_dataloader = val_dataloader

# You may also need to set evaluators compatible with PandaSet, or disable
# validation until a metric is available.

# ---------------------------------------------------------------------
# Training schedule/runtime
# ---------------------------------------------------------------------
train_cfg = dict(max_epochs=24, val_interval=0)

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2)
)

param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=24, eta_min=1e-6, by_epoch=True)
]

default_hooks = dict(
    checkpoint=dict(interval=2, max_keep_ckpts=3),
    logger=dict(interval=50)
)

custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002, update_buffers=True)
]

# Optional: initialize from nuScenes BEVFusion weights
load_from = 'projects/BEVFusion/ckpts/bevfusion_lidar_cam_nuscenes.pth'
