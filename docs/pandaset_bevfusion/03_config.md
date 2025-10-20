# Config for BEVFusion on PandaSet

Reference
- Starts from nuScenes BEVFusion lidar+cam: `projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py`
- PandaSet config: `projects/BEVFusion/configs/bevfusion_pandaset.py`

Highlights
- `_base_` inherits the lidar+cam model; we override dataset + head
- `input_modality = dict(use_lidar=True, use_camera=True)`
- Single-view camera loader: `BEVLoadMultiViewImageFromFiles(num_views=1)`
- Set `default_cam_key='FRONT'`
- Head uses 8-D code (no velocity): `bbox_coder.code_size=8`
- Validation/test disabled until a PandaSet evaluator is available

Minimal snippet (see full file for details)
```python
_base_ = ['./bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py']

dataset_type = 'PandaSetDataset'
data_root = 'data/pandaset/'
class_names = ('Car', 'Pedestrian', 'Pickup Truck', 'Semi-truck', 'Cyclist')
input_modality = dict(use_lidar=True, use_camera=True)

model = dict(
    bbox_head=dict(
        num_classes=len(class_names),
        common_heads=dict(center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2]),
        bbox_coder=dict(code_size=8),
        train_cfg=dict(dataset='PandaSet', code_weights=[1.0] * 8),
        test_cfg=dict(dataset='PandaSet')
    ),
    pts_voxel_encoder=dict(num_features=4),
    pts_middle_encoder=dict(in_channels=4)
)

train_pipeline = [
  dict(type='BEVLoadMultiViewImageFromFiles', num_views=1, to_float32=True, color_type='color'),
  dict(type='LoadPandaSetPointsFromPKL', coord_type='LIDAR', load_dim=4, use_dim=4),
  dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
  ...
]

train_dataloader = dict(
  dataset=dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(pts=data_root, img=data_root),
    ann_file='pandaset_infos_train.pkl',
    modality=input_modality,
    default_cam_key='FRONT',
    filter_empty_gt=True,
  )
)

val_dataloader = None
test_dataloader = None
train_cfg = dict(by_epoch=True, max_epochs=12, val_interval=0)
val_cfg = None
test_cfg = None
```

Notes
- To start from nuScenes pretrained weights, set `load_from` to a local checkpoint.
- For multi-view cameras, extend your infos to include multiple camera entries and set `num_views>1`.

