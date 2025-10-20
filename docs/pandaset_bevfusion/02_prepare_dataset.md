# Prepare PandaSet

Folder structure (example)
```
data/pandaset/
  001/
    lidar/
      00.pkl
      poses.json
    camera/front_camera/
      00.jpg
      intrinsics.json
      poses.json
    annotations/cuboids/
      00.pkl
  002/
    ...
```

Required info files
- Run: `python tools/create_pandaset_infos.py --root-dir data/pandaset`
- Produces: `data/pandaset/pandaset_infos_train.pkl`, `data/pandaset/pandaset_infos_val.pkl`
- Each info item should include:
  - `lidar_path`: relative LiDAR pkl path (e.g., `001/lidar/00.pkl`)
  - `img_path`: relative image path (e.g., `001/camera/front_camera/00.jpg`)
  - `anno_path`: relative cuboids pkl
  - `calib` dict with:
    - `intrinsics`: `fx, fy, cx, cy` (distortion optional)
    - `extrinsics`: per-frame `camera_pose`, `lidar_pose` (4x4 matrix or dict)

Calibration notes
- Intrinsics are converted to 3x3 `cam2img`.
- `lidar2cam = inv(camera_pose) @ lidar_pose` (assumed world_from_sensor inputs).
- Dict pose formats with `R`/`t` or `position`+`heading/yaw` are supported.

Classes
- Update `mmdet3d/datasets/pandaset_dataset.py: METAINFO` to match labels in
  your annotations. The default uses 5 classes:
  `('Car', 'Pedestrian', 'Pickup Truck', 'Semi-truck', 'Cyclist')`.
- Ensure label strings in cuboids pkl match these exactly or add a mapping.

