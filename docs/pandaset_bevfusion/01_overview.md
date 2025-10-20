# Overview

Goal
- Train BEVFusion on PandaSet using both LiDAR and camera (front) within MMDetection3D.

What we integrated
- Dataset: `mmdet3d/datasets/pandaset_dataset.py`
  - Loads PandaSet info pkl with LiDAR path, image path, and calibration
  - Exposes `images['FRONT']` with `img_path`, `cam2img` (3x3), `lidar2cam` (4x4)
  - Robust parsing for calibration poses (dict or matrix); safe fallbacks

- Config: `projects/BEVFusion/configs/bevfusion_pandaset.py`
  - Inherits the nuScenes BEVFusion lidar+cam base, adapted for PandaSet
  - Single-view camera loader, PandaSet classes, 8‑D code (no velocity)
  - Disables validation/test until a PandaSet evaluator is added

- BEVFusion utils: `projects/BEVFusion/bevfusion/utils.py`
  - Slice IoU inputs to 7‑D (x,y,z,dx,dy,dz,yaw) to avoid dimension mismatches

Why single camera
- PandaSet examples here wire up the front camera only. Extending to multi-view
  requires generating multi-camera info entries and setting `num_views>1`.

Notes
- Windows build for BEVFusion ops requires MSVC Build Tools and matching CUDA.
- If starting from scratch (no checkpoint), `load_from` is disabled by default.

