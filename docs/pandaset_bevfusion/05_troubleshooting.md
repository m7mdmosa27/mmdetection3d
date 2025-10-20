# Troubleshooting

Imports and build
- ModuleNotFoundError: mmengine — run under your conda env.
- projects.BEVFusion.bevfusion failed to import — build ops:
  python projects\\BEVFusion\\setup.py develop
- cl/nvcc missing — install MSVC Build Tools and CUDA toolkit compatible with your PyTorch.

Validation config errors
- MMEngine requires val_dataloader/val_cfg/val_evaluator to be all set or all None.
  PandaSet config disables val/test until a metric is added.

Camera paths on Windows
- If you see double prefix in paths like `data/pandaset/data/pandaset/...`, your
  infos already contained absolute/fully-prefixed paths. Current dataset stores
  absolute `img_path` to avoid duplicate joins.

Calibration formats
- `calib.extrinsics.camera_pose` and `lidar_pose` support:
  - 4x4 matrix (list/ndarray)
  - dict with `R` (3x3) and `t` (3)
  - dict with `position` x/y/z and `heading`/`yaw` (scalar or dict with `rad`)

Head/class/channel mismatches
- nuScenes-specific indices (e.g., heatmap channels 8/9) are avoided by setting
  bbox_head `train_cfg/test_cfg.dataset='PandaSet'`.

IoU/box code dimension mismatches
- We use 8-D code (no velocity). IoU is computed on the first 7 dims. If you
  re-enable velocity, set `bbox_coder.code_size=10` and provide vel dims.

Empty GT frames
- Set `filter_empty_gt=True` in the train dataset to skip empty targets.
- Ensure your labels match `METAINFO['classes']` in `pandaset_dataset.py`.

Useful scripts
- check_transforms_registration.py — verifies transform registry wiring
- pandaset_verification.py — quick visualization/tests for sample loading

