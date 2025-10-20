# PandaSet + BEVFusion Integration Guide

This guide documents how to train BEVFusion on PandaSet using images + LiDAR with MMDetection3D.

Contents
- 01_overview.md — What we added and why
- 02_prepare_dataset.md — Folder layout, info files, calibration requirements
- 03_config.md — How the PandaSet BEVFusion config is built and what to change
- 04_build_and_run.md — Environment, building custom ops, and training commands
- 05_troubleshooting.md — Common errors on Windows/conda and how to fix them

Quick Start
1) Prepare infos with image + calibration fields. See 02_prepare_dataset.md
2) Build BEVFusion custom ops. See 04_build_and_run.md
3) Train with the provided config:
   cmd /C "CALL C:\\Users\\<you>\\anaconda3\\Scripts\\activate pandaset && python tools\\train.py projects\\BEVFusion\\configs\\bevfusion_pandaset.py"

Key files in this repo
- projects/BEVFusion/configs/bevfusion_pandaset.py — PandaSet lidar+camera config
- mmdet3d/datasets/pandaset_dataset.py — Dataset with camera + LiDAR support
- projects/BEVFusion/bevfusion/utils.py — Minor IoU fix for GT/pred code dims

