# tools/create_pandaset_infos.py
#
# Generates PandaSet info files for MMDetection3D training
# Author: (Your Name)
#
# This script scans PandaSet sequences and creates:
#   - pandaset_infos_train.pkl
#   - pandaset_infos_val.pkl
#
# Each entry contains:
#   {
#     'sample_idx': '001_0000',
#     'lidar_path': '001/lidar/00.pkl',
#     'img_path': '001/camera/front_camera/00.jpg',
#     'anno_path': '001/annotations/cuboids/00.pkl',
#     'calib': { 'intrinsics': {...}, 'extrinsics': {...} }
#   }

import os
import json
import pickle
import random
from tqdm import tqdm


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def build_calib(front_cam_dir, lidar_dir, frame_idx):
    """Build calibration dictionary for one frame."""
    # Load intrinsics (same for all frames)
    intrinsics = load_json(os.path.join(front_cam_dir, 'intrinsics.json'))
    # Load extrinsics (poses.json) per frame
    cam_poses = load_json(os.path.join(front_cam_dir, 'poses.json'))
    lidar_poses = load_json(os.path.join(lidar_dir, 'poses.json'))

    if cam_poses is None or lidar_poses is None or intrinsics is None:
        return None

    # Select corresponding pose by frame index
    if frame_idx < len(cam_poses) and frame_idx < len(lidar_poses):
        cam_pose = cam_poses[frame_idx]
        lidar_pose = lidar_poses[frame_idx]
    else:
        return None

    calib = dict(
        intrinsics=dict(
            fx=intrinsics.get('fx', 0.0),
            fy=intrinsics.get('fy', 0.0),
            cx=intrinsics.get('cx', 0.0),
            cy=intrinsics.get('cy', 0.0),
            D=intrinsics.get('D', None)
        ),
        extrinsics=dict(
            camera_pose=cam_pose,
            lidar_pose=lidar_pose
        )
    )
    return calib


def collect_sequence_info(seq_dir, seq_id):
    """Collect frame-wise information from one sequence."""
    lidar_dir = os.path.join(seq_dir, 'lidar')
    front_cam_dir = os.path.join(seq_dir, 'camera', 'front_camera')
    cuboid_dir = os.path.join(seq_dir, 'annotations', 'cuboids')

    lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith('.pkl')])
    cam_files = sorted([f for f in os.listdir(front_cam_dir) if f.endswith('.jpg')])
    anno_files = sorted([f for f in os.listdir(cuboid_dir) if f.endswith('.pkl')])

    n_frames = min(len(lidar_files), len(cam_files), len(anno_files))
    infos = []

    for i in range(n_frames):
        lidar_name = lidar_files[i]
        cam_name = cam_files[i]
        anno_name = anno_files[i]

        frame_idx = i
        calib = build_calib(front_cam_dir, lidar_dir, frame_idx)

        info = dict(
            sample_idx=f'{seq_id}_{i:04d}',
            lidar_path=os.path.join(seq_id, 'lidar', lidar_name),
            img_path=os.path.join(seq_id, 'camera', 'front_camera', cam_name),
            anno_path=os.path.join(seq_id, 'annotations', 'cuboids', anno_name),
            calib=calib
        )
        infos.append(info)

    return infos


def create_pandaset_infos(root_dir, train_ratio=0.8, save_dir=None):
    """Create info files for all PandaSet sequences."""
    if save_dir is None:
        save_dir = root_dir

    seq_dirs = sorted([d for d in os.listdir(root_dir)
                       if os.path.isdir(os.path.join(root_dir, d))])
    all_infos = []

    print(f"Scanning {len(seq_dirs)} sequences in {root_dir}...")
    for seq_id in tqdm(seq_dirs):
        seq_path = os.path.join(root_dir, seq_id)
        seq_infos = collect_sequence_info(seq_path, seq_id)
        all_infos.extend(seq_infos)

    # Shuffle and split into train/val
    random.shuffle(all_infos)
    n_train = int(len(all_infos) * train_ratio)
    train_infos = all_infos[:n_train]
    val_infos = all_infos[n_train:]

    os.makedirs(save_dir, exist_ok=True)
    train_out = os.path.join(save_dir, 'pandaset_infos_train.pkl')
    val_out = os.path.join(save_dir, 'pandaset_infos_val.pkl')

    with open(train_out, 'wb') as f:
        pickle.dump(train_infos, f)
    with open(val_out, 'wb') as f:
        pickle.dump(val_infos, f)

    print(f"\n✅ Saved {len(train_infos)} train and {len(val_infos)} val samples.")
    print(f"  → {train_out}")
    print(f"  → {val_out}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create PandaSet infos for MMDetection3D.')
    parser.add_argument('--root-dir', type=str, required=True, help='Root directory of PandaSet (contains 001/, 002/, ...)')
    parser.add_argument('--save-dir', type=str, default=None, help='Where to save the .pkl info files')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train/val split ratio')
    args = parser.parse_args()

    create_pandaset_infos(args.root_dir, args.train_ratio, args.save_dir)
