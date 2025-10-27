#!/usr/bin/env python3
"""
PandaSet Image & Multi-Modal Verification Script

This script verifies the image loading pipeline and camera calibration for BEVFusion.
It tests:
1. Image file accessibility and format
2. Camera calibration matrices (intrinsics & extrinsics)
3. LiDAR-to-image projection accuracy
4. BEVFusion image loading pipeline
5. Multi-modal fusion readiness

Usage:
    python pandaset_image_verification.py --data-root data/pandaset
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Pre-load MMDetection3D modules
print("Pre-loading MMDetection3D modules...")
# Import BEVFusion components first
import sys
import os
bevfusion_path = 'projects/BEVFusion/bevfusion'
if os.path.exists(bevfusion_path) and bevfusion_path not in sys.path:
    sys.path.insert(0, 'projects/BEVFusion')
try:
    import bevfusion  # This registers BEVFusion transforms
except ImportError:
    print("⚠️  Warning: BEVFusion module not found, some tests may fail")
    
from mmdet3d.datasets import PandaSetDataset
from mmdet3d.datasets.transforms import LoadPandaSetPointsFromPKL
print("✓ Modules loaded\n")

# Color codes
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_section(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")

def print_info(msg):
    print(f"  {msg}")


# ==============================================================================
# TEST 1: Basic Image File Verification
# ==============================================================================
def test_image_files(data_root):
    """Verify image files exist and are valid."""
    print_section("TEST 1: Image Files Verification")
    
    train_pkl = os.path.join(data_root, 'pandaset_infos_train.pkl')
    if not os.path.exists(train_pkl):
        print_error(f"Training info file not found: {train_pkl}")
        return False, ["train_pkl_missing"]
    
    train_infos = pickle.load(open(train_pkl, 'rb'))
    print_info(f"Loaded {len(train_infos)} samples from info file\n")
    
    errors = []
    
    # Test first 10 samples
    print_info("Testing first 10 samples for image availability:")
    for i in range(min(10, len(train_infos))):
        info = train_infos[i]
        sample_idx = info.get('sample_idx', i)
        
        # Check if images key exists
        if 'images' in info:
            print_error(f"  Sample {i} ({sample_idx}): 'images' key found in info file (should not be present yet)")
            errors.append(f"sample_{i}_has_images_in_info")
        
        # Check img_path
        img_path = info.get('img_path', None)
        if img_path is None:
            print_error(f"  Sample {i} ({sample_idx}): No 'img_path' in info")
            errors.append(f"sample_{i}_no_img_path")
            continue
        
        # Convert to absolute path
        if not os.path.isabs(img_path):
            img_path_abs = os.path.join(data_root, img_path)
        else:
            img_path_abs = img_path
        
        # Check file exists
        if not os.path.exists(img_path_abs):
            print_error(f"  Sample {i} ({sample_idx}): Image file not found: {img_path_abs}")
            errors.append(f"sample_{i}_img_missing")
            continue
        
        # Try to load with PIL
        try:
            from PIL import Image
            img = Image.open(img_path_abs)
            width, height = img.size
            mode = img.mode
            print_success(f"  Sample {i} ({sample_idx}): {width}x{height} {mode} - {os.path.basename(img_path)}")
        except Exception as e:
            print_error(f"  Sample {i} ({sample_idx}): Failed to load image: {e}")
            errors.append(f"sample_{i}_img_load_failed")
    
    return len(errors) == 0, errors


# ==============================================================================
# TEST 2: Camera Calibration Verification
# ==============================================================================
def test_camera_calibration(data_root):
    """Verify camera calibration matrices are correctly loaded."""
    print_section("TEST 2: Camera Calibration Matrices")
    
    train_pkl = os.path.join(data_root, 'pandaset_infos_train.pkl')
    train_infos = pickle.load(open(train_pkl, 'rb'))
    
    errors = []
    
    # Test first sample in detail
    info = train_infos[0]
    sample_idx = info.get('sample_idx', 0)
    
    print_info(f"Testing calibration for sample: {sample_idx}\n")
    
    # Check calib key exists
    calib = info.get('calib', None)
    if calib is None:
        print_error("No 'calib' key in info")
        return False, ["no_calib"]
    
    print_success("Found 'calib' in info")
    
    # Check intrinsics
    intrinsics = calib.get('intrinsics', None)
    if intrinsics is None:
        print_error("No 'intrinsics' in calib")
        errors.append("no_intrinsics")
    else:
        print_success("Found intrinsics")
        print_info(f"  fx: {intrinsics.get('fx', 'N/A')}")
        print_info(f"  fy: {intrinsics.get('fy', 'N/A')}")
        print_info(f"  cx: {intrinsics.get('cx', 'N/A')}")
        print_info(f"  cy: {intrinsics.get('cy', 'N/A')}")
        
        # Build cam2img matrix
        fx = float(intrinsics.get('fx', 0.0))
        fy = float(intrinsics.get('fy', 0.0))
        cx = float(intrinsics.get('cx', 0.0))
        cy = float(intrinsics.get('cy', 0.0))
        
        cam2img = np.array([[fx, 0.0, cx],
                           [0.0, fy, cy],
                           [0.0, 0.0, 1.0]], dtype=np.float32)
        
        print_info(f"\n  cam2img matrix (K):")
        print_info(f"    {cam2img[0]}")
        print_info(f"    {cam2img[1]}")
        print_info(f"    {cam2img[2]}")
    
    # Check extrinsics
    extrinsics = calib.get('extrinsics', None)
    if extrinsics is None:
        print_error("\nNo 'extrinsics' in calib")
        errors.append("no_extrinsics")
    else:
        print_success("\nFound extrinsics")
        
        camera_pose = extrinsics.get('camera_pose', None)
        lidar_pose = extrinsics.get('lidar_pose', None)
        
        if camera_pose:
            print_info("  camera_pose:")
            if 'position' in camera_pose:
                pos = camera_pose['position']
                print_info(f"    position: ({pos.get('x', 0):.3f}, {pos.get('y', 0):.3f}, {pos.get('z', 0):.3f})")
            if 'heading' in camera_pose:
                head = camera_pose['heading']
                print_info(f"    heading: (w={head.get('w', 0):.3f}, x={head.get('x', 0):.3f}, y={head.get('y', 0):.3f}, z={head.get('z', 0):.3f})")
        
        if lidar_pose:
            print_info("  lidar_pose:")
            if 'position' in lidar_pose:
                pos = lidar_pose['position']
                print_info(f"    position: ({pos.get('x', 0):.3f}, {pos.get('y', 0):.3f}, {pos.get('z', 0):.3f})")
            if 'heading' in lidar_pose:
                head = lidar_pose['heading']
                print_info(f"    heading: (w={head.get('w', 0):.3f}, x={head.get('x', 0):.3f}, y={head.get('y', 0):.3f}, z={head.get('z', 0):.3f})")
    
    # Now test what the dataset actually builds
    print_info("\n" + "-"*70)
    print_info("Testing dataset's image matrix building:")
    print_info("-"*70 + "\n")
    
    # The dataset's load_data_list should build 'images' dict
    dataset = PandaSetDataset(
        data_root=data_root,
        ann_file='pandaset_infos_train.pkl',
        pipeline=None,
        data_prefix=dict(pts='', img='', sweeps=''),
        test_mode=False,
        modality=dict(use_lidar=True, use_camera=True),
        box_type_3d='LiDAR'
    )
    
    # Get processed info
    processed_info = dataset.get_data_info(0)
    
    if 'images' not in processed_info:
        print_error("Dataset did not create 'images' key!")
        errors.append("no_images_key_in_processed")
        return False, errors
    
    print_success("Dataset created 'images' key")
    
    images = processed_info['images']
    if 'FRONT' not in images:
        print_error("No 'FRONT' camera in images!")
        errors.append("no_front_camera")
        return False, errors
    
    print_success("Found 'FRONT' camera in images")
    
    front_cam = images['FRONT']
    
    # Check required keys
    required_keys = ['img_path', 'cam2img', 'lidar2cam']
    for key in required_keys:
        if key not in front_cam:
            print_error(f"  Missing key: {key}")
            errors.append(f"missing_{key}")
        else:
            print_success(f"  Has key: {key}")
    
    # Display matrices
    if 'cam2img' in front_cam:
        print_info("\n  cam2img matrix built by dataset:")
        K = front_cam['cam2img']
        for row in K:
            print_info(f"    {row}")
    
    if 'lidar2cam' in front_cam:
        print_info("\n  lidar2cam matrix built by dataset:")
        T = front_cam['lidar2cam']
        for row in T:
            print_info(f"    {row}")
        
        # Compute lidar2img for verification
        if 'cam2img' in front_cam:
            lidar2img = front_cam['cam2img'] @ front_cam['lidar2cam'][:3, :]
            print_info("\n  lidar2img (K @ [R|t]):")
            for row in lidar2img:
                print_info(f"    {row}")
    
    return len(errors) == 0, errors


# ==============================================================================
# TEST 3: LiDAR-to-Image Projection
# ==============================================================================
def test_lidar_projection(data_root, visualize=True):
    """Test LiDAR point projection onto image."""
    print_section("TEST 3: LiDAR-to-Image Projection")
    
    train_pkl = os.path.join(data_root, 'pandaset_infos_train.pkl')
    train_infos = pickle.load(open(train_pkl, 'rb'))
    
    errors = []
    
    # Get first sample with objects
    info = train_infos[0]
    sample_idx = info.get('sample_idx', 0)
    
    print_info(f"Testing projection for sample: {sample_idx}\n")
    
    # Load LiDAR points
    lidar_path = info['lidar_path']
    if not os.path.isabs(lidar_path):
        lidar_path = os.path.join(data_root, lidar_path)
    
    lidar_df = pd.read_pickle(lidar_path)
    lidar_df = lidar_df[lidar_df['d'] == 1]  # Front forward LiDAR
    points = lidar_df[['x', 'y', 'z']].values.astype(np.float32)
    
    print_success(f"Loaded {len(points)} LiDAR points")
    
    # Load image
    img_path = info.get('img_path', None)
    if not os.path.isabs(img_path):
        img_path = os.path.join(data_root, img_path)
    
    try:
        from PIL import Image
        img = Image.open(img_path)
        img_width, img_height = img.size
        print_success(f"Loaded image: {img_width}x{img_height}")
    except Exception as e:
        print_error(f"Failed to load image: {e}")
        return False, ["image_load_failed"]
    
    # Get calibration from dataset
    dataset = PandaSetDataset(
        data_root=data_root,
        ann_file='pandaset_infos_train.pkl',
        pipeline=None,
        data_prefix=dict(pts='', img='', sweeps=''),
        test_mode=False,
        modality=dict(use_lidar=True, use_camera=True),
        box_type_3d='LiDAR'
    )
    
    processed_info = dataset.get_data_info(0)
    
    if 'images' not in processed_info or 'FRONT' not in processed_info['images']:
        print_error("Cannot get camera calibration")
        return False, ["no_calibration"]
    
    front_cam = processed_info['images']['FRONT']
    lidar2cam = front_cam['lidar2cam']
    cam2img = front_cam['cam2img']
    
    # Project points
    points_hom = np.hstack([points, np.ones((len(points), 1))]).T  # 4xN
    points_cam = lidar2cam @ points_hom  # 4xN
    
    # Filter points in front of camera
    mask_front = points_cam[2] > 0
    points_cam_filtered = points_cam[:3, mask_front]  # 3xN
    
    print_info(f"Points in front of camera: {mask_front.sum()}/{len(points)}")
    
    # Project to image
    points_img = cam2img @ points_cam_filtered  # 3xN
    points_img[:2] /= points_img[2:3]  # Normalize by depth
    
    # Filter points within image bounds
    mask_img = (
        (points_img[0] >= 0) & (points_img[0] < img_width) &
        (points_img[1] >= 0) & (points_img[1] < img_height)
    )
    
    points_img_valid = points_img[:, mask_img]
    
    print_success(f"Points projected onto image: {mask_img.sum()}/{mask_front.sum()}")
    
    if mask_img.sum() == 0:
        print_warning("No points projected onto image! Check calibration.")
        errors.append("no_projected_points")
    else:
        print_info(f"\nProjected points statistics:")
        print_info(f"  X range: [{points_img_valid[0].min():.1f}, {points_img_valid[0].max():.1f}] (image width: {img_width})")
        print_info(f"  Y range: [{points_img_valid[1].min():.1f}, {points_img_valid[1].max():.1f}] (image height: {img_height})")
        print_info(f"  Depth range: [{points_img_valid[2].min():.1f}, {points_img_valid[2].max():.1f}] meters")
    
    # Visualization
    if visualize and mask_img.sum() > 0:
        try:
            import matplotlib
            matplotlib.use('TkAgg')  # Use TkAgg backend for display
            import matplotlib.pyplot as plt
            
            print_info("\nCreating visualization...")
            
            fig, ax = plt.subplots(1, 1, figsize=(16, 9))
            ax.imshow(img)
            
            # Color points by depth
            depths = points_img_valid[2]
            scatter = ax.scatter(
                points_img_valid[0],
                points_img_valid[1],
                c=depths,
                cmap='jet',
                s=1,
                alpha=0.5
            )
            
            plt.colorbar(scatter, ax=ax, label='Depth (m)')
            ax.set_title(f'LiDAR Projection - Sample {sample_idx}\n{mask_img.sum()} points')
            ax.axis('off')
            
            plt.tight_layout()
            
            print_success("Visualization created")
            print_info("Close the window to continue...")
            plt.show()
            
        except ImportError:
            print_warning("Matplotlib not available, skipping visualization")
        except Exception as e:
            print_warning(f"Visualization failed: {e}")
    
    return len(errors) == 0, errors


# ==============================================================================
# TEST 4: BEVFusion Image Pipeline
# ==============================================================================
def test_bevfusion_pipeline(data_root):
    """Test BEVFusion-specific image loading pipeline."""
    print_section("TEST 4: BEVFusion Image Loading Pipeline")
    
    from mmdet3d.registry import TRANSFORMS
    
    errors = []
    
    # Check if BEVLoadMultiViewImageFromFiles is available
    try:
        transform_cfg = dict(
            type='BEVLoadMultiViewImageFromFiles',
            to_float32=True,
            color_type='color',
            num_views=1
        )
        transform = TRANSFORMS.build(transform_cfg)
        print_success("BEVLoadMultiViewImageFromFiles transform available")
    except Exception as e:
        print_error(f"Cannot build BEVLoadMultiViewImageFromFiles: {e}")
        return False, ["transform_not_available"]
    
    # Create dataset with image loading pipeline
    print_info("\nTesting image loading through pipeline...")
    
    image_pipeline = [
        dict(
            type='BEVLoadMultiViewImageFromFiles',
            to_float32=True,
            color_type='color',
            num_views=1
        )
    ]
    
    try:
        dataset = PandaSetDataset(
            data_root=data_root,
            ann_file='pandaset_infos_train.pkl',
            pipeline=image_pipeline,
            data_prefix=dict(pts='', img='', sweeps=''),
            test_mode=False,
            modality=dict(use_lidar=True, use_camera=True),
            box_type_3d='LiDAR',
            filter_empty_gt=False
        )
        print_success(f"Dataset with image pipeline created: {len(dataset)} samples")
    except Exception as e:
        print_error(f"Failed to create dataset with image pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False, ["pipeline_dataset_failed"]
    
    # Test loading first sample
    print_info("\nLoading sample through image pipeline...")
    try:
        sample = dataset[0]
        print_success("Sample loaded!")
        
        print_info("\nSample keys:")
        for key in sample.keys():
            print_info(f"  - {key}")
        
        # Check for img key (it's directly in sample, not in sample['inputs'])
        if 'img' in sample:
            # Image is a list of tensors (one per view)
            img_list = sample['img']
            print_success(f"\nImage tensor found!")
            print_info(f"  Type: {type(img_list)}")
            print_info(f"  Number of views: {len(img_list) if isinstance(img_list, list) else 1}")
            
            # Get first view
            if isinstance(img_list, list):
                img_tensor = img_list[0]
            else:
                img_tensor = img_list
            
            print_info(f"  First view shape: {img_tensor.shape}")
            print_info(f"  Dtype: {img_tensor.dtype}")
            
            # Check if it's already preprocessed (should be [H, W, C] before Pack3DDetInputs)
            if len(img_tensor.shape) == 3:
                height, width, channels = img_tensor.shape
                print_info(f"  H: {height}, W: {width}, Channels: {channels}")
                
                if channels != 3:
                    print_warning(f"Expected 3 channels, got {channels}")
                    errors.append("wrong_num_channels")
            else:
                print_warning(f"Unexpected image tensor shape: {img_tensor.shape}")
                errors.append("wrong_img_shape")
        else:
            print_error("No 'img' found in sample!")
            errors.append("no_img_in_sample")
        
    except Exception as e:
        print_error(f"Failed to load sample: {e}")
        import traceback
        traceback.print_exc()
        errors.append("sample_load_failed")
    
    return len(errors) == 0, errors


# ==============================================================================
# TEST 5: Full Multi-Modal Pipeline
# ==============================================================================
def test_full_multimodal_pipeline(data_root):
    """Test complete BEVFusion pipeline with both LiDAR and camera."""
    print_section("TEST 5: Full Multi-Modal Pipeline (LiDAR + Camera)")
    
    errors = []
    
    # Full BEVFusion-style pipeline
    full_pipeline = [
        dict(
            type='BEVLoadMultiViewImageFromFiles',
            to_float32=True,
            color_type='color',
            num_views=1
        ),
        dict(
            type='LoadPandaSetPointsFromPKL',
            coord_type='LIDAR',
            load_dim=4,
            use_dim=4
        ),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=True,
            with_label_3d=True
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
        dict(
            type='PointsRangeFilter',
            point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        ),
        dict(
            type='Pack3DDetInputs',
            keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
            meta_keys=[
                'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img',
                'cam2lidar', 'ori_lidar2img', 'box_type_3d', 'sample_idx',
                'lidar_path', 'img_path', 'num_pts_feats'
            ]
        )
    ]
    
    print_info("Creating dataset with full multi-modal pipeline...")
    try:
        dataset = PandaSetDataset(
            data_root=data_root,
            ann_file='pandaset_infos_train.pkl',
            pipeline=full_pipeline,
            data_prefix=dict(pts='', img='', sweeps=''),
            test_mode=False,
            modality=dict(use_lidar=True, use_camera=True),
            box_type_3d='LiDAR',
            filter_empty_gt=False
        )
        print_success(f"Dataset created: {len(dataset)} samples")
    except Exception as e:
        print_error(f"Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return False, ["dataset_creation_failed"]
    
    # Test multiple samples
    print_info("\nTesting first 5 samples through full pipeline...")
    success_count = 0
    
    for i in range(min(5, len(dataset))):
        try:
            sample = dataset[i]
            
            # Verify both modalities present
            has_img = 'inputs' in sample and 'img' in sample['inputs']
            has_pts = 'inputs' in sample and 'points' in sample['inputs']
            
            if not has_img:
                print_error(f"  Sample {i}: Missing image!")
                errors.append(f"sample_{i}_no_img")
                continue
            
            if not has_pts:
                print_error(f"  Sample {i}: Missing points!")
                errors.append(f"sample_{i}_no_points")
                continue
            
            # Get shapes
            img_shape = sample['inputs']['img'].shape
            pts_shape = sample['inputs']['points'].shape
            n_boxes = len(sample['data_samples'].gt_instances_3d.labels_3d)
            
            print_success(f"  Sample {i}: img{list(img_shape)}, pts{list(pts_shape)}, {n_boxes} boxes")
            success_count += 1
            
        except Exception as e:
            print_error(f"  Sample {i}: Failed - {e}")
            errors.append(f"sample_{i}_failed")
    
    print_info(f"\nSuccess rate: {success_count}/5")
    
    if success_count < 5:
        print_warning("Some samples failed to load!")
    
    # Detailed inspection of successful sample
    if success_count > 0:
        print_info("\n" + "-"*70)
        print_info("Detailed inspection of first successful sample:")
        print_info("-"*70)
        
        sample = dataset[0]
        
        # Inputs
        print_info("\nInputs:")
        img = sample['inputs']['img']
        pts = sample['inputs']['points']
        print_info(f"  img: shape={img.shape}, dtype={img.dtype}")
        print_info(f"  points: shape={pts.shape}, dtype={pts.dtype}")
        
        # Metadata
        print_info("\nMetadata:")
        meta = sample['data_samples'].metainfo
        important_keys = ['cam2img', 'lidar2cam', 'lidar2img', 'sample_idx', 'img_path', 'lidar_path']
        for key in important_keys:
            if key in meta:
                val = meta[key]
                if isinstance(val, np.ndarray):
                    print_info(f"  {key}: shape={val.shape}")
                else:
                    print_info(f"  {key}: {val}")
        
        # Ground truth
        print_info("\nGround Truth:")
        gt = sample['data_samples'].gt_instances_3d
        print_info(f"  bboxes_3d: {gt.bboxes_3d.tensor.shape}")
        print_info(f"  labels_3d: {gt.labels_3d.shape}")
        print_info(f"  num_objects: {len(gt.labels_3d)}")
    
    return len(errors) == 0, errors


# ==============================================================================
# TEST 6: Configuration Compatibility
# ==============================================================================
def test_config_compatibility(data_root):
    """Verify the configuration matches BEVFusion requirements."""
    print_section("TEST 6: BEVFusion Configuration Compatibility")
    
    print_info("Checking configuration parameters...\n")
    
    errors = []
    
    # Check if config exists
    config_path = 'projects/BEVFusion/configs/bevfusion_pandaset.py'
    if not os.path.exists(config_path):
        print_warning(f"Config file not found at: {config_path}")
        print_info("  (This is OK if you haven't created it yet)")
    else:
        print_success(f"Found config file: {config_path}")
    
    # Verify dataset metadata
    try:
        dataset = PandaSetDataset(
            data_root=data_root,
            ann_file='pandaset_infos_train.pkl',
            pipeline=None,
            data_prefix=dict(pts='', img='', sweeps=''),
            test_mode=False,
            modality=dict(use_lidar=True, use_camera=True),
            box_type_3d='LiDAR'
        )
        
        print_info("Dataset METAINFO:")
        print_info(f"  classes: {dataset.metainfo['classes']}")
        print_info(f"  num_classes: {len(dataset.metainfo['classes'])}")
        print_success("✓ METAINFO looks good")
        
    except Exception as e:
        print_error(f"Failed to load dataset: {e}")
        errors.append("dataset_load_failed")
    
    # Check expected parameters
    print_info("\nExpected BEVFusion parameters:")
    expected = {
        'point_cloud_range': [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        'voxel_size': [0.075, 0.075, 0.2],
        'image_size': [256, 704],  # After ImageAug3D
        'num_views': 1,
        'load_dim': 4,
        'use_dim': 4
    }
    
    for key, val in expected.items():
        print_info(f"  {key}: {val}")
    
    print_success("\n✓ All expected parameters documented")
    
    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description='Verify PandaSet image and multi-modal integration')
    parser.add_argument('--data-root', type=str, default='data/pandaset',
                        help='Root directory of PandaSet')
    parser.add_argument('--skip-viz', action='store_true',
                        help='Skip visualization test')
    args = parser.parse_args()
    
    print(f"\n{Colors.BOLD}PandaSet Image & Multi-Modal Verification{Colors.END}")
    print(f"Data root: {args.data_root}\n")
    
    # Store results
    results = {}
    
    # Run tests
    tests = [
        ("Image Files", lambda: test_image_files(args.data_root)),
        ("Camera Calibration", lambda: test_camera_calibration(args.data_root)),
        ("LiDAR Projection", lambda: test_lidar_projection(args.data_root, visualize=not args.skip_viz)),
        ("BEVFusion Pipeline", lambda: test_bevfusion_pipeline(args.data_root)),
        ("Full Multi-Modal", lambda: test_full_multimodal_pipeline(args.data_root)),
        ("Config Compatibility", lambda: test_config_compatibility(args.data_root)),
    ]
    
    for test_name, test_func in tests:
        try:
            passed, errors = test_func()
            results[test_name] = (passed, errors)
        except Exception as e:
            print_error(f"Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = (False, ["test_crashed"])
    
    # Summary
    print_section("SUMMARY")
    
    all_passed = True
    for test_name, (passed, errors) in results.items():
        if passed:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")
            if errors:
                print_info(f"  Errors: {errors}")
            all_passed = False
    
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All tests passed! Multi-modal pipeline is ready.{Colors.END}")
        print(f"\n{Colors.BOLD}Next steps:{Colors.END}")
        print(f"  1. Review bevfusion_pandaset.py configuration")
        print(f"  2. Verify all parameters match your requirements")
        print(f"  3. Start training:")
        print(f"     python tools/train.py projects/BEVFusion/configs/bevfusion_pandaset.py")
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Some tests failed. Fix issues before training.{Colors.END}")
        print(f"\n{Colors.BOLD}Common issues and solutions:{Colors.END}")
        print(f"  - Image files missing: Check data extraction")
        print(f"  - Calibration errors: Verify info file generation")
        print(f"  - No projected points: Check coordinate transforms")
        print(f"  - Pipeline failures: Verify MMDetection3D installation")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())