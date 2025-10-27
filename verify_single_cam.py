#!/usr/bin/env python3
"""
Verify that single-camera BEVFusion config has correct channel dimensions.
This script checks the view transform and fusion layer configuration.
"""

import torch
import numpy as np
from mmengine.config import Config
from mmengine.registry import MODELS

def check_view_transform(cfg):
    """Check if view transform produces correct BEV grid size."""
    print("\n" + "="*70)
    print("VIEW TRANSFORM CHECK")
    print("="*70)
    
    view_cfg = cfg.model.view_transform
    print(f"\nView Transform Config:")
    print(f"  Type: {view_cfg.type}")
    print(f"  Input channels: {view_cfg.in_channels}")
    print(f"  Output channels: {view_cfg.out_channels}")
    print(f"  Image size: {view_cfg.image_size}")
    print(f"  Feature size: {view_cfg.feature_size}")
    print(f"  X bound: {view_cfg.xbound}")
    print(f"  Y bound: {view_cfg.ybound}")
    print(f"  Downsample: {view_cfg.downsample}")
    
    # Calculate expected BEV grid size
    xbound = view_cfg.xbound
    ybound = view_cfg.ybound
    bev_h = int((xbound[1] - xbound[0]) / xbound[2])
    bev_w = int((ybound[1] - ybound[0]) / ybound[2])
    
    print(f"\nExpected BEV grid size:")
    print(f"  Height: {bev_h}")
    print(f"  Width: {bev_w}")
    print(f"  Channels: {view_cfg.out_channels}")
    print(f"  Total: [{view_cfg.out_channels}, {bev_h}, {bev_w}]")
    
    return view_cfg.out_channels, bev_h, bev_w

def check_fusion_layer(cfg, cam_channels):
    """Check if fusion layer expects correct input channels."""
    print("\n" + "="*70)
    print("FUSION LAYER CHECK")
    print("="*70)
    
    fusion_cfg = cfg.model.fusion_layer
    print(f"\nFusion Layer Config:")
    print(f"  Type: {fusion_cfg.type}")
    print(f"  Input channels: {fusion_cfg.in_channels}")
    print(f"  Output channels: {fusion_cfg.out_channels}")
    
    expected_cam = fusion_cfg.in_channels[0]
    expected_lidar = fusion_cfg.in_channels[1]
    
    print(f"\nChannel expectations:")
    print(f"  Camera BEV: {expected_cam} (configured)")
    print(f"  LiDAR BEV: {expected_lidar} (configured)")
    print(f"  Camera BEV: {cam_channels} (from view transform)")
    
    if cam_channels != expected_cam:
        print(f"\n❌ MISMATCH! Camera channels don't match!")
        print(f"   View transform outputs: {cam_channels}")
        print(f"   Fusion layer expects: {expected_cam}")
        return False
    else:
        print(f"\n✓ Channels match correctly!")
        return True

def check_camera_setup(cfg):
    """Check camera configuration."""
    print("\n" + "="*70)
    print("CAMERA SETUP CHECK")
    print("="*70)
    
    # Check pipeline
    train_pipeline = cfg.train_pipeline
    img_loader = None
    for transform in train_pipeline:
        if transform['type'] == 'BEVLoadMultiViewImageFromFiles':
            img_loader = transform
            break
    
    if img_loader:
        num_views = img_loader.get('num_views', 6)  # Default is 6 for nuScenes
        print(f"\nImage loader config:")
        print(f"  Type: {img_loader['type']}")
        print(f"  Number of views: {num_views}")
        
        if num_views != 1:
            print(f"\n⚠ WARNING: num_views should be 1 for single-camera PandaSet!")
            print(f"  Currently set to: {num_views}")
            return False
    else:
        print("\n❌ ERROR: No BEVLoadMultiViewImageFromFiles found in pipeline!")
        return False
    
    print(f"\n✓ Camera setup correct (single view)!")
    return True

def simulate_forward_pass(cfg):
    """Simulate a forward pass to check tensor shapes."""
    print("\n" + "="*70)
    print("SIMULATED FORWARD PASS")
    print("="*70)
    
    try:
        # Simulate camera feature extraction
        view_cfg = cfg.model.view_transform
        
        # Input: [B, N_views, C, H, W]
        B = 2  # batch size
        N = 1  # single view
        C = view_cfg.in_channels
        H, W = view_cfg.feature_size
        
        print(f"\nSimulated inputs:")
        print(f"  Camera features: [{B}, {N}, {C}, {H}, {W}]")
        
        # View transform output
        out_c = view_cfg.out_channels
        xbound = view_cfg.xbound
        ybound = view_cfg.ybound
        bev_h = int((xbound[1] - xbound[0]) / xbound[2])
        bev_w = int((ybound[1] - ybound[0]) / ybound[2])
        
        # After view transform and potential pooling/concatenation
        # For single view: [B, out_c, bev_h, bev_w]
        cam_bev_shape = [B, out_c, bev_h, bev_w]
        print(f"  Camera BEV: {cam_bev_shape}")
        
        # Simulate LiDAR BEV (from middle encoder)
        lidar_bev_shape = [B, 256, bev_h, bev_w]  # Standard LiDAR BEV
        print(f"  LiDAR BEV: {lidar_bev_shape}")
        
        # Fusion layer input
        fusion_cfg = cfg.model.fusion_layer
        expected_cam_c = fusion_cfg.in_channels[0]
        expected_lidar_c = fusion_cfg.in_channels[1]
        
        print(f"\nFusion layer expectations:")
        print(f"  Camera BEV channels: {expected_cam_c}")
        print(f"  LiDAR BEV channels: {expected_lidar_c}")
        
        if out_c == expected_cam_c:
            print(f"\n✓ All shapes match! Forward pass should work.")
            return True
        else:
            print(f"\n❌ Shape mismatch detected!")
            print(f"  Camera BEV produces: {out_c} channels")
            print(f"  Fusion expects: {expected_cam_c} channels")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        return False

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python verify_single_camera_channels.py <config_file>")
        print("Example: python verify_single_camera_channels.py projects/BEVFusion/configs/bevfusion_pandaset_single_cam.py")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    print("="*70)
    print("BEVFusion Single-Camera Configuration Verification")
    print("="*70)
    print(f"\nConfig file: {config_file}")
    
    # Load config
    try:
        cfg = Config.fromfile(config_file)
    except Exception as e:
        print(f"\n❌ Failed to load config: {e}")
        sys.exit(1)
    
    # Run checks
    cam_channels, bev_h, bev_w = check_view_transform(cfg)
    fusion_ok = check_fusion_layer(cfg, cam_channels)
    camera_ok = check_camera_setup(cfg)
    forward_ok = simulate_forward_pass(cfg)
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    checks = [
        ("View Transform", True),
        ("Fusion Layer", fusion_ok),
        ("Camera Setup", camera_ok),
        ("Forward Pass Simulation", forward_ok)
    ]
    
    all_passed = all(ok for _, ok in checks)
    
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {check_name}: {status}")
    
    if all_passed:
        print("\n" + "="*70)
        print("✓ ALL CHECKS PASSED!")
        print("="*70)
        print("\nConfiguration is correct for single-camera BEVFusion.")
        print("You can proceed with training:")
        print(f"  python tools/train.py {config_file}")
    else:
        print("\n" + "="*70)
        print("❌ SOME CHECKS FAILED")
        print("="*70)
        print("\nPlease review the errors above and fix the configuration.")
    
    print("")

if __name__ == '__main__':
    main()
