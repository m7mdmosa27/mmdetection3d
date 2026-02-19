#!/usr/bin/env python
"""
Training wrapper that catches BEV pool crashes and logs diagnostic info.

Usage:
    python train_with_diagnostics.py [config_path]

This wraps the training to:
1. Catch BEV pool crashes
2. Log detailed info about the failing sample
3. Optionally skip the problematic sample and continue
"""

import os
import sys
import traceback
import numpy as np
import torch

sys.path.insert(0, '.')

from mmengine import Config
from mmengine.runner import Runner
from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS


@HOOKS.register_module(force=True)
class BEVPoolDiagnosticHook(Hook):
    """Hook to diagnose BEV pool issues during training."""
    
    def __init__(self, log_dir='bev_pool_diagnostics'):
        self.log_dir = log_dir
        self.crash_count = 0
        self.sample_stats = []
        os.makedirs(log_dir, exist_ok=True)
        
    def before_train_iter(self, runner, batch_idx, data_batch):
        """Store info about current batch for debugging."""
        self._current_batch = data_batch
        self._current_batch_idx = batch_idx
        
    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        """Check for any issues in the outputs."""
        # Check if loss is valid
        if outputs is not None and 'loss' in outputs:
            loss = outputs['loss']
            if torch.isnan(loss) or torch.isinf(loss):
                self._log_issue(runner, batch_idx, data_batch, "NaN/Inf loss detected")


def patch_bev_pool_with_diagnostics():
    """
    Monkey-patch the BEV pool to add diagnostic logging.
    Call this before training starts.
    """
    try:
        from projects.BEVFusion.bevfusion.ops.bev_pool.bev_pool import QuickCumsumCuda
        
        original_forward = QuickCumsumCuda.forward.__func__
        
        @staticmethod
        def patched_forward(ctx, x, geom_feats, ranks, B, D, H, W):
            # Sort by ranks first (same as original)
            x = x.contiguous()
            geom_feats = geom_feats.contiguous()
            ranks = ranks.contiguous()
            
            # Check for empty input
            if x.shape[0] == 0:
                print(f"[BEV_POOL_DIAG] Empty input tensor! Shape: {x.shape}")
                return torch.zeros(B, x.shape[-1], D, H, W, device=x.device, dtype=x.dtype)
            
            # Run original logic up to interval_starts
            sorted_indices = torch.argsort(ranks)
            x_sorted = x[sorted_indices]
            geom_feats_sorted = geom_feats[sorted_indices]
            ranks_sorted = ranks[sorted_indices]
            
            kept = torch.ones(x_sorted.shape[0], device=x.device, dtype=torch.bool)
            kept[:-1] = ranks_sorted[1:] != ranks_sorted[:-1]
            interval_starts = torch.where(kept)[0].int()
            
            # DIAGNOSTIC: Check for empty intervals
            if interval_starts.numel() == 0:
                print(f"[BEV_POOL_DIAG] Empty interval_starts detected!")
                print(f"  - Input shape: {x.shape}")
                print(f"  - Geom feats shape: {geom_feats.shape}")
                print(f"  - Ranks shape: {ranks.shape}")
                print(f"  - Unique ranks: {torch.unique(ranks).shape[0]}")
                print(f"  - B={B}, D={D}, H={H}, W={W}")
                
                # Log geom_feats statistics
                if geom_feats.shape[0] > 0:
                    print(f"  - Geom feats min: {geom_feats.min(dim=0).values.tolist()}")
                    print(f"  - Geom feats max: {geom_feats.max(dim=0).values.tolist()}")
                
                # Return zeros instead of crashing
                return torch.zeros(B, x.shape[-1], D, H, W, device=x.device, dtype=x.dtype)
            
            # Continue with original forward
            return original_forward(ctx, x_sorted, geom_feats_sorted, ranks_sorted, B, D, H, W)
        
        QuickCumsumCuda.forward = patched_forward
        print("[BEV_POOL_DIAG] BEV pool patched with diagnostics")
        
    except Exception as e:
        print(f"[BEV_POOL_DIAG] Failed to patch BEV pool: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', 
                        default='projects/BEVFusion/configs/bevfusion_pandaset_all_cameras.py',
                        help='Config file')
    parser.add_argument('--work-dir', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    
    print("="*70)
    print(" Training with BEV Pool Diagnostics")
    print("="*70)
    
    # Patch BEV pool before loading config
    patch_bev_pool_with_diagnostics()
    
    # Load config
    cfg = Config.fromfile(args.config)
    
    if args.work_dir:
        cfg.work_dir = args.work_dir
    
    # Add diagnostic hook
    if 'custom_hooks' not in cfg:
        cfg.custom_hooks = []
    
    cfg.custom_hooks.append(dict(type='BEVPoolDiagnosticHook'))
    
    # Build and run
    runner = Runner.from_cfg(cfg)
    
    if args.resume:
        runner.resume()
    
    try:
        runner.train()
    except Exception as e:
        print("\n" + "="*70)
        print(" TRAINING CRASHED")
        print("="*70)
        print(f"Error: {e}")
        traceback.print_exc()
        
        # Try to get more info
        if 'interval_starts' in str(e) or 'index -1' in str(e):
            print("\nThis is a BEV pool empty interval crash.")
            print("The diagnostic patch should have prevented this.")
            print("Check if the patch was applied correctly.")


if __name__ == '__main__':
    main()