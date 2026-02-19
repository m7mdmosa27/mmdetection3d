# Copyright (c) OpenMMLab. All rights reserved.
"""Hook to manage CUDA memory and reduce fragmentation."""

import torch
from mmengine.hooks import Hook
from mmengine.logging import print_log

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class MemoryManagementHook(Hook):
    """Hook to manage CUDA memory and reduce fragmentation.
    
    This hook periodically clears the CUDA cache to prevent memory
    fragmentation that can cause OOM errors at specific batches.
    Especially important when resuming training, as checkpoint loading
    can cause memory fragmentation.
    
    Args:
        clear_interval (int): Clear cache every N iterations. Default: 100.
        empty_cache (bool): Whether to call torch.cuda.empty_cache(). Default: True.
        synchronize (bool): Whether to synchronize before clearing. Default: False.
        clear_after_resume (bool): Clear cache immediately after resume. Default: True.
    """
    
    def __init__(self, clear_interval=100, empty_cache=True, synchronize=False, 
                 clear_after_resume=True):
        self.clear_interval = clear_interval
        self.empty_cache = empty_cache
        self.synchronize = synchronize
        self.clear_after_resume = clear_after_resume
        self._has_resumed = False
    
    def before_run(self, runner):
        """Clear cache after resume to reduce fragmentation from checkpoint loading.
        
        This is called after the runner is initialized and after any checkpoint
        resume has occurred, so we can safely clear the cache here.
        """
        if self.clear_after_resume and torch.cuda.is_available():
            # Check if we resumed by looking at runner state or config
            resumed = False
            if hasattr(runner, 'resume') and runner.resume:
                resumed = True
            elif hasattr(runner, 'cfg') and getattr(runner.cfg, 'resume', False):
                resumed = True
            elif hasattr(runner, '_resume') and runner._resume:
                resumed = True
            
            if resumed or self.clear_after_resume:  # Clear anyway if enabled
                if self.synchronize:
                    torch.cuda.synchronize()
                if self.empty_cache:
                    torch.cuda.empty_cache()
                    status = "after resume" if resumed else "at startup"
                    print_log(
                        f'[MemoryHook] Cleared CUDA cache {status} to reduce fragmentation',
                        logger='current'
                    )
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        print_log(
                            f'[MemoryHook] {status.capitalize()} - Allocated: {allocated:.2f}GB, '
                            f'Reserved: {reserved:.2f}GB',
                            logger='current'
                        )
    
    def after_train_iter(self, runner, batch_idx, data_batch, outputs=None):
        """Clear CUDA cache periodically to reduce fragmentation."""
        if runner.iter % self.clear_interval == 0:
            if self.synchronize:
                torch.cuda.synchronize()
            
            if self.empty_cache:
                torch.cuda.empty_cache()
                
                # Log memory stats if needed
                if runner.iter % (self.clear_interval * 10) == 0:
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        print_log(
                            f'[MemoryHook] Iter {runner.iter}: '
                            f'Allocated: {allocated:.2f}GB, '
                            f'Reserved: {reserved:.2f}GB',
                            logger='current'
                        )
