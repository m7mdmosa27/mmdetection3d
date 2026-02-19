# CUDA Out of Memory at Batch 600 - Solution Guide

## Problem Analysis

The CUDA OOM error happening at batch 600 every time is caused by **memory fragmentation**, **especially after resuming training**. The error message shows:
- `7.22 GiB is reserved by PyTorch but unallocated` - This is fragmented memory
- Trying to allocate `1.95 GiB` fails because there's no contiguous block large enough

## Root Cause

1. **Memory Fragmentation from Resume**: When resuming training, the checkpoint loads:
   - Model state dict
   - **Optimizer state** (AdamW stores momentum and variance for each parameter - this can be several GB)
   - Training state (epoch, iteration, etc.)
   
   This additional memory allocation can cause fragmentation, especially if the previous training session didn't clean up properly.

2. **Why Batch 600?**: 
   - Memory fragmentation accumulates over batches
   - A specific sample at that position may require a larger contiguous allocation
   - The backward pass needs additional memory for gradients
   - **After resume, the memory layout is different and more fragmented**

## Solutions Implemented

### 1. Memory Management Hook (Already Added)
A `MemoryManagementHook` has been added to your config that:
- **Clears CUDA cache immediately after resume** (critical for fixing the issue)
- Periodically clears the CUDA cache every 50 iterations to reduce fragmentation

### 2. Additional Solutions

#### Option A: Set PyTorch CUDA Memory Allocation Config (Recommended)
Before running training, set this environment variable:

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
python tools/train.py projects/BEVFusion/configs/bevfusion_pandaset_all_cameras.py
```

This limits the maximum split size to 512MB, which helps reduce fragmentation.

#### Option B: Reduce Max Voxels (If Option A doesn't work)
Edit the config file and reduce `max_voxels`:

```python
voxelize_cfg=dict(
    max_num_points=10,
    point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
    voxel_size=[0.075, 0.075, 0.2],
    max_voxels=(100000, 140000),  # Reduced from (120000, 160000)
    voxelize_reduce=True
)
```

#### Option C: Use Gradient Checkpointing
Add gradient checkpointing to reduce memory during backward pass (if supported by your model).

## How to Use

1. **Use the Memory Hook** (already configured):
   - The hook is already added to your config
   - It will clear cache immediately after resume (fixes the issue!)
   - It will also clear cache every 50 iterations during training

2. **Set Environment Variable** (Recommended):
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ```

3. **Clear CUDA Cache Before Resuming** (Optional but recommended):
   ```python
   # Add this before your training command or in a script
   import torch
   torch.cuda.empty_cache()
   ```

4. **Run Training with Resume**:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   python tools/train.py projects/BEVFusion/configs/bevfusion_pandaset_all_cameras.py \
       --work-dir work_dirs/first_train_first_all_enhance_with_50_epochs \
       --resume --amp
   ```

## Monitoring

The MemoryManagementHook will log memory stats every 500 iterations (50 * 10) so you can monitor memory usage.

## If Problem Persists

1. Reduce `clear_interval` in the hook (e.g., to 25 or 10)
2. Reduce `max_voxels` further
3. Reduce batch size (currently 1, but you could use gradient accumulation)
4. Check if a specific sample is problematic by examining batch 600
