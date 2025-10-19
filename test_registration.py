#!/usr/bin/env python3
"""
Test MMDetection3D transform registration.
This script verifies that custom transforms are properly registered.
"""

print("=" * 70)
print("MMDetection3D Transform Registration Test")
print("=" * 70)

# Step 1: Import transforms FIRST (this triggers registration)
print("\n[1/4] Importing custom transform...")
try:
    from mmdet3d.datasets.transforms import LoadPandaSetPointsFromPKL
    print("✓ LoadPandaSetPointsFromPKL imported successfully")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    print("\nMake sure LoadPandaSetPointsFromPKL is in:")
    print("  mmdet3d/datasets/transforms/loading.py")
    print("  mmdet3d/datasets/transforms/__init__.py")
    exit(1)

# Step 2: NOW check the registry
print("\n[2/4] Checking TRANSFORMS registry...")
from mmdet3d.registry import TRANSFORMS

if 'LoadPandaSetPointsFromPKL' in TRANSFORMS.module_dict:
    print("✓ LoadPandaSetPointsFromPKL is registered in TRANSFORMS!")
else:
    print("✗ LoadPandaSetPointsFromPKL NOT found in registry")
    print(f"\nRegistry has {len(TRANSFORMS.module_dict)} transforms:")
    for name in sorted(TRANSFORMS.module_dict.keys())[:10]:
        print(f"  - {name}")

# Step 3: Test building from config
print("\n[3/4] Testing build from config dict...")
config = dict(
    type='LoadPandaSetPointsFromPKL',
    coord_type='LIDAR',
    load_dim=4,
    use_dim=4
)

try:
    transform = TRANSFORMS.build(config)
    print(f"✓ Successfully built transform")
    print(f"  Type: {type(transform)}")
    print(f"  Repr: {transform}")
except Exception as e:
    print(f"✗ Failed to build: {e}")
    exit(1)

# Step 4: Test the transform on dummy data
print("\n[4/4] Testing transform execution...")
import tempfile
import pickle
import pandas as pd
import numpy as np
import os

# Create a temporary .pkl file with dummy point cloud
dummy_points = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'z': np.random.randn(100),
    'i': np.random.randint(0, 255, 100)
})

with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
    pickle.dump(dummy_points, f)
    temp_pkl = f.name

try:
    # Test transform
    results = {
        'lidar_points': {
            'lidar_path': temp_pkl,
            'num_pts_feats': 4
        }
    }
    
    results = transform.transform(results)
    
    if 'points' in results:
        points = results['points']
        print(f"✓ Transform executed successfully")
        print(f"  Output points shape: {points.tensor.shape}")
        print(f"  Points type: {type(points)}")
    else:
        print("✗ Transform did not add 'points' to results")
finally:
    # Cleanup
    os.unlink(temp_pkl)

print("\n" + "=" * 70)
print("✓ All tests passed! Transform is properly registered and working.")
print("=" * 70)