#!/usr/bin/env python3
"""
Diagnose registry issues between direct import and pipeline building.
"""

print("=" * 70)
print("Registry Diagnosis Script")
print("=" * 70)

# Test 1: Check what registries exist
print("\n[1] Checking available registries...")
from mmdet3d.registry import TRANSFORMS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS

print(f"mmdet3d.registry.TRANSFORMS id: {id(TRANSFORMS)}")
print(f"mmengine.registry.TRANSFORMS id: {id(MMENGINE_TRANSFORMS)}")
print(f"Are they the same object? {TRANSFORMS is MMENGINE_TRANSFORMS}")

# Test 2: Import custom transform
print("\n[2] Importing LoadPandaSetPointsFromPKL...")
from mmdet3d.datasets.transforms.loading import LoadPandaSetPointsFromPKL
print(f"✓ Import successful")

# Test 3: Check registration in both registries
print("\n[3] Checking registration...")
print(f"In mmdet3d.registry.TRANSFORMS? {'LoadPandaSetPointsFromPKL' in TRANSFORMS.module_dict}")
print(f"In mmengine.registry.TRANSFORMS? {'LoadPandaSetPointsFromPKL' in MMENGINE_TRANSFORMS.module_dict}")

# Test 4: Try to build from mmengine registry
print("\n[4] Testing build from mmengine.registry.TRANSFORMS...")
config = dict(
    type='LoadPandaSetPointsFromPKL',
    coord_type='LIDAR',
    load_dim=4,
    use_dim=4
)

try:
    transform = MMENGINE_TRANSFORMS.build(config)
    print(f"✓ Build successful from mmengine registry")
except KeyError as e:
    print(f"✗ Build failed from mmengine registry: {e}")

# Test 5: Try to build from mmdet3d registry
print("\n[5] Testing build from mmdet3d.registry.TRANSFORMS...")
try:
    transform = TRANSFORMS.build(config)
    print(f"✓ Build successful from mmdet3d registry")
except KeyError as e:
    print(f"✗ Build failed from mmdet3d registry: {e}")

# Test 6: Check what Compose uses
print("\n[6] Checking what Compose uses...")
from mmengine.dataset.base_dataset import Compose

import inspect
compose_source = inspect.getsourcefile(Compose)
print(f"Compose source: {compose_source}")

# Look at the __init__ method
compose_init_source = inspect.getsource(Compose.__init__)
if 'from mmengine.registry import TRANSFORMS' in compose_init_source:
    print("⚠ Compose imports TRANSFORMS from mmengine.registry")
elif 'TRANSFORMS' in compose_init_source:
    print("✓ Compose uses TRANSFORMS (need to check which one)")
else:
    print("? Cannot determine which TRANSFORMS Compose uses")

# Test 7: Show first 10 registered transforms in each registry
print("\n[7] Sample of registered transforms:")
print("\nmmdet3d.registry.TRANSFORMS (first 10):")
for i, name in enumerate(sorted(TRANSFORMS.module_dict.keys())[:10]):
    print(f"  {name}")

print("\nmmengine.registry.TRANSFORMS (first 10):")
for i, name in enumerate(sorted(MMENGINE_TRANSFORMS.module_dict.keys())[:10]):
    print(f"  {name}")

print("\n" + "=" * 70)
print("Diagnosis complete")
print("=" * 70)
