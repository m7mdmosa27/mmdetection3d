#!/usr/bin/env python3
"""
Check how MMDetection3D transforms are registered and fix the issue.
"""

print("=" * 70)
print("Checking MMDetection3D Transform Registration")
print("=" * 70)

# Step 1: Check what's in mmdet3d.datasets.transforms
print("\n[1] Checking mmdet3d.datasets.transforms module...")
import mmdet3d.datasets.transforms as transforms_module
print(f"Module: {transforms_module}")
print(f"Module file: {transforms_module.__file__}")

# Step 2: Check __init__.py contents
print("\n[2] Checking what's exported from transforms...")
if hasattr(transforms_module, '__all__'):
    print(f"__all__ = {transforms_module.__all__[:10]}...")  # First 10
else:
    print("No __all__ defined")

# Step 3: Try importing specific transforms
print("\n[3] Trying to import LoadAnnotations3D...")
try:
    from mmdet3d.datasets.transforms import LoadAnnotations3D
    print("✓ LoadAnnotations3D imported successfully")
except ImportError as e:
    print(f"✗ Cannot import LoadAnnotations3D: {e}")
    
    # Try to find where it actually is
    print("\n   Searching for LoadAnnotations3D...")
    import mmdet3d.datasets.transforms.loading as loading_module
    if hasattr(loading_module, 'LoadAnnotations3D'):
        print("   → Found in mmdet3d.datasets.transforms.loading")
    else:
        print("   → Not in loading module either")

# Step 4: Check registries after import
print("\n[4] Checking registries after import...")
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmdet3d.registry import TRANSFORMS as MMDET3D_TRANSFORMS

print(f"\nmmengine.registry.TRANSFORMS has {len(MMENGINE_TRANSFORMS.module_dict)} transforms")
print(f"mmdet3d.registry.TRANSFORMS has {len(MMDET3D_TRANSFORMS.module_dict)} transforms")

# Check if LoadAnnotations3D is in either
if 'LoadAnnotations3D' in MMENGINE_TRANSFORMS.module_dict:
    print("✓ LoadAnnotations3D is in mmengine registry")
else:
    print("✗ LoadAnnotations3D is NOT in mmengine registry")
    
if 'LoadAnnotations3D' in MMDET3D_TRANSFORMS.module_dict:
    print("✓ LoadAnnotations3D is in mmdet3d registry")
else:
    print("✗ LoadAnnotations3D is NOT in mmdet3d registry")

# Step 5: List all mmdet3d transforms in mmdet3d registry
print("\n[5] MMDet3D transforms in mmdet3d.registry.TRANSFORMS:")
mmdet3d_only = [k for k in MMDET3D_TRANSFORMS.module_dict.keys() 
                if k not in MMENGINE_TRANSFORMS.module_dict]
print(f"   Found {len(mmdet3d_only)} transforms")
for name in sorted(mmdet3d_only):
    print(f"   - {name}")
# if len(mmdet3d_only) > 15:
#     print(f"   ... and {len(mmdet3d_only) - 15} more")

# Step 6: The solution - show what needs to be imported
print("\n[6] Checking where transforms are defined...")
import os
transforms_dir = os.path.dirname(transforms_module.__file__)
py_files = [f for f in os.listdir(transforms_dir) if f.endswith('.py') and f != '__init__.py']
print(f"   Python files in transforms/: {py_files}")

# Step 7: Try importing all submodules
print("\n[7] Importing all transform submodules...")
for py_file in py_files:
    module_name = py_file[:-3]  # Remove .py
    try:
        exec(f"import mmdet3d.datasets.transforms.{module_name}")
        print(f"   ✓ Imported {module_name}")
    except Exception as e:
        print(f"   ✗ Failed to import {module_name}: {e}")

# Step 8: Check registries again
print("\n[8] Checking registries after importing all submodules...")
print(f"mmengine.registry.TRANSFORMS now has {len(MMENGINE_TRANSFORMS.module_dict)} transforms")
print(f"mmdet3d.registry.TRANSFORMS now has {len(MMDET3D_TRANSFORMS.module_dict)} transforms")

if 'LoadAnnotations3D' in MMENGINE_TRANSFORMS.module_dict:
    print("✓ LoadAnnotations3D is NOW in mmengine registry")
else:
    print("✗ LoadAnnotations3D is STILL NOT in mmengine registry")

print("\n" + "=" * 70)
print("Diagnosis complete")
print("=" * 70)



# Step 9: List all mmengine transforms in mmdet3d registry
print("\n[5] mmengine transforms in mmengine.registry.TRANSFORMS:")
mmengine_only = [k for k in MMENGINE_TRANSFORMS.module_dict.keys() 
                if k not in MMDET3D_TRANSFORMS.module_dict]
print(f"   Found {len(mmengine_only)} transforms")
for name in sorted(mmengine_only)[:15]:
    print(f"   - {name}")
# if len(mmengine_only) > 15:
#     print(f"   ... and {len(mmengine_only) - 15} more")