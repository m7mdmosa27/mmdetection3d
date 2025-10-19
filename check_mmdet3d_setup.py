#!/usr/bin/env python3
"""
MMDetection3D Setup Checker for PandaSet Integration

This script checks if all custom components are properly installed and registered.
Run this before the verification script to catch setup issues early.
"""

import os
import sys
from pathlib import Path

class bcolors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"{bcolors.GREEN}✓{bcolors.END} {description}: {filepath}")
        return True
    else:
        print(f"{bcolors.RED}✗{bcolors.END} {description} NOT FOUND: {filepath}")
        return False

def check_string_in_file(filepath, search_string, description):
    """Check if a string exists in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            if search_string in content:
                print(f"{bcolors.GREEN}✓{bcolors.END} {description}")
                return True
            else:
                print(f"{bcolors.RED}✗{bcolors.END} {description}")
                return False
    except Exception as e:
        print(f"{bcolors.RED}✗{bcolors.END} Error reading {filepath}: {e}")
        return False

def main():
    print(f"\n{bcolors.BOLD}{bcolors.BLUE}{'='*70}{bcolors.END}")
    print(f"{bcolors.BOLD}{bcolors.BLUE}MMDetection3D Setup Checker for PandaSet{bcolors.END}")
    print(f"{bcolors.BOLD}{bcolors.BLUE}{'='*70}{bcolors.END}\n")
    
    # Determine mmdet3d path
    try:
        import mmdet3d
        mmdet3d_path = Path(mmdet3d.__file__).parent
        print(f"{bcolors.GREEN}✓{bcolors.END} MMDetection3D found at: {mmdet3d_path}\n")
    except ImportError:
        print(f"{bcolors.RED}✗{bcolors.END} MMDetection3D not installed!")
        print("  Install with: pip install -e .")
        return 1
    
    all_checks_passed = True
    
    # Check 1: PandaSet Dataset File
    print(f"{bcolors.BOLD}[1/6] Checking PandaSet Dataset Class{bcolors.END}")
    dataset_file = mmdet3d_path / 'datasets' / 'pandaset_dataset.py'
    if check_file_exists(dataset_file, "pandaset_dataset.py"):
        check_string_in_file(
            dataset_file,
            '@DATASETS.register_module()',
            "  Has @DATASETS.register_module() decorator"
        )
        check_string_in_file(
            dataset_file,
            'class PandaSetDataset',
            "  Has PandaSetDataset class"
        )
    else:
        all_checks_passed = False
    print()
    
    # Check 2: Dataset Registration in __init__.py
    print(f"{bcolors.BOLD}[2/6] Checking Dataset Registration{bcolors.END}")
    datasets_init = mmdet3d_path / 'datasets' / '__init__.py'
    if check_file_exists(datasets_init, "datasets/__init__.py"):
        check_string_in_file(
            datasets_init,
            'PandaSetDataset',
            "  PandaSetDataset imported in __init__.py"
        )
    else:
        all_checks_passed = False
    print()
    
    # Check 3: Custom Point Cloud Loader
    print(f"{bcolors.BOLD}[3/6] Checking Custom Point Cloud Loader{bcolors.END}")
    loading_file = mmdet3d_path / 'datasets' / 'transforms' / 'loading.py'
    if check_file_exists(loading_file, "transforms/loading.py"):
        check_string_in_file(
            loading_file,
            'LoadPandaSetPointsFromPKL',
            "  Has LoadPandaSetPointsFromPKL class"
        )
        check_string_in_file(
            loading_file,
            '@TRANSFORMS.register_module()',
            "  Has @TRANSFORMS.register_module() decorator"
        )
    else:
        all_checks_passed = False
    print()
    
    # Check 4: Transform Registration in __init__.py
    print(f"{bcolors.BOLD}[4/6] Checking Transform Registration{bcolors.END}")
    transforms_init = mmdet3d_path / 'datasets' / 'transforms' / '__init__.py'
    if check_file_exists(transforms_init, "transforms/__init__.py"):
        check_string_in_file(
            transforms_init,
            'LoadPandaSetPointsFromPKL',
            "  LoadPandaSetPointsFromPKL imported in __init__.py"
        )
    else:
        all_checks_passed = False
    print()
    
    # Check 5: Import Test
    print(f"{bcolors.BOLD}[5/6] Testing Imports{bcolors.END}")
    try:
        from mmdet3d.datasets import PandaSetDataset
        print(f"{bcolors.GREEN}✓{bcolors.END} Can import PandaSetDataset")
    except ImportError as e:
        print(f"{bcolors.RED}✗{bcolors.END} Cannot import PandaSetDataset: {e}")
        all_checks_passed = False
    
    try:
        from mmdet3d.datasets.transforms import LoadPandaSetPointsFromPKL
        print(f"{bcolors.GREEN}✓{bcolors.END} Can import LoadPandaSetPointsFromPKL")
    except ImportError as e:
        print(f"{bcolors.RED}✗{bcolors.END} Cannot import LoadPandaSetPointsFromPKL: {e}")
        all_checks_passed = False
    print()
    
    # Check 6: Registry Test
    print(f"{bcolors.BOLD}[6/6] Testing Registry{bcolors.END}")
    try:
        # Import first to trigger registration
        from mmdet3d.datasets.transforms import LoadPandaSetPointsFromPKL
        from mmdet3d.datasets import PandaSetDataset
        from mmdet3d.registry import TRANSFORMS, DATASETS
        
        if 'PandaSetDataset' in DATASETS.module_dict:
            print(f"{bcolors.GREEN}✓{bcolors.END} PandaSetDataset registered in DATASETS")
        else:
            print(f"{bcolors.RED}✗{bcolors.END} PandaSetDataset NOT in DATASETS registry")
            all_checks_passed = False
        
        if 'LoadPandaSetPointsFromPKL' in TRANSFORMS.module_dict:
            print(f"{bcolors.GREEN}✓{bcolors.END} LoadPandaSetPointsFromPKL registered in TRANSFORMS")
        else:
            print(f"{bcolors.YELLOW}⚠{bcolors.END} LoadPandaSetPointsFromPKL NOT in TRANSFORMS registry")
            print(f"  (This is OK - lazy loading will register it when needed)")
    except Exception as e:
        print(f"{bcolors.RED}✗{bcolors.END} Registry test failed: {e}")
        all_checks_passed = False
    print()
    
    # Summary
    print(f"{bcolors.BOLD}{'='*70}{bcolors.END}")
    if all_checks_passed:
        print(f"{bcolors.GREEN}{bcolors.BOLD}✓ All checks passed!{bcolors.END}")
        print(f"You can now run: python pandaset_verification.py")
    else:
        print(f"{bcolors.RED}{bcolors.BOLD}✗ Some checks failed!{bcolors.END}")
        print(f"\nNext steps:")
        print(f"1. Make sure pandaset_dataset.py is in mmdet3d/datasets/")
        print(f"2. Make sure LoadPandaSetPointsFromPKL is in mmdet3d/datasets/transforms/loading.py")
        print(f"3. Add imports to both __init__.py files")
        print(f"4. Run this script again")
    print(f"{bcolors.BOLD}{'='*70}{bcolors.END}\n")
    
    return 0 if all_checks_passed else 1

if __name__ == '__main__':
    sys.exit(main())
