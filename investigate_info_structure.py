#!/usr/bin/env python
"""
Investigate PandaSet info file structure to understand how GT boxes are stored.
"""

import os
import pickle
import numpy as np
from pprint import pprint

DATA_ROOT = 'data/pandaset/'
TRAIN_PKL = os.path.join(DATA_ROOT, 'pandaset_infos_train.pkl')


def print_separator(title=""):
    print("\n" + "="*70)
    if title:
        print(f" {title}")
        print("="*70)


def explore_dict(d, prefix="", max_depth=3, current_depth=0):
    """Recursively explore dictionary structure."""
    if current_depth >= max_depth:
        return
    
    if isinstance(d, dict):
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}: <dict with {len(value)} keys>")
                explore_dict(value, prefix + "  ", max_depth, current_depth + 1)
            elif isinstance(value, (list, tuple)):
                if len(value) > 0:
                    print(f"{prefix}{key}: <{type(value).__name__} of {len(value)} {type(value[0]).__name__}>")
                    if isinstance(value[0], dict) and current_depth < max_depth - 1:
                        print(f"{prefix}  [0]: <dict with keys: {list(value[0].keys())[:5]}...>")
                else:
                    print(f"{prefix}{key}: <empty {type(value).__name__}>")
            elif isinstance(value, np.ndarray):
                print(f"{prefix}{key}: <ndarray shape={value.shape} dtype={value.dtype}>")
            else:
                val_str = str(value)
                if len(val_str) > 50:
                    val_str = val_str[:50] + "..."
                print(f"{prefix}{key}: {val_str} ({type(value).__name__})")
    elif isinstance(d, (list, tuple)) and len(d) > 0:
        print(f"{prefix}[0]: {type(d[0])}")
        if isinstance(d[0], dict):
            explore_dict(d[0], prefix + "  ", max_depth, current_depth + 1)


def main():
    print_separator("Info File Structure Investigation")
    
    if not os.path.exists(TRAIN_PKL):
        print(f"❌ File not found: {TRAIN_PKL}")
        return
    
    with open(TRAIN_PKL, 'rb') as f:
        train_infos = pickle.load(f)
    
    print(f"\nLoaded {len(train_infos)} samples")
    print(f"Type: {type(train_infos)}")
    
    if isinstance(train_infos, dict):
        print(f"Keys: {list(train_infos.keys())}")
        # Check if it's wrapped in 'data_list' or 'infos'
        if 'data_list' in train_infos:
            train_infos = train_infos['data_list']
            print(f"Using 'data_list' key, {len(train_infos)} samples")
        elif 'infos' in train_infos:
            train_infos = train_infos['infos']
            print(f"Using 'infos' key, {len(train_infos)} samples")
    
    # Examine first sample in detail
    print_separator("First Sample Structure")
    
    info = train_infos[0]
    print(f"\nTop-level keys: {list(info.keys())}")
    
    print("\n--- Detailed Structure ---")
    explore_dict(info, max_depth=4)
    
    # Look for GT boxes specifically
    print_separator("Looking for GT Boxes")
    
    # Common locations for GT boxes
    gt_locations = [
        'gt_boxes',
        'gt_bboxes_3d', 
        'ann_info',
        'annos',
        'annotations',
        'instances',
        'gt_instances',
    ]
    
    found_gt = False
    for loc in gt_locations:
        if loc in info:
            print(f"✓ Found '{loc}':")
            value = info[loc]
            if isinstance(value, np.ndarray):
                print(f"  Shape: {value.shape}")
                if len(value) > 0:
                    print(f"  First box: {value[0]}")
                    found_gt = True
            elif isinstance(value, dict):
                print(f"  Keys: {list(value.keys())}")
                if 'gt_boxes_3d' in value:
                    print(f"  gt_boxes_3d shape: {np.array(value['gt_boxes_3d']).shape}")
                    found_gt = True
            elif isinstance(value, list):
                print(f"  Length: {len(value)}")
                if len(value) > 0:
                    print(f"  First item type: {type(value[0])}")
                    if isinstance(value[0], dict):
                        print(f"  First item keys: {list(value[0].keys())}")
                        if 'bbox_3d' in value[0]:
                            print(f"  First bbox_3d: {value[0]['bbox_3d']}")
                            found_gt = True
    
    if not found_gt:
        print("\n❌ No standard GT box location found!")
        print("\nSearching all keys for potential GT data...")
        
        for key in info.keys():
            value = info[key]
            if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] >= 7:
                print(f"  Potential GT in '{key}': shape={value.shape}")
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict) and any(k in value[0] for k in ['bbox', 'box', 'center', 'position']):
                    print(f"  Potential GT in '{key}': list of {len(value)} dicts")
    
    # Check 'instances' structure which is MMDet3D 2.0 format
    print_separator("Checking 'instances' (MMDet3D 2.0 format)")
    
    if 'instances' in info:
        instances = info['instances']
        print(f"Number of instances: {len(instances)}")
        if len(instances) > 0:
            print(f"First instance keys: {list(instances[0].keys())}")
            print(f"First instance: {instances[0]}")
    else:
        print("No 'instances' key found")
    
    # Check 'ann_info' structure
    print_separator("Checking 'ann_info'")
    
    if 'ann_info' in info:
        ann_info = info['ann_info']
        print(f"ann_info keys: {list(ann_info.keys())}")
        for k, v in ann_info.items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: shape={v.shape}")
            elif isinstance(v, list):
                print(f"  {k}: list of {len(v)} items")
            else:
                print(f"  {k}: {type(v)}")
    else:
        print("No 'ann_info' key found")
    
    # Check if GT boxes are in the raw anno file
    print_separator("Checking Raw Annotation File")
    
    sample_idx = info.get('sample_idx', '')
    if '_' in str(sample_idx):
        scene_id, frame_str = str(sample_idx).split('_')
        frame_idx = int(frame_str)
        
        anno_path = os.path.join(DATA_ROOT, scene_id, 'annotations', 'cuboids', f'{frame_idx:02d}.pkl')
        if not os.path.exists(anno_path):
            anno_path = os.path.join(DATA_ROOT, scene_id, 'annotations', 'cuboids', f'{frame_idx}.pkl')
        
        if os.path.exists(anno_path):
            print(f"Loading: {anno_path}")
            with open(anno_path, 'rb') as f:
                raw_anno = pickle.load(f)
            
            print(f"Type: {type(raw_anno)}")
            if hasattr(raw_anno, 'columns'):
                print(f"DataFrame columns: {list(raw_anno.columns)}")
                print(f"Number of rows: {len(raw_anno)}")
                print(f"\nFirst 3 rows:")
                print(raw_anno.head(3))
                
                # Show position statistics
                if 'position.x' in raw_anno.columns:
                    print(f"\nPosition statistics:")
                    print(f"  X: min={raw_anno['position.x'].min():.2f}, max={raw_anno['position.x'].max():.2f}, mean={raw_anno['position.x'].mean():.2f}")
                    print(f"  Y: min={raw_anno['position.y'].min():.2f}, max={raw_anno['position.y'].max():.2f}, mean={raw_anno['position.y'].mean():.2f}")
                    print(f"  Z: min={raw_anno['position.z'].min():.2f}, max={raw_anno['position.z'].max():.2f}, mean={raw_anno['position.z'].mean():.2f}")
        else:
            print(f"Raw annotation file not found: {anno_path}")
    
    # Summary
    print_separator("SUMMARY")
    print("""
Key questions to answer:
1. Are GT boxes being stored in the info file at all?
2. If yes, under which key?
3. Are they in world or ego coordinates?
4. Does the data pipeline properly load them?

Next steps:
1. Check the info file creation script (create_pandaset_infos.py)
2. Verify GT boxes are being extracted from cuboid PKL files
3. Ensure they're being transformed to EGO frame (or do it in the pipeline)
""")


if __name__ == '__main__':
    main()