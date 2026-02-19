#!/usr/bin/env python3
"""
Script to verify that GT loading is working correctly after fixes.
"""

import os
import pickle
import argparse
import mmengine
import numpy as np


def verify_gt_loading(data_root, split='val'):
    """Verify that GT annotations can be loaded correctly."""
    
    # Load info file
    info_file = os.path.join(data_root, f'pandaset_infos_{split}.pkl')
    print(f"\n1. Loading info file: {info_file}")
    
    if not os.path.exists(info_file):
        print(f"   ERROR: Info file not found!")
        return False
    
    infos = mmengine.load(info_file)
    print(f"   ✓ Loaded {len(infos)} info entries")
    
    # Check sample structure
    if len(infos) > 0:
        sample = infos[0]
        print(f"\n2. Sample structure check:")
        print(f"   - sample_idx: {sample.get('sample_idx', 'MISSING')}")
        print(f"   - lidar_path: {sample.get('lidar_path', 'MISSING')}")
        print(f"   - anno_path: {sample.get('anno_path', 'MISSING')}")
        print(f"   - img_path: {sample.get('img_path', 'MISSING')}")
        print(f"   - calib: {'Present' if sample.get('calib') else 'MISSING'}")
    
    # Try to load annotations
    print(f"\n3. Testing annotation loading:")
    loaded = 0
    empty = 0
    failed = 0
    
    class_names = ['Car', 'Pedestrian', 'Pedestrian with Object', 
                   'Temporary Construction Barriers', 'Cones']
    class_counts = {cls: 0 for cls in class_names}
    
    for i, info in enumerate(infos[:10]):  # Test first 10
        sample_idx = info.get('sample_idx', '')
        anno_path = info.get('anno_path', None)
        
        if not anno_path:
            failed += 1
            continue
        
        # Build full path
        if not os.path.isabs(anno_path):
            full_path = os.path.join(data_root, anno_path)
        else:
            full_path = anno_path
        
        if not os.path.exists(full_path):
            print(f"   × Cannot find: {full_path}")
            failed += 1
            continue
        
        try:
            with open(full_path, 'rb') as f:
                annos = pickle.load(f)
            
            # Filter for front LiDAR
            from pandas import concat
            anno1 = annos[annos['cuboids.sensor_id']==1]
            anno2 = annos[annos['camera_used']==0]
            annos = concat([anno1, anno2], ignore_index=True).drop_duplicates().reset_index(drop=True)
            
            # Count objects by class
            n_objs = 0
            for _, obj in annos.iterrows():
                label = obj.get('label', '')
                if label in class_names:
                    class_counts[label] += 1
                    n_objs += 1
            
            if n_objs > 0:
                loaded += 1
                if i < 3:  # Show first 3
                    print(f"   ✓ {sample_idx}: {n_objs} objects")
            else:
                empty += 1
                
        except Exception as e:
            print(f"   × Error loading {sample_idx}: {e}")
            failed += 1
    
    print(f"\n4. Loading summary (first 10 samples):")
    print(f"   - Loaded: {loaded}")
    print(f"   - Empty: {empty}")
    print(f"   - Failed: {failed}")
    
    print(f"\n5. Object distribution:")
    for cls, count in class_counts.items():
        print(f"   - {cls}: {count}")
    
    # Check GT database
    db_file = os.path.join(data_root, 'pandaset_dbinfos_train.pkl')
    if os.path.exists(db_file):
        print(f"\n6. GT Database check:")
        db_infos = mmengine.load(db_file)
        for cls, infos in db_infos.items():
            print(f"   - {cls}: {len(infos)} samples")
    else:
        print(f"\n6. GT Database not found at {db_file}")
    
    return loaded > 0


def main():
    parser = argparse.ArgumentParser(description='Verify GT loading')
    parser.add_argument('--data-root', type=str, default='data/pandaset',
                        help='Root directory of PandaSet')
    parser.add_argument('--split', type=str, default='val',
                        help='Split to verify (train/val)')
    args = parser.parse_args()
    
    success = verify_gt_loading(args.data_root, args.split)
    
    if success:
        print("\n✅ GT loading verification PASSED!")
    else:
        print("\n❌ GT loading verification FAILED!")
        print("\nTroubleshooting steps:")
        print("1. Check that data_root path is correct")
        print("2. Verify annotation files exist in: data_root/<sequence>/annotations/cuboids/*.pkl")
        print("3. Re-run create_pandaset_infos.py to regenerate info files")
        print("4. Check file permissions")


if __name__ == '__main__':
    main()