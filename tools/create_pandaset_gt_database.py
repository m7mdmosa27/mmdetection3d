# tools/create_pandaset_gt_database.py

import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import mmengine

def create_gt_database(data_root, ann_file, save_path):
    """
    Extract ground truth objects from PandaSet and save to database.
    
    Args:
        data_root: Root directory of PandaSet
        ann_file: Annotation file (pandaset_infos_train.pkl)
        save_path: Output path for database file
    """
    print("Loading annotations...")
    infos = mmengine.load(ann_file)
    
    db_infos = {
        'Car': [],
        'Pedestrian': [],
        'Pedestrian with Object': [],
        'Temporary Construction Barriers': [],
        'Cones': []
    }
    
    print("Extracting ground truth objects...")
    for info_idx, info in enumerate(tqdm(infos)):
        # Load points
        lidar_path = Path(data_root) / info['lidar_path']
        with open(lidar_path, 'rb') as f:
            points_df = pickle.load(f)
        
        # Filter to front LiDAR
        points_df = points_df[points_df['d'] == 1]
        points = points_df[['x', 'y', 'z', 'i']].values
        
        # Load annotations
        anno_path = Path(data_root) / info['anno_path']
        with open(anno_path, 'rb') as f:
            annos = pickle.load(f)
        
        # Filter annotations
        annos = annos[annos['cuboids.sensor_id'] == 1]
        
        # Extract each object
        for _, obj in annos.iterrows():
            label = obj['label']
            if label not in db_infos:
                continue
            
            # Get box parameters
            center = np.array([obj['position.x'], obj['position.y'], obj['position.z']])
            size = np.array([obj['dimensions.x'], obj['dimensions.y'], obj['dimensions.z']])
            yaw = obj['yaw']
            
            # Extract points inside box
            # Simple box filter (can be improved with rotation)
            half_size = size / 2
            mask = (
                (points[:, 0] >= center[0] - half_size[0]) &
                (points[:, 0] <= center[0] + half_size[0]) &
                (points[:, 1] >= center[1] - half_size[1]) &
                (points[:, 1] <= center[1] + half_size[1]) &
                (points[:, 2] >= center[2] - half_size[2]) &
                (points[:, 2] <= center[2] + half_size[2])
            )
            
            obj_points = points[mask].copy()
            
            # Need at least 5 points
            if len(obj_points) < 0:
                continue
            
            # Center points relative to box center
            obj_points[:, :3] -= center
            
            # Store in database
            db_info = {
                'name': label,
                'path': f"gt_database/{label}_{info_idx}_{len(db_infos[label])}.bin",
                'gt_idx': len(db_infos[label]),
                'box3d_lidar': np.concatenate([center, size, [yaw]]),
                'num_points_in_gt': len(obj_points),
                'difficulty': 0
            }
            
            db_infos[label].append(db_info)
            
            # Save points to file
            save_dir = Path(data_root) / 'gt_database' / label
            save_dir.mkdir(parents=True, exist_ok=True)
            obj_points.astype(np.float32).tofile(
                save_dir / f"{info_idx}_{len(db_infos[label])-1}.bin"
            )
    
    # Save database info
    print(f"Saving database to {save_path}...")
    mmengine.dump(db_infos, save_path)
    
    # Print statistics
    print("\nDatabase Statistics:")
    for cls, infos in db_infos.items():
        print(f"  {cls}: {len(infos)} objects")
    
    return db_infos

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze PandaSet labels')
    parser.add_argument('--data-root', type=str, default='data/pandaset',
                        help='Root directory of PandaSet')
    parser.add_argument('--ann_file', type=str, default='pandaset_infos_train.pkl',
                        help='Info file to analyze')
    parser.add_argument('--save_path', type=str, default='pandaset_dbinfos_train.pkl',
                        help='Save path for database file')
    args = parser.parse_args()
    ann_file = args.data_root +'/' + args.ann_file
    save_path = args.data_root +'/' + args.save_path
    create_gt_database(args.data_root, ann_file, save_path)

if __name__ == '__main__':
    main()