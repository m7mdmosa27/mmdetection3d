# mmdet3d/datasets/pandaset_dataset.py
#
# PandaSet custom dataset for MMDetection3D
# UPDATED VERSION with support for ALL 6 CAMERAS
# Top 5 classes: Car, Pedestrian, Pedestrian with Object, Temporary Construction Barriers, Cones

import os
import json
import pickle
import numpy as np
import mmengine
from mmdet3d.datasets.det3d_dataset import Det3DDataset
from mmdet3d.registry import DATASETS, TRANSFORMS
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
from mmengine.dataset import Compose
from pandas import concat
import copy
import re


class MMDet3DCompose(Compose):
    """Custom Compose that uses mmdet3d.registry.TRANSFORMS."""
    
    def __init__(self, transforms):
        self.transforms = []
        if isinstance(transforms, dict):
            transforms = [transforms]
        for transform in transforms:
            if isinstance(transform, dict):
                transform = TRANSFORMS.build(transform)
                self.transforms.append(transform)
            else:
                self.transforms.append(transform)


# Camera mapping for PandaSet - use CAM_* keys for BEVFusion compatibility
PANDASET_CAMERAS = {
    'CAM_FRONT': 'front_camera',
    'CAM_FRONT_RIGHT': 'front_right_camera',
    'CAM_FRONT_LEFT': 'front_left_camera',
    'CAM_BACK': 'back_camera',
    'CAM_BACK_LEFT': 'left_camera',
    'CAM_BACK_RIGHT': 'right_camera'
}

CAMERA_ORDER = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def quaternion_to_rotation_matrix(w, x, y, z):
    """Convert quaternion to 3x3 rotation matrix."""
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ], dtype=np.float32)


def build_transform_matrix(position, heading):
    """Build 4x4 transformation matrix from position and quaternion heading."""
    t = np.array([
        position.get('x', 0.0),
        position.get('y', 0.0),
        position.get('z', 0.0)
    ], dtype=np.float32)
    
    w = heading.get('w', 1.0)
    x = heading.get('x', 0.0)
    y = heading.get('y', 0.0)
    z = heading.get('z', 0.0)
    
    R = quaternion_to_rotation_matrix(w, x, y, z)
    
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def extract_scene_and_frame(path):
    """Extract scene_id and frame_idx from a PandaSet path."""
    if path is None:
        return None, None
    
    path = str(path).replace('\\', '/')
    
    scene_match = re.search(r'/(\d{3})/', path)
    if not scene_match:
        scene_match = re.search(r'^(\d{3})/', path)
    
    if not scene_match:
        return None, None
    
    scene_id = scene_match.group(1)
    
    frame_match = re.search(r'/(\d+)\.(pkl|jpg)$', path)
    if not frame_match:
        return scene_id, None
    
    frame_idx = int(frame_match.group(1))
    
    return scene_id, frame_idx


@DATASETS.register_module()
class PandaSetDataset(Det3DDataset):
    """Custom dataset for PandaSet with support for all 6 cameras."""

    METAINFO = {
        'classes': ('Car', 'Pedestrian', 'Pedestrian with Object', 
                   'Temporary Construction Barriers', 'Cones')
    }

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 modality=dict(use_lidar=True, use_camera=True),
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 data_prefix=None,
                 default_cam_key='CAM_FRONT',
                 **kwargs):
        
        self._pipeline_cfg = pipeline
        self.default_cam_key = default_cam_key
        self._debug_count = 0  # For limiting debug output
        
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=None,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            data_prefix=data_prefix,
            **kwargs
        )
        
        if self._pipeline_cfg is not None:
            self.pipeline = MMDet3DCompose(self._pipeline_cfg)

    def _load_camera_data(self, scene_id, frame_idx):
        """Load camera calibration and image paths for ALL 6 cameras."""
        images = {}
        
        scene_dir = os.path.join(self.data_root, scene_id)
        
        if not os.path.isdir(scene_dir):
            return images
        
        # Load LiDAR poses
        lidar_poses_path = os.path.join(scene_dir, 'lidar', 'poses.json')
        if not os.path.exists(lidar_poses_path):
            return images
            
        with open(lidar_poses_path, 'r') as f:
            lidar_poses = json.load(f)
        
        if frame_idx >= len(lidar_poses):
            return images
        
        lidar_pose = lidar_poses[frame_idx]
        
        world_from_ego = build_transform_matrix(
            lidar_pose.get('position', {}),
            lidar_pose.get('heading', {})
        )
        
        # Load each camera in order
        for cam_key in CAMERA_ORDER:
            cam_name = PANDASET_CAMERAS[cam_key]
            cam_dir = os.path.join(scene_dir, 'camera', cam_name)
            
            if not os.path.isdir(cam_dir):
                continue
            
            # Image path
            img_path = os.path.join(cam_dir, f'{frame_idx:02d}.jpg')
            if not os.path.exists(img_path):
                img_path = os.path.join(cam_dir, f'{frame_idx}.jpg')
            if not os.path.exists(img_path):
                continue
            
            # Load intrinsics
            intrinsics_path = os.path.join(cam_dir, 'intrinsics.json')
            if not os.path.exists(intrinsics_path):
                continue
                
            with open(intrinsics_path, 'r') as f:
                intrinsics = json.load(f)
            
            fx = float(intrinsics.get('fx', 0.0))
            fy = float(intrinsics.get('fy', 0.0))
            cx = float(intrinsics.get('cx', 0.0))
            cy = float(intrinsics.get('cy', 0.0))
            
            # 3x3 intrinsic matrix (BEVFusion will convert to 4x4)
            cam2img = np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            
            # Load camera poses
            cam_poses_path = os.path.join(cam_dir, 'poses.json')
            if not os.path.exists(cam_poses_path):
                continue
                
            with open(cam_poses_path, 'r') as f:
                cam_poses = json.load(f)
            
            if frame_idx >= len(cam_poses):
                continue
            
            cam_pose = cam_poses[frame_idx]
            
            world_from_camera = build_transform_matrix(
                cam_pose.get('position', {}),
                cam_pose.get('heading', {})
            )
            
            camera_from_world = np.linalg.inv(world_from_camera)
            camera_from_ego = camera_from_world @ world_from_ego
            lidar2cam = camera_from_ego
            
            # Store camera info (BEVFusion expects this format)
            images[cam_key] = {
                'img_path': img_path,
                'cam2img': cam2img,       # 3x3 matrix
                'lidar2cam': lidar2cam,   # 4x4 matrix
            }
        
        return images

    def load_data_list(self):
        """Load annotations and build camera data for all 6 cameras."""
        assert os.path.exists(self.ann_file), f"Annotation file not found: {self.ann_file}"
        infos = mmengine.load(self.ann_file)

        data_list = []
        cameras_loaded_count = {cam: 0 for cam in CAMERA_ORDER}
        parse_failures = 0
        
        for idx, info in enumerate(infos):
            lidar_rel = info['lidar_path']
            lidar_abs = os.path.join(self.data_root, lidar_rel) if not os.path.isabs(lidar_rel) else lidar_rel
            
            anno_rel = info.get('anno_path', None)
            anno_path = os.path.join(self.data_root, anno_rel) if (anno_rel is not None and not os.path.isabs(anno_rel)) else anno_rel
            
            # Extract scene and frame from paths
            scene_id, frame_idx = extract_scene_and_frame(lidar_rel)
            if scene_id is None or frame_idx is None:
                scene_id, frame_idx = extract_scene_and_frame(anno_rel)
            
            if scene_id is None or frame_idx is None:
                parse_failures += 1
                sample_idx_str = f"{idx:06d}"
            else:
                sample_idx_str = f"{scene_id}_{frame_idx:04d}"
            
            # Load camera data for ALL 6 cameras
            images = {}
            if self.modality.get('use_camera', False) and scene_id is not None and frame_idx is not None:
                images = self._load_camera_data(scene_id, frame_idx)
                
                for cam_key in images:
                    cameras_loaded_count[cam_key] += 1

            item = dict(
                sample_idx=sample_idx_str,
                lidar_points=dict(lidar_path=lidar_abs, num_pts_feats=4),
                anno_path=anno_path,
                calib=info.get('calib', None),
                images=images,  # CRITICAL: images dict with all cameras
            )
                
            data_list.append(item)
        
        # Log camera loading statistics
        print(f"\n{'='*60}")
        print("PandaSet Dataset Camera Loading Statistics:")
        print(f"{'='*60}")
        print(f"Total samples: {len(data_list)}")
        print(f"Parse failures: {parse_failures}")
        for cam_key, count in cameras_loaded_count.items():
            pct = (count / len(data_list) * 100) if len(data_list) > 0 else 0
            print(f"  {cam_key:15s}: {count:4d}/{len(data_list)} ({pct:.1f}%)")
        
        # Debug: show first sample's images dict
        if len(data_list) > 0 and 'images' in data_list[0]:
            print(f"\nFirst sample images keys: {list(data_list[0]['images'].keys())}")
        print(f"{'='*60}\n")

        return data_list

    def parse_ann_info(self, info):
        """Parse cuboid annotations into MMDetection3D format."""
        anno_path = info.get('anno_path', None)
        
        if anno_path is not None:
            if not os.path.isabs(anno_path):
                if not anno_path.startswith(self.data_root):
                    anno_path = os.path.join(self.data_root, anno_path)
        
        if (anno_path is None) or (not os.path.exists(anno_path)):
            empty_boxes = LiDARInstance3DBoxes(
                np.zeros((0, 7), dtype=np.float32), box_dim=7, origin=(0.5, 0.5, 0.5)
            )
            return dict(
                gt_bboxes_3d=empty_boxes,
                gt_labels_3d=np.zeros((0,), dtype=np.int64)
            )

        annos = pickle.load(open(anno_path, 'rb'))
        
        # Filter for front LiDAR (sensor_id=1) or camera_used=0
        lidar_type = None
        if lidar_type == 1:
            anno1 = annos[annos['cuboids.sensor_id']==1]
            anno2 = annos[annos['camera_used']==0]
            annos = concat([anno1, anno2], ignore_index=True).drop_duplicates().reset_index(drop=True)
        elif lidar_type == 0:
            annos = annos[annos['cuboids.sensor_id']==0]
        elif lidar_type == -1:
            annos = annos[annos['cuboids.sensor_id']==-1]
        elif lidar_type is None:
            annos = annos

        gt_bboxes_3d, gt_labels_3d = [], []
        for _, obj in annos.iterrows():
            label = obj.get('label', None)
            
            if label not in self.metainfo['classes']:
                continue

            box = [
                obj['position.x'], obj['position.y'], obj['position.z'],
                obj['dimensions.x'], obj['dimensions.y'], obj['dimensions.z'],
                obj['yaw'],
                0.0
            ]
            gt_bboxes_3d.append(box)
            gt_labels_3d.append(self.metainfo['classes'].index(label))

        if len(gt_bboxes_3d) == 0:
            boxes_3d = LiDARInstance3DBoxes(
                np.zeros((0, 9), dtype=np.float32), box_dim=9, origin=(0.5, 0.5, 0.5)
            )
            gt_labels_3d = np.zeros((0,), dtype=np.int64)
        else:
            gt_bboxes_3d = np.array(gt_bboxes_3d, dtype=np.float32)
            gt_labels_3d = np.array(gt_labels_3d, dtype=np.int64)
            
            if gt_bboxes_3d.shape[1] < 9:
                pad_size = 9 - gt_bboxes_3d.shape[1]
                pad = np.zeros((gt_bboxes_3d.shape[0], pad_size), dtype=np.float32)
                gt_bboxes_3d = np.concatenate([gt_bboxes_3d, pad], axis=1)
            
            boxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d, box_dim=9, origin=(0.5, 0.5, 0.5))
        
        return dict(
            gt_bboxes_3d=boxes_3d,
            gt_labels_3d=gt_labels_3d
        )

    def get_ann_info(self, index):
        info = self.data_list[index]
        return self.parse_ann_info(info)

    def prepare_data(self, index):
        """Ensure ann_info exists before pipeline."""
        ori_input_dict = self.get_data_info(index)
        input_dict = ori_input_dict.copy()
        input_dict['box_type_3d'] = self.box_type_3d
        input_dict['box_mode_3d'] = self.box_mode_3d
        
        # DEBUG: Print images info for first few samples
        if self._debug_count < 3:
            if 'images' in input_dict:
                print(f"\n[DEBUG prepare_data #{self._debug_count}] images has {len(input_dict['images'])} cameras: {list(input_dict['images'].keys())}")
            else:
                print(f"\n[DEBUG prepare_data #{self._debug_count}] NO 'images' key in input_dict!")
                print(f"  Available keys: {list(input_dict.keys())}")
            self._debug_count += 1
        
        input_dict['ann_info'] = self.parse_ann_info(ori_input_dict)

        if not self.test_mode and self.filter_empty_gt:
            if len(input_dict['ann_info']['gt_labels_3d']) == 0:
                return None
        
        example = self.pipeline(input_dict)
        
        if not self.test_mode and self.filter_empty_gt:
            if example is None or len(example['data_samples'].gt_instances_3d.labels_3d) == 0:
                return None
        
        return example

    def get_data_info(self, idx: int) -> dict:
        """Get data info while preserving all fields including images."""
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes_data = memoryview(self.data_bytes[start_addr:end_addr])
            data_info = pickle.loads(bytes_data)
        else:
            data_info = copy.deepcopy(self.data_list[idx])
        
        return data_info