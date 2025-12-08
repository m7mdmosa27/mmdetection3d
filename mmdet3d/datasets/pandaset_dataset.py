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


class MMDet3DCompose(Compose):
    """Custom Compose that uses mmdet3d.registry.TRANSFORMS instead of mmengine's."""
    
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


# Camera mapping for PandaSet
PANDASET_CAMERAS = {
    'FRONT': 'front_camera',
    'BACK': 'back_camera',
    'FRONT_LEFT': 'front_left_camera',
    'FRONT_RIGHT': 'front_right_camera',
    'LEFT': 'left_camera',
    'RIGHT': 'right_camera'
}

# Ordered list for multi-camera loading
CAMERA_ORDER = ['FRONT', 'BACK', 'FRONT_LEFT', 'FRONT_RIGHT', 'LEFT', 'RIGHT']


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


@DATASETS.register_module()
class PandaSetDataset(Det3DDataset):
    """Custom dataset for PandaSet with support for all 6 cameras.
    
    Supports:
    - Single camera (default_cam_key)
    - Multi-camera (all 6 cameras)
    - LiDAR only mode
    
    Camera mapping:
    - FRONT -> front_camera
    - BACK -> back_camera
    - FRONT_LEFT -> front_left_camera
    - FRONT_RIGHT -> front_right_camera
    - LEFT -> left_camera
    - RIGHT -> right_camera
    
    Top 5 Classes (87.6% coverage):
    - Car
    - Pedestrian
    - Pedestrian with Object
    - Temporary Construction Barriers
    - Cones
    """

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
                 default_cam_key='FRONT',
                 **kwargs):
        
        self._pipeline_cfg = pipeline
        self.default_cam_key = default_cam_key
        
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

    def _load_camera_calibration(self, scene_dir, cam_name, frame_idx):
        """Load camera intrinsics and extrinsics for a specific camera and frame.
        
        Args:
            scene_dir: Path to scene directory (e.g., data/pandaset/001)
            cam_name: Camera folder name (e.g., 'front_camera')
            frame_idx: Frame index as integer
            
        Returns:
            dict with 'cam2img' (3x3) and 'lidar2cam' (4x4) matrices, or None if failed
        """
        cam_dir = os.path.join(scene_dir, 'camera', cam_name)
        
        # Load intrinsics
        intrinsics_path = os.path.join(cam_dir, 'intrinsics.json')
        if not os.path.exists(intrinsics_path):
            return None
            
        with open(intrinsics_path, 'r') as f:
            intrinsics = json.load(f)
        
        fx = float(intrinsics.get('fx', 0.0))
        fy = float(intrinsics.get('fy', 0.0))
        cx = float(intrinsics.get('cx', 0.0))
        cy = float(intrinsics.get('cy', 0.0))
        
        cam2img = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Load camera poses
        cam_poses_path = os.path.join(cam_dir, 'poses.json')
        if not os.path.exists(cam_poses_path):
            return None
            
        with open(cam_poses_path, 'r') as f:
            cam_poses = json.load(f)
        
        if frame_idx >= len(cam_poses):
            return None
            
        cam_pose = cam_poses[frame_idx]
        
        # Load LiDAR poses
        lidar_dir = os.path.join(scene_dir, 'lidar')
        lidar_poses_path = os.path.join(lidar_dir, 'poses.json')
        if not os.path.exists(lidar_poses_path):
            return None
            
        with open(lidar_poses_path, 'r') as f:
            lidar_poses = json.load(f)
        
        if frame_idx >= len(lidar_poses):
            return None
            
        lidar_pose = lidar_poses[frame_idx]
        
        # Build transformation matrices
        # world_from_camera
        world_from_camera = build_transform_matrix(
            cam_pose.get('position', {}),
            cam_pose.get('heading', {})
        )
        
        # world_from_ego (lidar)
        world_from_ego = build_transform_matrix(
            lidar_pose.get('position', {}),
            lidar_pose.get('heading', {})
        )
        
        # camera_from_ego = camera_from_world @ world_from_ego
        camera_from_world = np.linalg.inv(world_from_camera)
        camera_from_ego = camera_from_world @ world_from_ego
        
        # This is lidar2cam for BEVFusion
        lidar2cam = camera_from_ego
        
        return {
            'cam2img': cam2img,
            'lidar2cam': lidar2cam
        }

    def load_data_list(self):
        """Load annotations and build camera data for all 6 cameras."""
        assert os.path.exists(self.ann_file), f"Annotation file not found: {self.ann_file}"
        infos = mmengine.load(self.ann_file)

        data_list = []
        for info in infos:
            # Parse sample_idx to get scene and frame
            sample_idx = str(info.get('sample_idx', ''))
            
            # Determine scene directory
            lidar_rel = info['lidar_path']
            lidar_abs = os.path.join(self.data_root, lidar_rel) if not os.path.isabs(lidar_rel) else lidar_rel
            
            # Extract scene dir from lidar path
            # lidar_path format: XXX/lidar/YY.pkl or data/pandaset/XXX/lidar/YY.pkl
            lidar_parts = lidar_rel.replace('\\', '/').split('/')
            scene_id = None
            frame_idx = None
            
            for i, part in enumerate(lidar_parts):
                if part == 'lidar' and i > 0:
                    scene_id = lidar_parts[i-1]
                    frame_file = lidar_parts[i+1] if i+1 < len(lidar_parts) else None
                    if frame_file:
                        frame_idx = int(frame_file.replace('.pkl', ''))
                    break
            
            if scene_id is None:
                # Fallback: try to extract from sample_idx
                if '_' in sample_idx:
                    parts = sample_idx.split('_')
                    scene_id = parts[0]
                    frame_idx = int(parts[1]) if len(parts) > 1 else 0
            
            scene_dir = os.path.join(self.data_root, scene_id) if scene_id else None
            
            # Annotation path
            anno_rel = info.get('anno_path', None)
            anno_path = os.path.join(self.data_root, anno_rel) if (anno_rel is not None and not os.path.isabs(anno_rel)) else anno_rel
            
            # Build images dict for ALL cameras
            images = {}
            
            if self.modality.get('use_camera', False) and scene_dir and frame_idx is not None:
                for cam_key in CAMERA_ORDER:
                    cam_name = PANDASET_CAMERAS[cam_key]
                    
                    # Image path
                    img_path = os.path.join(scene_dir, 'camera', cam_name, f'{frame_idx:02d}.jpg')
                    
                    if not os.path.exists(img_path):
                        continue
                    
                    # Load calibration
                    calib = self._load_camera_calibration(scene_dir, cam_name, frame_idx)
                    
                    if calib is None:
                        continue
                    
                    images[cam_key] = {
                        'img_path': img_path,
                        'cam2img': calib['cam2img'],
                        'lidar2cam': calib['lidar2cam']
                    }
            
            # Fallback: use old calib from info if available (for single camera)
            if len(images) == 0:
                calib = info.get('calib', None)
                img_rel = info.get('img_path', None)
                
                if img_rel is not None and calib is not None and isinstance(calib, dict):
                    img_abs = os.path.join(self.data_root, img_rel) if not os.path.isabs(img_rel) else img_rel
                    
                    intr = calib.get('intrinsics', {})
                    fx = float(intr.get('fx', 0.0))
                    fy = float(intr.get('fy', 0.0))
                    cx = float(intr.get('cx', 0.0))
                    cy = float(intr.get('cy', 0.0))
                    cam2img = np.array([[fx, 0.0, cx],
                                        [0.0, fy, cy],
                                        [0.0, 0.0, 1.0]], dtype=np.float32)

                    extr = calib.get('extrinsics', {})
                    camera_pose = extr.get('camera_pose', {})
                    lidar_pose = extr.get('lidar_pose', {})
                    
                    world_from_camera = build_transform_matrix(
                        camera_pose.get('position', {}),
                        camera_pose.get('heading', {})
                    )
                    
                    world_from_ego = build_transform_matrix(
                        lidar_pose.get('position', {}),
                        lidar_pose.get('heading', {})
                    )
                    
                    camera_from_world = np.linalg.inv(world_from_camera)
                    camera_from_ego = camera_from_world @ world_from_ego
                    lidar2cam = camera_from_ego

                    images = {
                        'FRONT': {
                            'img_path': img_abs if img_abs is not None else img_rel,
                            'cam2img': cam2img,
                            'lidar2cam': lidar2cam
                        }
                    }

            item = dict(
                sample_idx=sample_idx,
                lidar_points=dict(lidar_path=lidar_abs, num_pts_feats=4),
                anno_path=anno_path,
                calib=info.get('calib', None)
            )
            
            if len(images) > 0:
                item['images'] = images
                
            data_list.append(item)

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
        lidar_type = 1
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

            # Build 7D box [x, y, z, dx, dy, dz, yaw]
            box = [
                obj['position.x'], obj['position.y'], obj['position.z'],
                obj['dimensions.x'], obj['dimensions.y'], obj['dimensions.z'],
                obj['yaw'],
                0.0  # Pad to 8D
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
            
            # Pad to 9 dims (7 box + 2 velocity)
            if gt_bboxes_3d.shape[1] < 9:
                pad_size = 9 - gt_bboxes_3d.shape[1]
                pad = np.zeros((gt_bboxes_3d.shape[0], pad_size), dtype=np.float32)
                gt_bboxes_3d = np.concatenate([gt_bboxes_3d, pad], axis=1)
            
            boxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d, box_dim=9, origin=(0.5, 0.5, 0.5))
        
        ann_info = dict(
            gt_bboxes_3d=boxes_3d,
            gt_labels_3d=gt_labels_3d
        )
        return ann_info

    def get_ann_info(self, index):
        info = self.data_list[index]
        return self.parse_ann_info(info)

    def prepare_data(self, index):
        """Ensure ann_info exists before pipeline."""
        ori_input_dict = self.get_data_info(index)
        input_dict = ori_input_dict.copy()
        input_dict['box_type_3d'] = self.box_type_3d
        input_dict['box_mode_3d'] = self.box_mode_3d
        
        # Always parse annotations (so evaluator can see ground truth)
        input_dict['ann_info'] = self.parse_ann_info(ori_input_dict)

        # Only skip samples with empty GT during training, not during testing
        if not self.test_mode and self.filter_empty_gt:
            if len(input_dict['ann_info']['gt_labels_3d']) == 0:
                return None
        
        example = self.pipeline(input_dict)
        
        if not self.test_mode and self.filter_empty_gt:
            if example is None or len(example['data_samples'].gt_instances_3d.labels_3d) == 0:
                return None
        
        return example

    def get_data_info(self, idx: int) -> dict:
        """Get data info while preserving string sample_idx."""
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(self.data_bytes[start_addr:end_addr])
            data_info = pickle.loads(bytes)
        else:
            data_info = copy.deepcopy(self.data_list[idx])
        
        # CRITICAL: Don't overwrite sample_idx with integer index
        # Keep the original string sample_idx (e.g., '005_0078')
        
        return data_info