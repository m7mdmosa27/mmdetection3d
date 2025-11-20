# mmdet3d/datasets/pandaset_dataset.py
#
# PandaSet custom dataset for MMDetection3D
# FIXED VERSION with correct camera projection for BEVFusion

import os
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


@DATASETS.register_module()
class PandaSetDataset(Det3DDataset):
    """Custom dataset for PandaSet with fixed camera projection.
    
    Key fixes:
    1. Proper ego-frame coordinate system
    2. Correct lidar2cam for BEV pooling
    3. Camera extrinsics that work with DepthLSSTransform
    
    The coordinate system:
    - Points stored in WORLD coordinates (PandaSet format)
    - Transformed to EGO coordinates (vehicle frame) by PandaSetWorldToEgo
    - Camera extrinsics are ego_from_camera (for projection)
    - lidar2cam is camera_from_ego (inverse of ego_from_camera)
    
    FIXED: Coordinate transformation now matches PandaSet SDK (geometry.py)
    
    The key insight from PandaSet SDK:
    - Points are in WORLD coordinates
    - Camera pose is world_from_camera transform
    - To get camera_from_world: inv(camera_pose)
    - LiDAR pose is NOT needed for projection!

    Args:
        data_root (str): Root path to PandaSet.
        ann_file (str): Path to info file (e.g., `pandaset_infos_train.pkl`).
        pipeline (list[dict], optional): Data pipeline.
        modality (dict): Modality dict, e.g., {'use_lidar': True, 'use_camera': True}.
        box_type_3d (str): Type of 3D box ('LiDAR' or 'Depth').
        filter_empty_gt (bool): Whether to filter frames without GT.
        test_mode (bool): If True, does not load annotations.
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
                 **kwargs):
        
        self._pipeline_cfg = pipeline
        
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

    def load_data_list(self):
        """Load annotations and build proper camera extrinsics."""
        assert os.path.exists(self.ann_file), f"Annotation file not found: {self.ann_file}"
        infos = mmengine.load(self.ann_file)

        data_list = []
        for info in infos:
            lidar_rel = info['lidar_path']
            lidar_abs = os.path.join(self.data_root, lidar_rel) if not os.path.isabs(lidar_rel) else lidar_rel
            anno_rel = info.get('anno_path', None)
            anno_path = os.path.join(self.data_root, anno_rel) if (anno_rel is not None and not os.path.isabs(anno_rel)) else anno_rel
            calib = info.get('calib', None)

            images = None
            img_rel = info.get('img_path', None)
            img_abs = None
            if isinstance(img_rel, str):
                img_abs = os.path.join(self.data_root, img_rel) if not os.path.isabs(img_rel) else img_rel
            
            if img_rel is not None and calib is not None and isinstance(calib, dict):
                # Build camera intrinsics
                intr = calib.get('intrinsics', {})
                fx = float(intr.get('fx', 0.0))
                fy = float(intr.get('fy', 0.0))
                cx = float(intr.get('cx', 0.0))
                cy = float(intr.get('cy', 0.0))
                cam2img = np.array([[fx, 0.0, cx],
                                    [0.0, fy, cy],
                                    [0.0, 0.0, 1.0]], dtype=np.float32)

                # CRITICAL FIX: Build extrinsics for BEVFusion
                # BEVFusion expects:
                # - Points in EGO frame (done by PandaSetWorldToEgo transform)
                # - lidar2cam = camera_from_ego transformation
                
                extr = calib.get('extrinsics', {})
                camera_pose = extr.get('camera_pose', {})
                lidar_pose = extr.get('lidar_pose', {})
                
                # Build world_from_camera matrix
                cam_pos = camera_pose.get('position', {})
                cam_heading = camera_pose.get('heading', {})
                
                cam_t = np.array([
                    cam_pos.get('x', 0.0),
                    cam_pos.get('y', 0.0),
                    cam_pos.get('z', 0.0)
                ], dtype=np.float32)
                
                w = cam_heading.get('w', 1.0)
                x = cam_heading.get('x', 0.0)
                y = cam_heading.get('y', 0.0)
                z = cam_heading.get('z', 0.0)
                
                cam_R = np.array([
                    [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
                    [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
                    [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
                ], dtype=np.float32)
                
                world_from_camera = np.eye(4, dtype=np.float32)
                world_from_camera[:3, :3] = cam_R
                world_from_camera[:3, 3] = cam_t
                
                # Build world_from_ego (lidar) matrix
                lidar_pos = lidar_pose.get('position', {})
                lidar_heading = lidar_pose.get('heading', {})
                
                lidar_t = np.array([
                    lidar_pos.get('x', 0.0),
                    lidar_pos.get('y', 0.0),
                    lidar_pos.get('z', 0.0)
                ], dtype=np.float32)
                
                w = lidar_heading.get('w', 1.0)
                x = lidar_heading.get('x', 0.0)
                y = lidar_heading.get('y', 0.0)
                z = lidar_heading.get('z', 0.0)
                
                lidar_R = np.array([
                    [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
                    [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
                    [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
                ], dtype=np.float32)
                
                world_from_ego = np.eye(4, dtype=np.float32)
                world_from_ego[:3, :3] = lidar_R
                world_from_ego[:3, 3] = lidar_t
                
                # CRITICAL: BEVFusion needs camera_from_ego
                # camera_from_ego = camera_from_world @ world_from_ego
                camera_from_world = np.linalg.inv(world_from_camera)
                camera_from_ego = camera_from_world @ world_from_ego
                
                # This is what BEVFusion will use for projection
                lidar2cam = camera_from_ego

                images = {
                    'FRONT': {
                        'img_path': img_abs if img_abs is not None else img_rel,
                        'cam2img': cam2img,
                        'lidar2cam': lidar2cam
                    }
                }

            item = dict(
                sample_idx=info.get('sample_idx', None),
                lidar_points=dict(lidar_path=lidar_abs, num_pts_feats=4),
                anno_path=anno_path,
                calib=calib
            )
            if images is not None:
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
        
        # Filter for front LiDAR
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
        else:
            raise ValueError(f"Invalid lidar type: {lidar_type}")

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
        # Get data from parent (handles serialization, etc.)
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(self.data_bytes[start_addr:end_addr])
            data_info = pickle.loads(bytes)
        else:
            data_info = copy.deepcopy(self.data_list[idx])
        
        # CRITICAL: Don't overwrite sample_idx with integer index
        # Keep the original string sample_idx (e.g., '005_0078')
        # Parent class would set data_info['sample_idx'] = idx here
        
        return data_info