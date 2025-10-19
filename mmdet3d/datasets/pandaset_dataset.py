# mmdet3d/datasets/pandaset_dataset.py
#
# PandaSet custom dataset for MMDetection3D
# Uses forward-facing LiDAR + front_camera for 3D detection (cuboids only)
# Reads native .pkl and .json files (no KITTI conversion required)

import os
import pickle
import numpy as np
import mmengine
from mmdet3d.datasets.det3d_dataset import Det3DDataset
from mmdet3d.registry import DATASETS, TRANSFORMS
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
from mmengine.dataset import Compose


class MMDet3DCompose(Compose):
    """Custom Compose that uses mmdet3d.registry.TRANSFORMS instead of mmengine's.
    
    This fixes the issue where mmengine's Compose can't find mmdet3d transforms
    because they're registered in a different registry.
    """
    
    def __init__(self, transforms):
        self.transforms = []
        if isinstance(transforms, dict):
            transforms = [transforms]
        for transform in transforms:
            if isinstance(transform, dict):
                # Use mmdet3d's TRANSFORMS registry instead of mmengine's
                transform = TRANSFORMS.build(transform)
                self.transforms.append(transform)
            else:
                self.transforms.append(transform)


@DATASETS.register_module()
class PandaSetDataset(Det3DDataset):
    """Custom dataset for PandaSet (front_camera + forward LiDAR, 3D cuboids).

    The dataset expects a folder structure like:

    data/pandaset/
        ├── 001/
        │   ├── lidar/
        │   │   ├── 00.pkl
        │   │   ├── poses.json
        │   │   └── timestamps.json
        │   ├── camera/
        │   │   └── front_camera/
        │   │       ├── 00.jpg
        │   │       ├── intrinsics.json
        │   │       ├── poses.json
        │   │       └── timestamps.json
        │   ├── annotations/
        │   │   └── cuboids/
        │   │       ├── 00.pkl
        │   │       └── ...
        │   └── meta/
        │       └── gps.json
        ├── 002/
        └── ...

    Args:
        data_root (str): Root path to PandaSet.
        ann_file (str): Path to info file (e.g., `pandaset_infos_train.pkl`).
        pipeline (list[dict], optional): Data pipeline.
        modality (dict): Modality dict, e.g., {'use_lidar': True, 'use_camera': True}.
        box_type_3d (str): Type of 3D box ('LiDAR' or 'Depth').
        filter_empty_gt (bool): Whether to filter frames without GT.
        test_mode (bool): If True, does not load annotations.
    """

    # NOTE: Update these classes based on analyze_pandaset_labels.py output
    # These should be the exact label strings from PandaSet, no mapping needed
    METAINFO = {
        'classes': ('Car', 'Pedestrian', 'Pickup Truck', 'Semi-truck', 'Cyclist')
        # ☝️ Replace with your top 5 classes after running analyze_pandaset_labels.py
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
        
        # Store pipeline before calling super().__init__
        # We'll build it manually to use the correct registry
        self._pipeline_cfg = pipeline
        
        # Temporarily set pipeline to None to prevent base class from building it
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=None,  # We'll build it ourselves
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            data_prefix=data_prefix,
            **kwargs
        )
        
        # Now build the pipeline with the correct registry
        if self._pipeline_cfg is not None:
            self.pipeline = MMDet3DCompose(self._pipeline_cfg)

    # -------------------------------------------------------
    # Load dataset information from pre-built info file
    # -------------------------------------------------------
    def load_data_list(self):
        """Load annotations and paths from ann_file."""
        assert os.path.exists(self.ann_file), f"Annotation file not found: {self.ann_file}"
        infos = mmengine.load(self.ann_file)

        data_list = []
        for info in infos:
            # Expect info fields to be relative to data_root;
            # keep them relative and control prefix via data_prefix in config
            lidar_rel = info['lidar_path']
            lidar_abs = os.path.join(self.data_root, lidar_rel) if not os.path.isabs(lidar_rel) else lidar_rel
            anno_rel = info.get('anno_path', None)
            anno_path = os.path.join(self.data_root, anno_rel) if (anno_rel is not None and not os.path.isabs(anno_rel)) else anno_rel
            calib = info.get('calib', None)

            # Build single-view camera dict compatible with BEVFusion loaders
            images = None
            img_rel = info.get('img_path', None)
            img_abs = None
            if isinstance(img_rel, str):
                img_abs = os.path.join(self.data_root, img_rel) if not os.path.isabs(img_rel) else img_rel
            if img_rel is not None and calib is not None and isinstance(calib, dict):
                # Intrinsics K (3x3)
                intr = calib.get('intrinsics', {})
                fx = float(intr.get('fx', 0.0))
                fy = float(intr.get('fy', 0.0))
                cx = float(intr.get('cx', 0.0))
                cy = float(intr.get('cy', 0.0))
                cam2img = np.array([[fx, 0.0, cx],
                                    [0.0, fy, cy],
                                    [0.0, 0.0, 1.0]], dtype=np.float32)

                # Extrinsics: derive lidar->cam from poses if available
                extr = calib.get('extrinsics', {})
                def to_mat4(m):
                    if m is None:
                        return np.eye(4, dtype=np.float32)
                    # array-like provided
                    if isinstance(m, (list, tuple, np.ndarray)):
                        arr = np.array(m)
                        if arr.size == 16:
                            return arr.reshape(4, 4).astype(np.float32)
                        if arr.shape == (4, 4):
                            return arr.astype(np.float32)
                    # dict forms
                    if isinstance(m, dict):
                        if 'matrix' in m:
                            return to_mat4(m['matrix'])
                        # common rotation/translation patterns
                        for rot_key in ('R', 'rotation', 'rot', 'r'):
                            if rot_key in m:
                                R = np.array(m[rot_key], dtype=np.float32).reshape(3, 3)
                                t_val = m.get('t') or m.get('translation') or m.get('trans') or m.get('t_vec') or [0.0, 0.0, 0.0]
                                t = np.array(t_val, dtype=np.float32).reshape(3)
                                T = np.eye(4, dtype=np.float32)
                                T[:3, :3] = R
                                T[:3, 3] = t
                                return T
                        if 'position' in m:
                            pos = m['position']
                            t = np.array([pos.get('x', 0.0), pos.get('y', 0.0), pos.get('z', 0.0)], dtype=np.float32)
                            yaw_val = m.get('heading', m.get('yaw', 0.0))
                            if isinstance(yaw_val, (int, float)):
                                yaw = float(yaw_val)
                            elif isinstance(yaw_val, dict):
                                yaw = float(yaw_val.get('rad', yaw_val.get('value', 0.0))) if any(
                                    k in yaw_val for k in ('rad', 'value')) else 0.0
                            else:
                                yaw = 0.0
                            c, s = np.cos(yaw), np.sin(yaw)
                            R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
                            T = np.eye(4, dtype=np.float32)
                            T[:3, :3] = R
                            T[:3, 3] = t
                            return T
                    return np.eye(4, dtype=np.float32)

                cam_pose = to_mat4(extr.get('camera_pose'))
                lidar_pose = to_mat4(extr.get('lidar_pose'))
                # Assume poses are world_from_sensor; then lidar2cam = inv(world_from_cam) @ world_from_lidar
                def safe_inv(T):
                    try:
                        return np.linalg.inv(T)
                    except Exception:
                        return np.eye(4, dtype=np.float32)
                lidar2cam = safe_inv(cam_pose) @ lidar_pose

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

    # -------------------------------------------------------
    # Parse annotation file (.pkl with cuboids)
    # -------------------------------------------------------
    def parse_ann_info(self, info):
        """Parse cuboid annotations into MMDetection3D format.
        
        Only annotations with labels in METAINFO['classes'] are kept.
        All other labels are ignored.
        """
        anno_path = info.get('anno_path', None)
        
        # Handle path conversion carefully
        # If anno_path already starts with data_root, don't join again
        if anno_path is not None:
            if not os.path.isabs(anno_path):
                # Check if path already contains data_root
                if not anno_path.startswith(self.data_root):
                    anno_path = os.path.join(self.data_root, anno_path)
                # If path starts with data_root (common case from info files), use as-is
        
        if (anno_path is None) or (not os.path.exists(anno_path)):
            empty_boxes = LiDARInstance3DBoxes(
                np.zeros((0, 7), dtype=np.float32), box_dim=7, origin=(0.5, 0.5, 0.5)
            )
            return dict(
                gt_bboxes_3d=empty_boxes,
                gt_labels_3d=np.zeros((0,), dtype=np.int64)
            )

        # Load DataFrame with cuboids
        annos = pickle.load(open(anno_path, 'rb'))

        gt_bboxes_3d, gt_labels_3d = [], []
        for _, obj in annos.iterrows():
            # Each row is one cuboid
            label = obj.get('label', None)
            
            # Only keep labels that are in our class list (ignore all others)
            if label not in self.metainfo['classes']:
                continue

            box = [
                obj['position.x'], obj['position.y'], obj['position.z'],
                obj['dimensions.x'], obj['dimensions.y'], obj['dimensions.z'],
                obj['yaw']
            ]
            gt_bboxes_3d.append(box)
            gt_labels_3d.append(self.metainfo['classes'].index(label))

        if len(gt_bboxes_3d) == 0:
            boxes_3d = LiDARInstance3DBoxes(
                np.zeros((0, 7), dtype=np.float32), box_dim=7, origin=(0.5, 0.5, 0.5)
            )
            gt_labels_3d = np.zeros((0,), dtype=np.int64)
        else:
            gt_bboxes_3d = np.array(gt_bboxes_3d, dtype=np.float32)
            gt_labels_3d = np.array(gt_labels_3d, dtype=np.int64)
            # Wrap numpy boxes into LiDARInstance3DBoxes
            boxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d, box_dim=7, origin=(0.5, 0.5, 0.5))
        ann_info = dict(
            gt_bboxes_3d=boxes_3d,
            gt_labels_3d=gt_labels_3d
        )
        return ann_info

    # -------------------------------------------------------
    # MMDet3D expects get_ann_info to return annotation dict
    # -------------------------------------------------------
    def get_ann_info(self, index):
        info = self.data_list[index]
        return self.parse_ann_info(info)

    def prepare_data(self, index):
        """Ensure ann_info exists before pipeline, then defer to base logic."""
        # Borrow logic from Det3DDataset.prepare_data with a precheck
        ori_input_dict = self.get_data_info(index)
        input_dict = ori_input_dict.copy()
        input_dict['box_type_3d'] = self.box_type_3d
        input_dict['box_mode_3d'] = self.box_mode_3d
        # Camera image paths should already be converted in parse_data_info
        if not self.test_mode:
            if 'ann_info' not in input_dict:
                input_dict['ann_info'] = self.parse_ann_info(ori_input_dict)
            if self.filter_empty_gt and len(input_dict['ann_info']['gt_labels_3d']) == 0:
                return None
        example = self.pipeline(input_dict)
        if not self.test_mode and self.filter_empty_gt:
            if example is None or len(example['data_samples'].gt_instances_3d.labels_3d) == 0:
                return None
        return example

    # Use base class get_data_info implementation
