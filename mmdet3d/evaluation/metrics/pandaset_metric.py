"""
PandaSetMetric - FIXED VERSION

This version transforms GT boxes from WORLD to EGO coordinates to match predictions.

Replace the content of:
    mmdet3d/evaluation/metrics/pandaset_metric.py
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Optional, Sequence

import mmengine
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS
from pandas import concat


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


def transform_boxes_world_to_ego(boxes, ego_from_world):
    """Transform boxes from world to ego frame.
    
    Args:
        boxes: (N, 9) array [x, y, z, dx, dy, dz, yaw, vx, vy]
        ego_from_world: 4x4 transformation matrix
        
    Returns:
        Transformed boxes (N, 9)
    """
    if len(boxes) == 0:
        return boxes
    
    boxes = boxes.copy()
    
    R = ego_from_world[:3, :3]
    t = ego_from_world[:3, 3]
    
    # Transform centers
    centers_world = boxes[:, :3]
    centers_ego = (R @ centers_world.T).T + t
    
    # Transform yaw
    rotation_z = np.arctan2(R[1, 0], R[0, 0])
    yaw_world = boxes[:, 6]
    yaw_ego = yaw_world - rotation_z
    yaw_ego = np.arctan2(np.sin(yaw_ego), np.cos(yaw_ego))
    
    # Update boxes
    boxes[:, :3] = centers_ego
    boxes[:, 6] = yaw_ego
    
    return boxes


def box3d_iou_bev(boxes_a, boxes_b):
    """Compute BEV IoU between two sets of boxes (axis-aligned approximation)."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    
    ious = np.zeros((len(boxes_a), len(boxes_b)))
    
    for i, box_a in enumerate(boxes_a):
        xa, ya = box_a[0], box_a[1]
        dxa, dya = box_a[3], box_a[4]
        
        for j, box_b in enumerate(boxes_b):
            xb, yb = box_b[0], box_b[1]
            dxb, dyb = box_b[3], box_b[4]
            
            # Axis-aligned intersection
            x_overlap = max(0, min(xa + dxa/2, xb + dxb/2) - max(xa - dxa/2, xb - dxb/2))
            y_overlap = max(0, min(ya + dya/2, yb + dyb/2) - max(ya - dya/2, yb - dyb/2))
            
            intersection = x_overlap * y_overlap
            area_a = dxa * dya
            area_b = dxb * dyb
            union = area_a + area_b - intersection
            
            ious[i, j] = intersection / union if union > 0 else 0
    
    return ious


@METRICS.register_module(force=True)
class PandaSetMetric(BaseMetric):
    """PandaSet evaluation metric with coordinate frame alignment.
    
    This version transforms GT boxes from WORLD to EGO coordinates
    to match the predictions.
    
    Args:
        ann_file: Path to annotation info file
        iou_thresholds: List of IoU thresholds for evaluation
        score_threshold: Minimum confidence score for predictions
        collect_device: Device for collecting results ('cpu' or 'gpu')
        prefix: Metric prefix for logging
    """
    
    def __init__(self,
                 ann_file: str,
                 iou_thresholds: List[float] = [0.25, 0.5, 0.7],
                 score_threshold: float = 0.1,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ann_file = ann_file
        self.iou_thresholds = iou_thresholds
        self.score_threshold = score_threshold
        self.class_names = ['Car', 'Pedestrian', 'Pedestrian with Object', 
                           'Temporary Construction Barriers', 'Cones']
        
        self.logger = mmengine.MMLogger.get_current_instance()
        self.logger.info(f"PandaSetMetric initialized (with EGO transform)")
        self.logger.info(f"  - ann_file: {ann_file}")
        self.logger.info(f"  - iou_thresholds: {iou_thresholds}")
        self.logger.info(f"  - score_threshold: {score_threshold}")
        
        # Load GT annotations with coordinate transform
        self._load_ground_truth()
    
    def _get_ego_from_world(self, info):
        """Get ego_from_world transform from info."""
        calib = info.get('calib', None)
        if calib is None:
            return None
        
        extr = calib.get('extrinsics', {})
        lidar_pose = extr.get('lidar_pose', {})
        
        if not lidar_pose:
            return None
        
        position = lidar_pose.get('position', {})
        heading = lidar_pose.get('heading', {})
        
        if not position or not heading:
            return None
        
        # Build world_from_ego (LiDAR pose in world)
        world_from_ego = build_transform_matrix(position, heading)
        
        # Compute inverse: ego_from_world
        ego_from_world = np.linalg.inv(world_from_ego)
        
        return ego_from_world
    
    def _load_ground_truth(self):
        """Load ground truth from annotation file and transform to EGO frame."""
        self.logger.info(f"Loading GT from {self.ann_file}")
        
        if not os.path.exists(self.ann_file):
            self.logger.error(f"Annotation file not found: {self.ann_file}")
            self.gt_dict = {}
            return
        
        infos = mmengine.load(self.ann_file)
        self.logger.info(f"Loaded {len(infos)} info entries")
        
        # Determine data_root from ann_file path
        data_root = os.path.dirname(self.ann_file)
        if not data_root:
            data_root = 'data/pandaset'
        
        self.gt_dict = {}
        loaded_count = 0
        failed_count = 0
        empty_count = 0
        transform_count = 0
        
        for info in infos:
            sample_idx = str(info.get('sample_idx', ''))
            anno_path = info.get('anno_path', None)
            
            if not anno_path:
                failed_count += 1
                continue
            
            # Try multiple path resolutions
            final_path = None
            possible_paths = [
                anno_path,
                os.path.join(data_root, anno_path),
                os.path.join('data/pandaset', anno_path),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    final_path = path
                    break
            
            if final_path is None:
                failed_count += 1
                continue
            
            # Get transformation matrix for this sample
            ego_from_world = self._get_ego_from_world(info)
            
            try:
                with open(final_path, 'rb') as f:
                    annos = pickle.load(f)
                
                # Don't filter by sensor - use all annotations
                annos_filtered = annos
                
                boxes, labels = [], []
                for _, obj in annos_filtered.iterrows():
                    label = obj.get('label', '')
                    if label in self.class_names:
                        boxes.append([
                            obj['position.x'], obj['position.y'], obj['position.z'],
                            obj['dimensions.x'], obj['dimensions.y'], obj['dimensions.z'],
                            obj['yaw'], 0, 0  # velocity placeholders
                        ])
                        labels.append(self.class_names.index(label))
                
                if len(boxes) > 0:
                    boxes = np.array(boxes, dtype=np.float32)
                    labels = np.array(labels, dtype=np.int64)
                    
                    # Transform GT boxes from WORLD to EGO frame
                    if ego_from_world is not None:
                        boxes = transform_boxes_world_to_ego(boxes, ego_from_world)
                        transform_count += 1
                    
                    self.gt_dict[sample_idx] = {
                        'boxes': boxes,
                        'labels': labels
                    }
                    loaded_count += 1
                else:
                    empty_count += 1
                    
            except Exception as e:
                if failed_count <= 3:
                    self.logger.warning(f"Error loading {final_path}: {e}")
                failed_count += 1
        
        self.logger.info(f"GT Loading Summary:")
        self.logger.info(f"  - Loaded: {loaded_count}")
        self.logger.info(f"  - Transformed to EGO: {transform_count}")
        self.logger.info(f"  - Empty: {empty_count}")
        self.logger.info(f"  - Failed: {failed_count}")
        
        # Debug: show GT statistics
        if len(self.gt_dict) > 0:
            total_boxes = sum(len(gt['boxes']) for gt in self.gt_dict.values())
            self.logger.info(f"  - Total GT boxes: {total_boxes}")
            
            # Show coordinate statistics for first few samples
            sample_keys = list(self.gt_dict.keys())[:3]
            for key in sample_keys:
                gt = self.gt_dict[key]
                if len(gt['boxes']) > 0:
                    center_mean = gt['boxes'][:, :3].mean(axis=0)
                    self.logger.info(f"  - Sample {key} GT center mean (EGO): [{center_mean[0]:.1f}, {center_mean[1]:.1f}, {center_mean[2]:.1f}]")
            
            for class_id, class_name in enumerate(self.class_names):
                n_gt = sum((gt['labels'] == class_id).sum() for gt in self.gt_dict.values())
                self.logger.info(f"  - {class_name}: {n_gt} GT boxes")
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process predictions from a batch."""
        for data_sample in data_samples:
            sample_idx = str(data_sample.get('sample_idx', ''))
            
            pred_instances = data_sample['pred_instances_3d']
            pred_boxes = pred_instances['bboxes_3d'].tensor.cpu().numpy()
            pred_scores = pred_instances['scores_3d'].cpu().numpy()
            pred_labels = pred_instances['labels_3d'].cpu().numpy()
            
            # Debug: show prediction statistics for first few samples
            if len(self.results) < 5:
                self.logger.info(f"Sample {sample_idx}:")
                self.logger.info(f"  Total predictions: {len(pred_boxes)}")
                if len(pred_scores) > 0:
                    self.logger.info(f"  Score range: [{pred_scores.min():.4f}, {pred_scores.max():.4f}]")
                    self.logger.info(f"  Scores > {self.score_threshold}: {(pred_scores > self.score_threshold).sum()}")
                    
                    # Show per-class predictions
                    for class_id, class_name in enumerate(self.class_names):
                        mask = pred_labels == class_id
                        n_pred = mask.sum()
                        if n_pred > 0:
                            max_score = pred_scores[mask].max()
                            self.logger.info(f"  {class_name}: {n_pred} preds, max_score={max_score:.4f}")
                
                # Compare with GT - now both should be in EGO frame
                if sample_idx in self.gt_dict:
                    gt = self.gt_dict[sample_idx]
                    self.logger.info(f"  GT boxes: {len(gt['boxes'])}")
                    
                    # Check coordinate alignment
                    if len(pred_boxes) > 0 and len(gt['boxes']) > 0:
                        pred_centers = pred_boxes[:, :3].mean(axis=0)
                        gt_centers = gt['boxes'][:, :3].mean(axis=0)
                        self.logger.info(f"  Pred center mean: [{pred_centers[0]:.1f}, {pred_centers[1]:.1f}, {pred_centers[2]:.1f}]")
                        self.logger.info(f"  GT center mean (EGO): [{gt_centers[0]:.1f}, {gt_centers[1]:.1f}, {gt_centers[2]:.1f}]")
            
            # Filter by score threshold
            keep = pred_scores >= self.score_threshold
            
            self.results.append({
                'sample_idx': sample_idx,
                'boxes': pred_boxes[keep],
                'scores': pred_scores[keep],
                'labels': pred_labels[keep]
            })

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute mAP metrics."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Computing metrics for {len(results)} samples...")
        self.logger.info(f"{'='*60}")
        
        # Match predictions with GT
        matched = 0
        for r in results:
            if r['sample_idx'] in self.gt_dict:
                matched += 1
        
        self.logger.info(f"Prediction samples: {len(results)}")
        self.logger.info(f"GT samples: {len(self.gt_dict)}")
        self.logger.info(f"Matched samples: {matched}")
        
        # Compute per-class statistics
        self.logger.info(f"\nPer-class statistics:")
        for class_id, class_name in enumerate(self.class_names):
            n_pred = sum((r['labels'] == class_id).sum() for r in results)
            n_gt = sum((gt['labels'] == class_id).sum() for gt in self.gt_dict.values())
            self.logger.info(f"  {class_name}: {n_pred} preds, {n_gt} GT")
        
        # Compute mAP for each IoU threshold
        metrics = {}
        
        for iou_thr in self.iou_thresholds:
            self.logger.info(f"\n--- IoU Threshold: {iou_thr} ---")
            
            aps = []
            for class_id, class_name in enumerate(self.class_names):
                ap = self._compute_ap_for_class(results, class_id, iou_thr)
                aps.append(ap)
                self.logger.info(f"  {class_name} AP@{iou_thr}: {ap:.4f}")
            
            mean_ap = np.mean(aps)
            self.logger.info(f"  mAP@{iou_thr}: {mean_ap:.4f}")
            metrics[f'mAP@{iou_thr}'] = mean_ap
        
        return metrics
    
    def _compute_ap_for_class(self, results, class_id, iou_thr):
        """Compute AP for a single class."""
        # Collect all predictions and GT for this class
        all_pred_boxes = []
        all_pred_scores = []
        all_gt_boxes = []
        sample_indices = []
        
        for result in results:
            sample_idx = result['sample_idx']
            
            # Get predictions for this class
            mask = result['labels'] == class_id
            pred_boxes = result['boxes'][mask]
            pred_scores = result['scores'][mask]
            
            # Get GT for this class
            if sample_idx in self.gt_dict:
                gt = self.gt_dict[sample_idx]
                gt_mask = gt['labels'] == class_id
                gt_boxes = gt['boxes'][gt_mask]
            else:
                gt_boxes = np.zeros((0, 9), dtype=np.float32)
            
            all_pred_boxes.append(pred_boxes)
            all_pred_scores.append(pred_scores)
            all_gt_boxes.append(gt_boxes)
            sample_indices.append(sample_idx)
        
        # Count total GT
        num_gt = sum(len(gt) for gt in all_gt_boxes)
        
        if num_gt == 0:
            return 0.0
        
        # Flatten predictions
        valid_pred_boxes = [b for b in all_pred_boxes if len(b) > 0]
        valid_pred_scores = [s for s in all_pred_scores if len(s) > 0]
        
        if len(valid_pred_boxes) == 0:
            return 0.0
        
        all_boxes = np.concatenate(valid_pred_boxes)
        all_scores = np.concatenate(valid_pred_scores)
        
        # Sort by score
        sort_idx = np.argsort(-all_scores)
        all_boxes = all_boxes[sort_idx]
        all_scores = all_scores[sort_idx]
        
        # Match predictions to GT
        tp = np.zeros(len(all_boxes))
        fp = np.zeros(len(all_boxes))
        
        # Track which GT boxes have been matched
        gt_matched = {i: np.zeros(len(gt), dtype=bool) for i, gt in enumerate(all_gt_boxes)}
        
        # Build index mapping from flat predictions back to samples
        flat_to_sample = []
        for i, pred_boxes in enumerate(all_pred_boxes):
            flat_to_sample.extend([i] * len(pred_boxes))
        flat_to_sample = np.array(flat_to_sample)[sort_idx]
        
        for pred_idx, (box, score) in enumerate(zip(all_boxes, all_scores)):
            sample_idx_int = flat_to_sample[pred_idx]
            gt_boxes = all_gt_boxes[sample_idx_int]
            
            if len(gt_boxes) == 0:
                fp[pred_idx] = 1
                continue
            
            # Compute IoU with all GT boxes in this sample
            ious = box3d_iou_bev(box.reshape(1, -1), gt_boxes)
            
            # Find best matching GT
            max_iou_idx = ious.argmax()
            max_iou = ious[0, max_iou_idx]
            
            if max_iou >= iou_thr and not gt_matched[sample_idx_int][max_iou_idx]:
                tp[pred_idx] = 1
                gt_matched[sample_idx_int][max_iou_idx] = True
            else:
                fp[pred_idx] = 1
        
        # Compute precision-recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recall = tp_cumsum / num_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Compute AP using 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = precision[recall >= t]
            ap += p.max() if len(p) > 0 else 0
        ap /= 11
        
        return ap