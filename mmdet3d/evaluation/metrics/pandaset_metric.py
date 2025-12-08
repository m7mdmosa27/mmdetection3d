# mmdet3d/evaluation/metrics/pandaset_metric.py
#
# Improved PandaSet Metric for BEVFusion evaluation
# With better debugging and coordinate handling

from typing import Dict, List, Optional, Sequence
import numpy as np
import pickle
import os
import mmengine
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS
from mmdet3d.structures import LiDARInstance3DBoxes


def box3d_iou_bev(boxes_a, boxes_b):
    """Compute BEV IoU between two sets of boxes.
    
    Args:
        boxes_a: (N, 9) array of boxes [x, y, z, dx, dy, dz, yaw, vx, vy]
        boxes_b: (M, 9) array of boxes
        
    Returns:
        (N, M) IoU matrix
    """
    try:
        from shapely.geometry import Polygon
    except ImportError:
        mmengine.MMLogger.get_current_instance().warning(
            "Shapely not installed, using simple BEV IoU"
        )
        return _simple_bev_iou(boxes_a, boxes_b)
    
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    
    ious = np.zeros((len(boxes_a), len(boxes_b)))
    
    for i, box_a in enumerate(boxes_a):
        corners_a = _get_box_corners_2d(box_a)
        try:
            poly_a = Polygon(corners_a)
            if not poly_a.is_valid:
                poly_a = poly_a.buffer(0)
        except:
            continue
        
        for j, box_b in enumerate(boxes_b):
            corners_b = _get_box_corners_2d(box_b)
            try:
                poly_b = Polygon(corners_b)
                if not poly_b.is_valid:
                    poly_b = poly_b.buffer(0)
            except:
                continue
            
            try:
                intersection = poly_a.intersection(poly_b).area
                union = poly_a.union(poly_b).area
                ious[i, j] = intersection / union if union > 0 else 0
            except:
                continue
    
    return ious


def _get_box_corners_2d(box):
    """Get 2D box corners (BEV) from [x, y, z, dx, dy, dz, yaw, ...]"""
    x, y = box[0], box[1]
    dx, dy = box[3], box[4]
    yaw = box[6]
    
    # Corner offsets (centered box)
    corners = np.array([
        [dx/2, dy/2],
        [dx/2, -dy/2],
        [-dx/2, -dy/2],
        [-dx/2, dy/2]
    ])
    
    # Rotation matrix
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rot = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw, cos_yaw]
    ])
    
    # Rotate and translate
    corners = corners @ rot.T + np.array([x, y])
    return corners


def _simple_bev_iou(boxes_a, boxes_b):
    """Simple axis-aligned BEV IoU (fallback without Shapely)."""
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


@METRICS.register_module()
class PandaSetMetric(BaseMetric):
    """PandaSet evaluation metric with improved debugging.
    
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
        self.logger.info(f"PandaSetMetric initialized")
        self.logger.info(f"  - ann_file: {ann_file}")
        self.logger.info(f"  - iou_thresholds: {iou_thresholds}")
        self.logger.info(f"  - score_threshold: {score_threshold}")
        
        # Load GT annotations
        self._load_ground_truth()
    
    def _load_ground_truth(self):
        """Load ground truth from annotation file."""
        from pandas import concat
        
        self.logger.info(f"Loading GT from {self.ann_file}")
        
        if not os.path.exists(self.ann_file):
            self.logger.error(f"Annotation file not found: {self.ann_file}")
            self.gt_dict = {}
            return
        
        infos = mmengine.load(self.ann_file)
        self.logger.info(f"Loaded {len(infos)} info entries")
        
        self.gt_dict = {}
        loaded_count = 0
        failed_count = 0
        empty_count = 0
        
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
                os.path.join('data/pandaset', anno_path),
                anno_path.replace('data/pandaset/', '').replace('data\\pandaset\\', ''),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    final_path = path
                    break
            
            if final_path is None:
                failed_count += 1
                continue
            
            try:
                with open(final_path, 'rb') as f:
                    annos = pickle.load(f)
                
                # Filter for front LiDAR annotations
                anno1 = annos[annos['cuboids.sensor_id']==1]
                anno2 = annos[annos['camera_used']==0]
                annos = concat([anno1, anno2], ignore_index=True).drop_duplicates().reset_index(drop=True)
                
                boxes, labels = [], []
                for _, obj in annos.iterrows():
                    label = obj.get('label', '')
                    if label in self.class_names:
                        boxes.append([
                            obj['position.x'], obj['position.y'], obj['position.z'],
                            obj['dimensions.x'], obj['dimensions.y'], obj['dimensions.z'],
                            obj['yaw'], 0, 0  # velocity placeholders
                        ])
                        labels.append(self.class_names.index(label))
                
                if len(boxes) > 0:
                    self.gt_dict[sample_idx] = {
                        'boxes': np.array(boxes, dtype=np.float32),
                        'labels': np.array(labels, dtype=np.int64)
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
        self.logger.info(f"  - Empty: {empty_count}")
        self.logger.info(f"  - Failed: {failed_count}")
        
        # Debug: show GT statistics
        if len(self.gt_dict) > 0:
            total_boxes = sum(len(gt['boxes']) for gt in self.gt_dict.values())
            self.logger.info(f"  - Total GT boxes: {total_boxes}")
            
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
                    self.logger.info(f"  Top 5 scores: {sorted(pred_scores, reverse=True)[:5]}")
                    
                    # Show per-class predictions
                    for class_id, class_name in enumerate(self.class_names):
                        mask = pred_labels == class_id
                        n_pred = mask.sum()
                        if n_pred > 0:
                            max_score = pred_scores[mask].max()
                            self.logger.info(f"  {class_name}: {n_pred} preds, max_score={max_score:.4f}")
                
                # Compare with GT
                if sample_idx in self.gt_dict:
                    gt = self.gt_dict[sample_idx]
                    self.logger.info(f"  GT boxes: {len(gt['boxes'])}")
                    
                    # Check coordinate alignment
                    if len(pred_boxes) > 0 and len(gt['boxes']) > 0:
                        pred_centers = pred_boxes[:, :3].mean(axis=0)
                        gt_centers = gt['boxes'][:, :3].mean(axis=0)
                        self.logger.info(f"  Pred center mean: [{pred_centers[0]:.1f}, {pred_centers[1]:.1f}, {pred_centers[2]:.1f}]")
                        self.logger.info(f"  GT center mean: [{gt_centers[0]:.1f}, {gt_centers[1]:.1f}, {gt_centers[2]:.1f}]")
            
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
        
        # Check sample matching
        pred_samples = set(r['sample_idx'] for r in results)
        gt_samples = set(self.gt_dict.keys())
        matched = pred_samples & gt_samples
        
        self.logger.info(f"Prediction samples: {len(pred_samples)}")
        self.logger.info(f"GT samples: {len(gt_samples)}")
        self.logger.info(f"Matched samples: {len(matched)}")
        
        if len(matched) == 0:
            self.logger.error("NO MATCHED SAMPLES! Check sample_idx format.")
            self.logger.info(f"First 5 pred sample_idx: {list(pred_samples)[:5]}")
            self.logger.info(f"First 5 GT sample_idx: {list(gt_samples)[:5]}")
            return {f'mAP@{t}': 0.0 for t in self.iou_thresholds}
        
        # Count predictions and GT per class
        self.logger.info(f"\nPer-class statistics:")
        for class_id, class_name in enumerate(self.class_names):
            n_pred = sum(len(r['boxes'][r['labels'] == class_id]) for r in results)
            n_gt = sum(len(gt['boxes'][gt['labels'] == class_id]) 
                      for gt in self.gt_dict.values())
            self.logger.info(f"  {class_name}: {n_pred} preds, {n_gt} GT")
        
        # Compute AP for each class and threshold
        metrics = {}
        
        for iou_thr in self.iou_thresholds:
            self.logger.info(f"\n--- IoU Threshold: {iou_thr} ---")
            aps = []
            
            for class_id, class_name in enumerate(self.class_names):
                ap = self._compute_class_ap(results, class_id, iou_thr)
                aps.append(ap)
                self.logger.info(f"  {class_name} AP@{iou_thr}: {ap:.4f}")
            
            mAP = np.mean(aps)
            metrics[f'mAP@{iou_thr}'] = float(mAP)
            self.logger.info(f"  mAP@{iou_thr}: {mAP:.4f}")
        
        return metrics
    
    def _compute_class_ap(self, results, class_id, iou_thr):
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