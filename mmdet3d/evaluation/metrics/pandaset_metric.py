# mmdet3d/evaluation/metrics/pandaset_metric.py
"""
Custom metric evaluator for PandaSet 3D object detection.

Computes:
- Average Precision (AP) per class at different IoU thresholds
- Mean Average Precision (mAP)
- Per-class precision and recall
"""

import numpy as np
import pickle
from typing import Dict, List, Optional, Sequence
from collections import defaultdict

import mmengine
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet3d.registry import METRICS
from mmdet3d.structures import LiDARInstance3DBoxes


def compute_iou_3d(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute 3D IoU between two boxes using simple volume overlap."""
    
    # Extract box parameters: [x, y, z, dx, dy, dz, yaw]
    x1, y1, z1, dx1, dy1, dz1, yaw1 = box1[:7]
    x2, y2, z2, dx2, dy2, dz2, yaw2 = box2[:7]
    
    # For simplicity, compute axis-aligned box overlap (ignore rotation)
    # Get box corners
    x1_min, x1_max = x1 - dx1/2, x1 + dx1/2
    y1_min, y1_max = y1 - dy1/2, y1 + dy1/2
    z1_min, z1_max = z1 - dz1/2, z1 + dz1/2
    
    x2_min, x2_max = x2 - dx2/2, x2 + dx2/2
    y2_min, y2_max = y2 - dy2/2, y2 + dy2/2
    z2_min, z2_max = z2 - dz2/2, z2 + dz2/2
    
    # Compute intersection
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    z_overlap = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
    
    intersection = x_overlap * y_overlap * z_overlap
    
    # Compute volumes
    vol1 = dx1 * dy1 * dz1
    vol2 = dx2 * dy2 * dz2
    
    # IoU = intersection / union
    union = vol1 + vol2 - intersection
    iou = intersection / (union + 1e-8)
    
    return float(iou)


def compute_ap(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """
    Compute Average Precision using 11-point interpolation.
    
    Args:
        precisions: Array of precision values
        recalls: Array of recall values
    
    Returns:
        Average Precision value
    """
    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap


@METRICS.register_module()
class PandaSetMetric(BaseMetric):
    """
    Custom metric for PandaSet 3D object detection evaluation.
    
    Computes Average Precision (AP) at different IoU thresholds and mAP.
    
    Args:
        ann_file (str): Path to annotation file (e.g., pandaset_infos_val.pkl)
        iou_thresholds (List[float]): IoU thresholds for AP calculation.
            Default: [0.5, 0.7] for easy and hard evaluation
        score_threshold (float): Minimum score for predictions. Default: 0.0
        collect_device (str): Device for collecting results. Default: 'cpu'
        prefix (str): Prefix for metric names. Default: 'pandaset'
    """
    
    default_prefix: Optional[str] = 'pandaset'
    
    def __init__(self,
                 ann_file: str,
                 iou_thresholds: List[float] = [0.5, 0.7],
                 score_threshold: float = 0.0,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        
        self.ann_file = ann_file
        self.iou_thresholds = iou_thresholds
        self.score_threshold = score_threshold
        
        # Load ground truth annotations
        self.logger = MMLogger.get_current_instance()
        self.logger.info(f'Loading PandaSet annotations from {ann_file}...')
        self.gt_annos = self._load_annotations()
        self.logger.info(f'Loaded {len(self.gt_annos)} ground truth samples')
    
    def _load_annotations(self) -> Dict:
        """Load ground truth annotations from pickle file."""
        import os
        
        if not os.path.exists(self.ann_file):
            raise FileNotFoundError(f"Annotation file not found: {self.ann_file}")
        
        infos = mmengine.load(self.ann_file)
        
        gt_annos = {}
        for info in infos:
            sample_idx = info['sample_idx']
            anno_path = info.get('anno_path', None)
            
            if anno_path is None or not os.path.exists(anno_path):
                gt_annos[sample_idx] = {
                    'boxes': np.zeros((0, 7), dtype=np.float32),
                    'labels': np.array([], dtype=np.int64)
                }
                continue
            
            # Load cuboid annotations
            annos = pickle.load(open(anno_path, 'rb'))
            
            # Filter for front-facing LiDAR (same as training)
            from pandas import concat
            lidar_type = 1
            if lidar_type == 1:
                anno1 = annos[annos['cuboids.sensor_id'] == 1]
                anno2 = annos[annos['camera_used'] == 0]
                annos = concat([anno1, anno2], ignore_index=True).drop_duplicates().reset_index(drop=True)
            
            boxes, labels = [], []
            for _, obj in annos.iterrows():
                box = [
                    obj['position.x'], obj['position.y'], obj['position.z'],
                    obj['dimensions.x'], obj['dimensions.y'], obj['dimensions.z'],
                    obj['yaw']
                ]
                boxes.append(box)
                labels.append(obj['label'])
            
            gt_annos[sample_idx] = {
                'boxes': np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 7), dtype=np.float32),
                'labels': labels if labels else []
            }
        
        return gt_annos
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """
        Process one batch of data samples and predictions.
        
        Args:
            data_batch: Dictionary containing the batch data
            data_samples: List of data samples with predictions
        """
        for data_sample in data_samples:
            result = dict()
            
            # Get sample index
            sample_idx = data_sample['sample_idx']
            
            # Get predictions
            pred_3d = data_sample['pred_instances_3d']
            pred_boxes = pred_3d['bboxes_3d'].tensor.cpu().numpy()  # [N, 9]
            pred_scores = pred_3d['scores_3d'].cpu().numpy()  # [N]
            pred_labels = pred_3d['labels_3d'].cpu().numpy()  # [N]
            
            # Filter by score threshold
            valid_mask = pred_scores >= self.score_threshold
            pred_boxes = pred_boxes[valid_mask]
            pred_scores = pred_scores[valid_mask]
            pred_labels = pred_labels[valid_mask]
            
            # Store results
            result['sample_idx'] = sample_idx
            result['pred_boxes'] = pred_boxes[:, :7]  # Only use first 7 dims
            result['pred_scores'] = pred_scores
            result['pred_labels'] = pred_labels
            
            self.results.append(result)
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            results: List of prediction results from all processes
        
        Returns:
            Dictionary of metric names and values
        """
        logger = MMLogger.get_current_instance()
        
        # Get class names from dataset metadata
        # Assuming standard PandaSet 5 classes
        class_names = [
            'Car', 'Pedestrian', 'Pedestrian with Object', 
            'Temporary Construction Barriers', 'Cones'
        ]
        num_classes = len(class_names)
        
        logger.info('Computing PandaSet 3D detection metrics...')
        logger.info(f'Number of samples: {len(results)}')
        logger.info(f'Classes: {class_names}')
        logger.info(f'IoU thresholds: {self.iou_thresholds}')
        
        # Organize predictions by class
        pred_by_class = defaultdict(list)
        gt_by_class = defaultdict(list)
        
        for result in results:
            sample_idx = result['sample_idx']
            
            # Get ground truth for this sample
            if sample_idx not in self.gt_annos:
                logger.warning(f'Sample {sample_idx} not found in ground truth')
                continue
            
            gt_data = self.gt_annos[sample_idx]
            gt_boxes = gt_data['boxes']
            gt_labels_str = gt_data['labels']
            
            # Convert string labels to indices
            gt_labels = []
            for label_str in gt_labels_str:
                if label_str in class_names:
                    gt_labels.append(class_names.index(label_str))
            gt_labels = np.array(gt_labels, dtype=np.int64)
            
            # Filter GT boxes by valid labels
            if len(gt_labels) > 0:
                valid_gt_mask = np.array([i < len(gt_labels) for i in range(len(gt_boxes))])
                gt_boxes = gt_boxes[valid_gt_mask]
            
            # Store predictions by class
            for box, score, label in zip(result['pred_boxes'], result['pred_scores'], result['pred_labels']):
                if label < num_classes:
                    pred_by_class[label].append({
                        'box': box,
                        'score': score,
                        'sample_idx': sample_idx
                    })
            
            # Store ground truth by class
            for box, label in zip(gt_boxes, gt_labels):
                if label < num_classes:
                    gt_by_class[label].append({
                        'box': box,
                        'sample_idx': sample_idx,
                        'matched': False
                    })
        
        # Compute AP for each class and IoU threshold
        metrics = {}
        all_aps = []
        
        for iou_thr in self.iou_thresholds:
            aps_at_iou = []
            
            for class_idx in range(num_classes):
                class_name = class_names[class_idx]
                
                preds = pred_by_class[class_idx]
                gts = gt_by_class[class_idx]
                
                if len(gts) == 0:
                    # No ground truth for this class
                    logger.info(f'{class_name}: No ground truth samples')
                    continue
                
                if len(preds) == 0:
                    # No predictions for this class
                    ap = 0.0
                    logger.info(f'{class_name} @ IoU={iou_thr:.2f}: AP = {ap:.4f} (no predictions)')
                    metrics[f'{class_name}_AP_{iou_thr:.2f}'] = ap
                    aps_at_iou.append(ap)
                    continue
                
                # Sort predictions by score (descending)
                preds = sorted(preds, key=lambda x: x['score'], reverse=True)
                
                # Compute precision and recall
                tp = np.zeros(len(preds))
                fp = np.zeros(len(preds))
                
                # Reset matched flags for each IoU threshold
                for gt in gts:
                    gt['matched'] = False
                
                for pred_idx, pred in enumerate(preds):
                    # Find best matching ground truth
                    best_iou = 0.0
                    best_gt_idx = -1
                    
                    for gt_idx, gt in enumerate(gts):
                        if gt['sample_idx'] != pred['sample_idx']:
                            continue
                        if gt['matched']:
                            continue
                        
                        iou = compute_iou_3d(pred['box'], gt['box'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    if best_iou >= iou_thr and best_gt_idx >= 0:
                        tp[pred_idx] = 1
                        gts[best_gt_idx]['matched'] = True
                    else:
                        fp[pred_idx] = 1
                
                # Compute cumulative precision and recall
                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(fp)
                
                recalls = tp_cumsum / len(gts)
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
                
                # Compute AP
                ap = compute_ap(precisions, recalls)
                
                logger.info(f'{class_name} @ IoU={iou_thr:.2f}: AP = {ap:.4f} '
                           f'(GT: {len(gts)}, Pred: {len(preds)}, TP: {int(tp.sum())})')
                
                metrics[f'{class_name}_AP_{iou_thr:.2f}'] = ap
                aps_at_iou.append(ap)
            
            # Compute mAP for this IoU threshold
            if aps_at_iou:
                mAP = np.mean(aps_at_iou)
                metrics[f'mAP_{iou_thr:.2f}'] = mAP
                all_aps.extend(aps_at_iou)
                logger.info(f'mAP @ IoU={iou_thr:.2f}: {mAP:.4f}')
        
        # Overall mAP across all IoU thresholds
        if all_aps:
            metrics['mAP'] = np.mean(all_aps)
            logger.info(f'Overall mAP: {metrics["mAP"]:.4f}')
        
        return metrics
