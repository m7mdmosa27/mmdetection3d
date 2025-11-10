# mmdet3d/evaluation/metrics/pandaset_metric.py

from typing import Dict, List, Optional, Sequence
import numpy as np
import pickle
import os
import mmengine
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS
from mmdet3d.structures import LiDARInstance3DBoxes


def box3d_iou(boxes_a, boxes_b):
    """Compute 3D IoU (simplified BEV IoU)"""
    from shapely.geometry import Polygon
    
    ious = np.zeros((len(boxes_a), len(boxes_b)))
    for i, box_a in enumerate(boxes_a):
        corners_a = get_box_corners_2d(box_a)
        poly_a = Polygon(corners_a)
        
        for j, box_b in enumerate(boxes_b):
            corners_b = get_box_corners_2d(box_b)
            poly_b = Polygon(corners_b)
            
            if not poly_a.is_valid or not poly_b.is_valid:
                continue
                
            intersection = poly_a.intersection(poly_b).area
            union = poly_a.union(poly_b).area
            ious[i, j] = intersection / union if union > 0 else 0
    
    return ious


def get_box_corners_2d(box):
    """Get 2D box corners (BEV) from [x, y, z, dx, dy, dz, yaw, ...]"""
    x, y, dx, dy, yaw = box[0], box[1], box[3], box[4], box[6]
    
    # Corner offsets
    corners = np.array([
        [dx/2, dy/2], [dx/2, -dy/2], 
        [-dx/2, -dy/2], [-dx/2, dy/2]
    ])
    
    # Rotation matrix
    rot = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])
    
    # Rotate and translate
    corners = corners @ rot.T + np.array([x, y])
    return corners


# mmdet3d/evaluation/metrics/pandaset_metric.py

@METRICS.register_module()
class PandaSetMetric(BaseMetric):
    def __init__(self,
                 ann_file: str,
                 iou_thresholds: List[float] = [0.5, 0.7],
                 score_threshold: float = 0.25,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ann_file = ann_file
        self.iou_thresholds = iou_thresholds
        self.score_threshold = score_threshold
        self.class_names = ['Car', 'Pedestrian', 'Pedestrian with Object', 
                           'Temporary Construction Barriers', 'Cones']
        
        self.logger = mmengine.MMLogger.get_current_instance()
        self.logger.info(f"Loading GT from {ann_file}")
        infos = mmengine.load(ann_file)
        
        self.logger.info(f"Loaded {len(infos)} info entries")
        
        # Check first entry structure
        if len(infos) > 0:
            self.logger.info(f"First sample_idx: {infos[0].get('sample_idx')}")
            self.logger.info(f"First anno_path: {infos[0].get('anno_path')}")
        
        self.gt_dict = {}
        loaded_count = 0
        failed_count = 0
        
        for info in infos:
            sample_idx = str(info.get('sample_idx', ''))
            anno_path = info.get('anno_path', None)
            
            if not anno_path:
                continue
            
            # Try multiple path resolution strategies
            if os.path.exists(anno_path):
                final_path = anno_path
            elif os.path.exists(os.path.join('data/pandaset', anno_path)):
                final_path = os.path.join('data/pandaset', anno_path)
            else:
                # Try without data/pandaset prefix if anno_path starts with it
                if anno_path.startswith('data/pandaset'):
                    final_path = anno_path
                elif anno_path.startswith('data\\pandaset'):
                    final_path = anno_path
                else:
                    failed_count += 1
                    if failed_count <= 3:  # Show first 3 failures
                        self.logger.warning(f"Cannot find: {anno_path}")
                    continue
            
            if not os.path.exists(final_path):
                failed_count += 1
                continue
                
            try:
                with open(final_path, 'rb') as f:
                    annos = pickle.load(f)
                
                # Filter by lidar type
                anno1 = annos[annos['cuboids.sensor_id']==1]
                anno2 = annos[annos['camera_used']==0]
                from pandas import concat
                annos = concat([anno1, anno2], ignore_index=True).drop_duplicates().reset_index(drop=True)
                
                boxes, labels = [], []
                for _, obj in annos.iterrows():
                    label = obj.get('label', '')
                    if label in self.class_names:
                        boxes.append([
                            obj['position.x'], obj['position.y'], obj['position.z'],
                            obj['dimensions.x'], obj['dimensions.y'], obj['dimensions.z'],
                            obj['yaw'], 0, 0
                        ])
                        labels.append(self.class_names.index(label))
                
                if len(boxes) > 0:
                    self.gt_dict[sample_idx] = {
                        'boxes': np.array(boxes, dtype=np.float32),
                        'labels': np.array(labels, dtype=np.int64)
                    }
                    loaded_count += 1
            except Exception as e:
                self.logger.warning(f"Error loading {final_path}: {e}")
                failed_count += 1
        
        self.logger.info(f"Loaded GT for {loaded_count} samples ({failed_count} failed)")
        
        if len(self.gt_dict) > 0:
            sample_keys = list(self.gt_dict.keys())[:3]
            self.logger.info(f"Sample GT keys: {sample_keys}")
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            sample_idx = str(data_sample.get('sample_idx', ''))
            
            pred_instances = data_sample['pred_instances_3d']
            pred_boxes = pred_instances['bboxes_3d'].tensor.cpu().numpy()
            pred_scores = pred_instances['scores_3d'].cpu().numpy()
            pred_labels = pred_instances['labels_3d'].cpu().numpy()
            
            # DEBUG: Show score distribution for first few samples
            if len(self.results) < 3:
                self.logger.info(f"Sample {sample_idx}:")
                self.logger.info(f"  Score range: [{pred_scores.min():.4f}, {pred_scores.max():.4f}]")
                self.logger.info(f"  Scores > 0.01: {(pred_scores > 0.01).sum()}")
                self.logger.info(f"  Scores > 0.05: {(pred_scores > 0.05).sum()}")
                self.logger.info(f"  Top 5 scores: {sorted(pred_scores, reverse=True)[:5]}")
            
            # Filter by score
            keep = pred_scores >= self.score_threshold
            
            self.results.append({
                'sample_idx': sample_idx,
                'boxes': pred_boxes[keep],
                'scores': pred_scores[keep],
                'labels': pred_labels[keep]
            })

    def compute_metrics(self, results: list) -> Dict[str, float]:
        self.logger.info(f"\nComputing metrics for {len(results)} samples...")
        
        # DEBUG: Check sample_idx matches
        pred_samples = set(r['sample_idx'] for r in results)
        gt_samples = set(self.gt_dict.keys())
        matched = pred_samples & gt_samples
        
        self.logger.info(f"Prediction samples: {len(pred_samples)}")
        self.logger.info(f"GT samples: {len(gt_samples)}")
        self.logger.info(f"Matched samples: {len(matched)}")
        
        if len(matched) > 0:
            self.logger.info(f"First 3 matched: {list(matched)[:3]}")
        else:
            self.logger.info(f"First 3 pred samples: {list(pred_samples)[:3]}")
            self.logger.info(f"First 3 GT samples: {list(gt_samples)[:3]}")
        
        # Count total predictions and GT per class
        for class_id, class_name in enumerate(self.class_names):
            n_pred = sum(len(r['boxes'][r['labels'] == class_id]) for r in results)
            n_gt = sum(len(gt['boxes'][gt['labels'] == class_id]) 
                    for gt in self.gt_dict.values())
            self.logger.info(f"{class_name}: {n_pred} preds, {n_gt} GT")
        
        metrics = {}
        
        for iou_thr in self.iou_thresholds:
            aps = []
            
            for class_id, class_name in enumerate(self.class_names):
                # Collect all predictions and GT for this class
                all_pred_boxes, all_pred_scores = [], []
                all_gt_boxes = []
                
                for result in results:
                    sample_idx = result['sample_idx']
                    
                    # Predictions
                    mask = result['labels'] == class_id
                    pred_boxes = result['boxes'][mask]
                    pred_scores = result['scores'][mask]
                    
                    # GT
                    if sample_idx in self.gt_dict:
                        gt = self.gt_dict[sample_idx]
                        gt_mask = gt['labels'] == class_id
                        gt_boxes = gt['boxes'][gt_mask]
                    else:
                        gt_boxes = np.array([])
                    
                    all_pred_boxes.append(pred_boxes)
                    all_pred_scores.append(pred_scores)
                    all_gt_boxes.append(gt_boxes)
                
                # Compute AP
                ap = self._compute_ap(all_pred_boxes, all_pred_scores, all_gt_boxes, iou_thr)
                aps.append(ap)
                self.logger.info(f"{class_id}  {class_name} AP@{iou_thr}: {ap:.4f}")
            
            mAP = np.mean(aps)
            metrics[f'mAP@{iou_thr}'] = float(mAP)
            self.logger.info(f"\nmAP@{iou_thr}: {mAP:.4f}\n")
        
        return metrics
    
    def _compute_ap(self, pred_boxes_list, pred_scores_list, gt_boxes_list, iou_thr):
        """Compute AP for one class"""
        # Count total GT
        num_gt = sum(len(gt) for gt in gt_boxes_list)
        
        if num_gt == 0:
            return 0.0
        
        # Flatten predictions
        valid_scores = [s for s in pred_scores_list if len(s) > 0]
        valid_boxes = [b for b in pred_boxes_list if len(b) > 0]
        
        if len(valid_scores) == 0:
            return 0.0  # No predictions for this class
        
        all_scores = np.concatenate(valid_scores)
        all_boxes = np.concatenate(valid_boxes)
        
        # Sort by confidence
        sort_idx = np.argsort(-all_scores)
        all_boxes = all_boxes[sort_idx]
        all_scores = all_scores[sort_idx]
        
        # Match predictions to GT (simple greedy matching)
        tp = np.zeros(len(all_boxes))
        pred_count = 0
        
        for sample_idx, (pred_box_list, gt_boxes) in enumerate(zip(pred_boxes_list, gt_boxes_list)):
            n_pred = len(pred_box_list)
            if n_pred == 0:
                continue
            
            # Get predictions for this sample
            sample_preds = all_boxes[pred_count:pred_count + n_pred]
            
            if len(gt_boxes) > 0:
                # Compute IoU between predictions and GT
                ious = box3d_iou(sample_preds, gt_boxes)
                
                # Greedy matching: each pred gets best GT
                for i in range(len(sample_preds)):
                    if ious.shape[1] > 0:
                        max_iou = ious[i].max()
                        if max_iou >= iou_thr:
                            tp[pred_count + i] = 1
            
            pred_count += n_pred
        
        # Compute precision-recall
        tp_cumsum = np.cumsum(tp)
        recall = tp_cumsum / num_gt
        precision = tp_cumsum / np.arange(1, len(tp) + 1)
        
        # Compute AP (11-point interpolation)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = precision[recall >= t]
            ap += p.max() if len(p) > 0 else 0
        ap /= 11
        
        return ap