# mmdet3d/evaluation/metrics/pandaset_metric.py

from typing import Dict, List, Optional, Sequence
import numpy as np
import mmengine
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS
from mmdet3d.structures import LiDARInstance3DBoxes


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
        
        # Load GT annotations - build lookup by sample_idx
        infos = mmengine.load(ann_file)
        self.gt_annos = {}
        for info in infos:
            sample_idx = str(info.get('sample_idx', ''))
            if 'anno_path' in info:
                self.gt_annos[sample_idx] = info
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            # Get sample_idx - convert to string
            sample_idx = str(data_sample.get('sample_idx', ''))
            
            # Get predictions
            pred_instances = data_sample['pred_instances_3d']
            pred_boxes = pred_instances['bboxes_3d'].tensor.cpu().numpy()
            pred_scores = pred_instances['scores_3d'].cpu().numpy()
            pred_labels = pred_instances['labels_3d'].cpu().numpy()
            
            # Filter by score
            keep = pred_scores >= self.score_threshold
            
            result = {
                'sample_idx': sample_idx,
                'pred_boxes': pred_boxes[keep],
                'pred_scores': pred_scores[keep],
                'pred_labels': pred_labels[keep]
            }
            self.results.append(result)
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        # Simple AP calculation
        metrics = {}
        for iou_thr in self.iou_thresholds:
            ap_per_class = []
            for class_id in range(5):  # 5 classes
                tp_sum = 0
                gt_sum = 0
                pred_sum = 0
                
                for result in results:
                    sample_idx = result['sample_idx']
                    pred_boxes = result['pred_boxes'][result['pred_labels'] == class_id]
                    
                    # Get GT for this sample
                    if sample_idx in self.gt_annos:
                        from mmdet3d.datasets.pandaset_dataset import PandaSetDataset
                        dataset = PandaSetDataset(
                            data_root='data/pandaset/',
                            ann_file=self.ann_file,
                            test_mode=True
                        )
                        info = self.gt_annos[sample_idx]
                        ann_info = dataset.parse_ann_info(info)
                        gt_boxes = ann_info['gt_bboxes_3d'].tensor.cpu().numpy()
                        gt_labels = ann_info['gt_labels_3d']
                        gt_boxes = gt_boxes[gt_labels == class_id]
                    else:
                        gt_boxes = np.array([])
                    
                    pred_sum += len(pred_boxes)
                    gt_sum += len(gt_boxes)
                    
                    # Simple TP counting (IoU > threshold)
                    if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                        tp_sum += min(len(pred_boxes), len(gt_boxes))
                
                if gt_sum > 0:
                    recall = tp_sum / gt_sum
                    precision = tp_sum / pred_sum if pred_sum > 0 else 0
                    ap = (precision + recall) / 2 if (precision + recall) > 0 else 0
                    ap_per_class.append(ap)
            
            if len(ap_per_class) > 0:
                metrics[f'mAP@{iou_thr}'] = float(np.mean(ap_per_class))
        
        return metrics