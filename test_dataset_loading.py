# test_dataset_loading_fixed.py
import sys
sys.path.insert(0, '.')

from mmengine import Config

# Load config first - this triggers custom_imports
cfg = Config.fromfile('projects/BEVFusion/configs/bevfusion_pandaset.py')

# NOW import and check registration
from mmdet3d.datasets.pandaset_dataset import PandaSetDataset
from mmdet3d.registry import DATASETS

print(f"PandaSetDataset registered in mmdet3d.registry: {'PandaSetDataset' in DATASETS}")
print(f"Available datasets: {list(DATASETS.module_dict.keys())[:10]}...")  # Show first 10

# Try to build
try:
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    print(f'\n✅ Dataset built successfully!')
    print(f'Dataset length: {len(dataset)}')
    print(f'Dataset type: {type(dataset)}')
    
    # Try loading first sample
    print(f'\nTrying to load first sample...')
    sample = dataset[0]
    
    if sample is None:
        print('❌ Sample is None (filtered out by filter_empty_gt)!')
    else:
        print('✅ Sample loaded')
        if isinstance(sample, dict) and 'data_samples' in sample:
            ds = sample['data_samples']
            n_objs = len(ds.gt_instances_3d.labels_3d)
            print(f'GT instances: {n_objs} objects')
            print(f'Points shape: {sample["inputs"]["points"].shape}')
            
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()