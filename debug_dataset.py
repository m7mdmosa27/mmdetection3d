# debug_dataset.py
import torch
from mmengine.config import Config
from mmdet3d.registry import DATASETS

# Load your config
cfg = Config.fromfile('projects/BEVFusion/configs/bevfusion_pandaset_all_cameras.py')

# Build dataset
dataset = DATASETS.build(cfg.train_dataloader.dataset)

# Test specific samples around where the error occurs
# Batch 800 with batch_size 4 = samples around 3200
problematic_samples = []
print("Dataset length:", len(dataset))
for i in range(800, 900):  # Test around sample 3200
    try:
        data = dataset[i]
        # if i == 1:
        #     print(f"Sample {i} data: {data}")
        if data is None:
            print(f"Sample {i} returned None")
            problematic_samples.append(i)
        else:
            # Check for empty tensors
            if 'points' in data['inputs']:
                if len(data['inputs']['points']) == 0:
                    print(f"Sample {i} has empty points")
                    problematic_samples.append(i)
    except Exception as e:
        print(f"Sample {i} failed: {e}")
        problematic_samples.append(i)

print(f"\nProblematic samples: {problematic_samples}")