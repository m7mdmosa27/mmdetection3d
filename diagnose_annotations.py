#!/usr/bin/env python3
"""
Diagnose why annotations are being filtered out.
"""

import os
import pickle
import pandas as pd
from collections import Counter

# Load info file
data_root = 'data/pandaset'
train_pkl = os.path.join(data_root, 'pandaset_infos_train.pkl')
train_infos = pickle.load(open(train_pkl, 'rb'))

print("=" * 70)
print("Annotation Label Analysis")
print("=" * 70)

# Define your class names
class_names = ('Car', 'Truck', 'Bus', 'Pedestrian', 'Cyclist', 'Motorcycle')

print(f"\nYour defined classes: {class_names}")
print(f"\nAnalyzing {len(train_infos)} samples...")

# Collect all labels from all annotations
all_labels = []
matched_labels = []
unmatched_labels = []

for i, info in enumerate(train_infos[:20]):  # Check first 20 samples
    anno_path = info['anno_path']
    if not os.path.isabs(anno_path):
        anno_path = os.path.join(data_root, anno_path)
    
    if not os.path.exists(anno_path):
        continue
    
    # Load annotations
    annos = pd.read_pickle(anno_path)
    
    for _, obj in annos.iterrows():
        label = obj.get('label', None)
        all_labels.append(label)
        
        if label in class_names:
            matched_labels.append(label)
        else:
            unmatched_labels.append(label)

print(f"\nTotal objects in first 20 frames: {len(all_labels)}")
print(f"Matched objects: {len(matched_labels)}")
print(f"Unmatched objects: {len(unmatched_labels)}")

# Show label distribution
print(f"\n{'='*70}")
print("All labels found in dataset:")
print(f"{'='*70}")
label_counts = Counter(all_labels)
for label, count in label_counts.most_common():
    matched = "✓" if label in class_names else "✗"
    print(f"{matched} {label:40s}: {count:4d}")

print(f"\n{'='*70}")
print("Recommended class mapping:")
print(f"{'='*70}")

# Suggest mappings
mapping_suggestions = {
    'Pickup Truck': 'Truck',
    'Semi-truck': 'Truck',
    'Other Vehicle - Uncommon': 'Car',
    'Other Vehicle - Construction Vehicle': 'Truck',
    'Personal Mobility Device': 'Cyclist',
    'Motorized Scooter': 'Cyclist',
}

print("\nOption 1: Expand your class list to include all PandaSet classes")
print("Option 2: Create a mapping dictionary like this:\n")
print("label_mapping = {")
for original, mapped in mapping_suggestions.items():
    if original in [l for l, _ in label_counts.most_common()]:
        print(f"    '{original}': '{mapped}',")
print("}")

print("\n" + "=" * 70)
print("How to fix:")
print("=" * 70)
print("1. Update METAINFO['classes'] to include all PandaSet classes, OR")
print("2. Add label mapping in parse_ann_info() to map PandaSet labels to your classes")
print("=" * 70)
