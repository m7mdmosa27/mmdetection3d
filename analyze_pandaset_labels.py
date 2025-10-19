#!/usr/bin/env python3
"""
Analyze PandaSet labels and generate top N classes configuration.
This script counts all labels in the dataset and recommends the most common ones.
"""

import os
import pickle
import pandas as pd
from collections import Counter

def analyze_labels(data_root, info_file='pandaset_infos_train.pkl', top_n=5):
    """Analyze labels in PandaSet and return top N classes."""
    
    info_path = os.path.join(data_root, info_file)
    if not os.path.exists(info_path):
        print(f"Error: Info file not found: {info_path}")
        return None
    
    print("=" * 70)
    print(f"Analyzing PandaSet Labels ({info_file})")
    print("=" * 70)
    
    infos = pickle.load(open(info_path, 'rb'))
    print(f"\nTotal frames: {len(infos)}")
    
    # Collect all labels
    all_labels = []
    frames_with_labels = 0
    
    print("Scanning annotations...")
    for i, info in enumerate(infos):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(infos)} frames...", end='\r')
        
        anno_path = info['anno_path']
        if not os.path.isabs(anno_path):
            anno_path = os.path.join(data_root, anno_path)
        
        if not os.path.exists(anno_path):
            continue
        
        # Load annotations
        try:
            annos = pd.read_pickle(anno_path)
            if len(annos) > 0:
                frames_with_labels += 1
                for _, obj in annos.iterrows():
                    label = obj.get('label', None)
                    if label:
                        all_labels.append(label)
        except Exception as e:
            print(f"\nWarning: Failed to load {anno_path}: {e}")
    
    print(f"\n  Processed {len(infos)}/{len(infos)} frames")
    
    # Count labels
    label_counts = Counter(all_labels)
    total_objects = len(all_labels)
    
    print(f"\nTotal objects: {total_objects}")
    print(f"Frames with objects: {frames_with_labels}")
    print(f"Unique labels: {len(label_counts)}")
    
    # Show all labels with counts and percentages
    print(f"\n{'='*70}")
    print("All Labels (sorted by frequency):")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Label':<40} {'Count':<10} {'Percentage':<10}")
    print("-" * 70)
    
    for rank, (label, count) in enumerate(label_counts.most_common(), 1):
        percentage = (count / total_objects) * 100
        print(f"{rank:<6} {label:<40} {count:<10} {percentage:>6.2f}%")
    
    # Get top N classes
    top_classes = [label for label, _ in label_counts.most_common(top_n)]
    top_counts = {label: count for label, count in label_counts.most_common(top_n)}
    
    print(f"\n{'='*70}")
    print(f"Top {top_n} Classes (Recommended):")
    print(f"{'='*70}")
    
    coverage = sum(top_counts.values())
    coverage_pct = (coverage / total_objects) * 100
    
    for i, label in enumerate(top_classes, 1):
        count = top_counts[label]
        pct = (count / total_objects) * 100
        print(f"{i}. {label:<40} ({count:>6} objects, {pct:>5.1f}%)")
    
    print(f"\nCoverage: {coverage}/{total_objects} objects ({coverage_pct:.1f}%)")
    print(f"Ignored:  {total_objects - coverage}/{total_objects} objects ({100 - coverage_pct:.1f}%)")
    
    # Generate code snippet
    print(f"\n{'='*70}")
    print("Configuration for PandaSetDataset:")
    print(f"{'='*70}")
    print("\nAdd this to your pandaset_dataset.py:\n")
    
    print("METAINFO = {")
    print(f"    'classes': {tuple(top_classes)}")
    print("}")
    
    print("\n# No label mapping needed - using exact PandaSet labels")
    print("# All other labels will be automatically ignored")
    
    return top_classes, label_counts


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze PandaSet labels')
    parser.add_argument('--data-root', type=str, default='data/pandaset',
                        help='Root directory of PandaSet')
    parser.add_argument('--info-file', type=str, default='pandaset_infos_train.pkl',
                        help='Info file to analyze')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of top classes to select')
    args = parser.parse_args()
    
    top_classes, label_counts = analyze_labels(args.data_root, args.info_file, args.top_n)
    
    if top_classes:
        print(f"\n{'='*70}")
        print("Next Steps:")
        print(f"{'='*70}")
        print("1. Copy the METAINFO configuration above to pandaset_dataset.py")
        print("2. Remove or comment out the LABEL_MAPPING (not needed)")
        print("3. Run verification: python pandaset_verification.py --data-root data/pandaset")
        print("=" * 70)


if __name__ == '__main__':
    main()
