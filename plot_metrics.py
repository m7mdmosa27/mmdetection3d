import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
import sys

# ==========================================
# 1. Parsing Logic
# ==========================================

def parse_training_log(log_content):
    """
    Parse training log to extract:
      1. Training Metrics (Loss, LR, etc.)
      2. Validation Summary (mAP)
      3. Per-Class Validation Metrics (Car AP, Pedestrian AP, etc.)
    """
    # -- Data Containers --
    train_metrics = defaultdict(list)
    val_metrics = defaultdict(list)
    
    # Structure: class_metrics['Car'] = [(epoch, score), (epoch, score)...]
    class_metrics = defaultdict(list) 

    # -- Regex Patterns --
    
    # 1. Train Line: Epoch(train) [10][2450/2496] ...
    train_pattern = r'Epoch\(train\)\s+\[(\d+)\]\[\s*(\d+)/\d+\].*?lr:\s+([\d.e+-]+).*?loss:\s+([\d.]+).*?loss_heatmap:\s+([\d.]+).*?layer_-1_loss_cls:\s+([\d.]+).*?layer_-1_loss_bbox:\s+([\d.]+).*?matched_ious:\s+([\d.]+)'
    grad_pattern = r'grad_norm:\s+([\d.]+|nan)'

    # 2. Val Summary Line: Epoch(val) [16] ... pandaset/mAP@0.25: 0.1694 ...
    val_summary_pattern = r'Epoch\(val\)\s+\[(\d+)\]\[.*?pandaset/mAP@0.25:\s+([\d.]+).*?pandaset/mAP@0.5:\s+([\d.]+).*?pandaset/mAP@0.7:\s+([\d.]+)'

    # 3. Class AP Line (Specifically AP@0.5): ... INFO -   Car AP@0.5: 0.2717
    # Captures Group 1: "Car", Group 2: "0.2717"
    class_ap_pattern = r'INFO -\s+(.+?)\s+AP@0.5:\s+([\d.]+)'

    global_step = 0
    
    # Buffer to hold class scores until we know which epoch they belong to
    # Dictionary: {'Car': 0.2717, 'Pedestrian': 0.1052}
    temp_class_buffer = {} 

    lines = log_content.split('\n')
    
    for line in lines:
        # --- A. Parse Training ---
        if 'Epoch(train)' in line:
            match = re.search(train_pattern, line)
            if match:
                epoch = int(match.group(1))
                batch = int(match.group(2))
                train_metrics['epoch'].append(epoch)
                train_metrics['batch'].append(batch)
                train_metrics['lr'].append(float(match.group(3)))
                train_metrics['loss'].append(float(match.group(4)))
                train_metrics['loss_heatmap'].append(float(match.group(5)))
                train_metrics['layer_-1_loss_cls'].append(float(match.group(6)))
                train_metrics['layer_-1_loss_bbox'].append(float(match.group(7)))
                train_metrics['matched_ious'].append(float(match.group(8)))
                train_metrics['global_step'].append(global_step)
                
                grad_match = re.search(grad_pattern, line)
                val = float(grad_match.group(1)) if grad_match and grad_match.group(1) != 'nan' else float('nan')
                train_metrics['grad_norm'].append(val)

                global_step += 1

        # --- B. Parse Class Metrics (Buffer them) ---
        # We look for lines like "Car AP@0.5: 0.2717"
        elif 'AP@0.5:' in line and 'mAP' not in line: 
            # 'mAP' check prevents matching the summary line or generic headers
            match = re.search(class_ap_pattern, line)
            if match:
                class_name = match.group(1).strip()
                score = float(match.group(2))
                temp_class_buffer[class_name] = score

        # --- C. Parse Validation Summary (Trigger flush of buffer) ---
        elif 'Epoch(val)' in line and 'mAP@0.25' in line:
            val_match = re.search(val_summary_pattern, line)
            if val_match:
                epoch = int(val_match.group(1))
                
                # Store General Val Metrics
                val_metrics['epoch'].append(epoch)
                val_metrics['mAP_0.25'].append(float(val_match.group(2)))
                val_metrics['mAP_0.5'].append(float(val_match.group(3)))
                val_metrics['mAP_0.7'].append(float(val_match.group(4)))

                # Flush Class Buffer to Main Storage
                if temp_class_buffer:
                    for c_name, c_score in temp_class_buffer.items():
                        class_metrics[c_name].append((epoch, c_score))
                    temp_class_buffer = {} # Reset buffer

    # Convert lists to arrays where appropriate
    for key in train_metrics:
        train_metrics[key] = np.array(train_metrics[key])
    for key in val_metrics:
        val_metrics[key] = np.array(val_metrics[key])
        
    return train_metrics, val_metrics, class_metrics

# ==========================================
# 2. Individual Plotting Functions
# ==========================================

def get_colors(num_items):
    return plt.cm.tab10(np.linspace(0, 1, num_items))

def plot_total_loss(data, output_dir):
    print("   -> Generating Total Loss Plot...")
    epochs = data['epoch']
    unique_epochs = np.unique(epochs)
    colors = get_colors(len(unique_epochs))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, ep in enumerate(unique_epochs):
        mask = epochs == ep
        ax.plot(data['global_step'][mask], data['loss'][mask], 'o-', 
                label=f'Epoch {ep}', markersize=3, color=colors[i], alpha=0.8, linewidth=1)
    ax.set_yscale('log')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss per Step (Log Scale)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(output_dir / '1_total_loss.png', dpi=300)
    plt.close()

def plot_loss_components(data, output_dir):
    print("   -> Generating Loss Components Plot...")
    fig, ax = plt.subplots(figsize=(12, 6))
    step = data['global_step']
    ax.plot(step, data['loss_heatmap'], 'o-', label='Heatmap', markersize=2, alpha=0.6)
    ax.plot(step, data['layer_-1_loss_cls'], 's-', label='Classification', markersize=2, alpha=0.6)
    ax.plot(step, data['layer_-1_loss_bbox'], '^-', label='BBox', markersize=2, alpha=0.6)
    ax.set_yscale('log')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss Value')
    ax.set_title('Loss Components Breakdown')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '2_loss_components.png', dpi=300)
    plt.close()

def plot_matched_iou_step(data, output_dir):
    print("   -> Generating Matched IoU (Step) Plot...")
    epochs = data['epoch']
    unique_epochs = np.unique(epochs)
    colors = get_colors(len(unique_epochs))
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, ep in enumerate(unique_epochs):
        mask = epochs == ep
        ax.plot(data['global_step'][mask], data['matched_ious'][mask], 'o-',
                label=f'Epoch {ep}', markersize=3, color=colors[i], alpha=0.8, linewidth=1)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Matched IoU')
    ax.set_title('Matched IoU Progress')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_dir / '3_matched_iou_step.png', dpi=300)
    plt.close()

def plot_lr_schedule(data, output_dir):
    print("   -> Generating LR Schedule Plot...")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data['global_step'], data['lr'], '-', color='green', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.tight_layout()
    plt.savefig(output_dir / '4_lr_schedule.png', dpi=300)
    plt.close()

def plot_batch_level_metrics(data, output_dir):
    print("   -> Generating Batch-Level Detail Plots (Wide)...")
    epochs = data['epoch']
    batches = data['batch']
    unique_epochs = np.unique(epochs)
    colors = get_colors(len(unique_epochs))
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(20, 6))
    for i, ep in enumerate(unique_epochs):
        mask = epochs == ep
        ax.plot(batches[mask], data['loss_heatmap'][mask], 'o-', 
                label=f'Epoch {ep}', markersize=4, color=colors[i], alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('Batch ID')
    ax.set_ylabel('Heatmap Loss')
    ax.set_title('Heatmap Loss per Batch (Overlayed Epochs)')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '5_batch_heatmap_loss.png', dpi=300)
    plt.close()

    # BBox
    fig, ax = plt.subplots(figsize=(20, 6))
    for i, ep in enumerate(unique_epochs):
        mask = epochs == ep
        ax.plot(batches[mask], data['layer_-1_loss_bbox'][mask], 'o-', 
                label=f'Epoch {ep}', markersize=4, color=colors[i], alpha=0.7)
    ax.set_xlabel('Batch ID')
    ax.set_ylabel('BBox Loss')
    ax.set_title('BBox Loss per Batch (Overlayed Epochs)')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '6_batch_bbox_loss.png', dpi=300)
    plt.close()

def plot_epoch_distributions(data, output_dir):
    print("   -> Generating Epoch Box Plots...")
    epochs = data['epoch']
    unique_epochs = np.unique(epochs)
    colors = get_colors(len(unique_epochs))
    fig, ax = plt.subplots(figsize=(10, 6))
    epoch_losses = [data['loss'][epochs == e] for e in unique_epochs]
    bp = ax.boxplot(epoch_losses, labels=[f'E{e}' for e in unique_epochs], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Loss Distribution per Epoch')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / '7_epoch_loss_boxplot.png', dpi=300)
    plt.close()

def plot_epoch_averages_highlight(data, output_dir):
    print("   -> Generating Epoch Averages...")
    epochs = data['epoch']
    unique_epochs = np.unique(epochs)
    
    avg_loss = [np.mean(data['loss'][epochs == e]) for e in unique_epochs]
    avg_iou = [np.mean(data['matched_ious'][epochs == e]) for e in unique_epochs]
    
    # Average Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(unique_epochs, avg_loss, 's-', color='darkred', linewidth=2, label='Mean Loss')
    min_loss = min(avg_loss)
    best_loss_epoch = unique_epochs[avg_loss.index(min_loss)]
    for e, val in zip(unique_epochs, avg_loss):
        weight = 'bold' if val == min_loss else 'normal'
        ax.annotate(f'{val:.3f}', (e, val), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9, fontweight=weight)
    ax.scatter([best_loss_epoch], [min_loss], color='gold', s=150, zorder=5, marker='*', edgecolors='black', label=f'Best: {min_loss:.4f}')
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Loss')
    ax.set_title('Average Total Loss per Epoch')
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xticks(unique_epochs)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / '8_avg_epoch_loss.png', dpi=300)
    plt.close()

    # Average IoU
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(unique_epochs, avg_iou, 'o-', color='darkblue', linewidth=2, label='Mean IoU')
    max_iou = max(avg_iou)
    best_iou_epoch = unique_epochs[avg_iou.index(max_iou)]
    ax.scatter([best_iou_epoch], [max_iou], color='gold', s=200, zorder=10, marker='*', edgecolors='black', label=f'Best: {max_iou:.4f}')
    for e, val in zip(unique_epochs, avg_iou):
        if val == max_iou:
            ax.annotate(f'{val:.4f}\n(Best)', (e, val), xytext=(0, 15), textcoords='offset points', ha='center', fontsize=10, fontweight='bold', color='darkred')
        else:
            ax.annotate(f'{val:.3f}', (e, val), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average IoU')
    ax.set_title('Average Matched IoU per Epoch')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    ax.set_xticks(unique_epochs)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_dir / '8_avg_epoch_iou.png', dpi=300)
    plt.close()

def plot_iou_batch_progress(data, output_dir):
    print("   -> Generating Batch IoU Trends (Wide)...")
    epochs = data['epoch']
    batches = data['batch']
    ious = data['matched_ious']
    unique_batches = np.unique(batches)
    unique_epochs = np.unique(epochs)
    fig, ax = plt.subplots(figsize=(20, 8))
    batch_data = defaultdict(lambda: {'epochs': [], 'vals': []})
    for i in range(len(epochs)):
        batch_data[batches[i]]['epochs'].append(epochs[i])
        batch_data[batches[i]]['vals'].append(ious[i])
    batch_colors = plt.cm.get_cmap('hsv', len(unique_batches))
    sorted_batch_ids = sorted(batch_data.keys())
    for i, bid in enumerate(sorted_batch_ids):
        b_epochs = batch_data[bid]['epochs']
        b_vals = batch_data[bid]['vals']
        ax.plot(b_epochs, b_vals, 'o-', label=f'Batch {bid}', markersize=4, color=batch_colors(i), alpha=0.6, linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Matched IoU')
    ax.set_title('IoU Progression for Each Batch across Epochs')
    ax.set_xticks(unique_epochs)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(output_dir / '9_batch_iou_progress.png', dpi=300)
    plt.close()

def plot_validation_metrics(val_data, output_dir):
    print("   -> Generating Validation Accuracy (mAP) Plot...")
    epochs = val_data['epoch']
    map_50 = val_data['mAP_0.5']
    
    if len(epochs) == 0:
        print("      [Warning] No validation data found. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, val_data['mAP_0.25'], 'o-', color='green', linewidth=2, label='mAP @ 0.25')
    ax.plot(epochs, map_50, 's-', color='blue', linewidth=2, label='mAP @ 0.50')
    ax.plot(epochs, val_data['mAP_0.7'], '^-', color='orange', linewidth=2, label='mAP @ 0.70')

    max_50 = max(map_50)
    best_epoch = epochs[np.argmax(map_50)]
    ax.scatter([best_epoch], [max_50], color='gold', s=200, zorder=10, marker='*', edgecolors='black', label=f'Best mAP@0.5: {max_50:.4f}')

    for e, val in zip(epochs, map_50):
        font_weight = 'bold' if val == max_50 else 'normal'
        ax.annotate(f'{val:.3f}', (e, val), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9, fontweight=font_weight)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Average Precision (mAP)')
    ax.set_title('Validation Accuracy over Epochs')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / '10_validation_accuracy.png', dpi=300)
    plt.close()

def plot_per_class_accuracy(class_data, output_dir):
    """Plot 11: Accuracy (AP@0.5) per specific class"""
    print("   -> Generating Per-Class Accuracy Plot...")
    
    if not class_data:
        print("      [Warning] No per-class data found. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Generate distinct colors for each class
    classes = sorted(class_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

    for i, class_name in enumerate(classes):
        data_points = class_data[class_name]
        # Unzip data (epoch, score)
        epochs, scores = zip(*data_points)
        
        ax.plot(epochs, scores, 'o-', linewidth=2, label=class_name, color=colors[i])
        
        # Optional: Annotate the last point to make reading easier
        last_epoch = epochs[-1]
        last_score = scores[-1]
        ax.text(last_epoch, last_score, f' {last_score:.2f}', fontsize=8, va='center', color=colors[i])

    ax.set_xlabel('Epoch')
    ax.set_ylabel('AP @ 0.50')
    ax.set_title('Per-Class Validation Accuracy (AP @ 0.5)')
    ax.grid(True, alpha=0.3)
    
    # Combine legend with graph or put outside
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Classes")
    
    plt.tight_layout()
    plt.savefig(output_dir / '11_class_accuracy.png', dpi=300)
    plt.close()

def generate_stats_image(data, output_dir):
    print("   -> Generating Statistics Table...")
    epochs = data['epoch']
    unique_epochs = np.unique(epochs)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    lines = ["TRAINING STATISTICS SUMMARY", "="*40, f"Total Epochs: {len(unique_epochs)}", f"Total Steps:  {len(data['global_step'])}", ""]
    stats_lines = []
    for ep in unique_epochs:
        mask = epochs == ep
        loss_val = np.mean(data['loss'][mask])
        iou_val = np.mean(data['matched_ious'][mask])
        stats_lines.append(f"Epoch {ep:<3} | Loss: {loss_val:.4f} | IoU: {iou_val:.4f}")
    col_text = "\n".join(stats_lines)
    ax.text(0.05, 0.95, "\n".join(lines) + "\n" + col_text, transform=ax.transAxes, verticalalignment='top', fontfamily='monospace', fontsize=12, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
    plt.tight_layout()
    plt.savefig(output_dir / '0_stats_summary.png', dpi=300)
    plt.close()


# ==========================================
# 3. Main Logic
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Generate Training Plots from Log File")
    parser.add_argument('log_file', type=str, help='Path to log file')
    
    parser.add_argument('--all', action='store_true', help='Generate ALL plots')
    parser.add_argument('--plot_loss', action='store_true', help='Plot Total Loss')
    parser.add_argument('--plot_components', action='store_true', help='Plot Loss Components')
    parser.add_argument('--plot_iou', action='store_true', help='Plot Matched IoU (Step)')
    parser.add_argument('--plot_lr', action='store_true', help='Plot Learning Rate')
    parser.add_argument('--plot_batch', action='store_true', help='Plot Batch-level details')
    parser.add_argument('--plot_dist', action='store_true', help='Plot Epoch Distributions')
    parser.add_argument('--plot_avg', action='store_true', help='Plot Epoch Averages')
    parser.add_argument('--plot_batch_progress', action='store_true', help='Plot IoU per Batch across Epochs')
    parser.add_argument('--plot_val', action='store_true', help='Plot Validation Accuracy')
    parser.add_argument('--plot_class', action='store_true', help='Plot Per-Class Accuracy')
    parser.add_argument('--plot_stats', action='store_true', help='Generate Stats Image')
    
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: File {log_path} not found.")
        sys.exit(1)
        
    print(f"Reading log: {log_path}")
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Parse metrics (Returns three dicts now)
    train_metrics, val_metrics, class_metrics = parse_training_log(content)
    
    if len(train_metrics['epoch']) == 0:
        print("No training data found in log file.")
        sys.exit(1)
        
    print(f"Found {len(train_metrics['epoch'])} training steps.")
    if len(val_metrics['epoch']) > 0:
        print(f"Found {len(val_metrics['epoch'])} validation checkpoints.")
    if len(class_metrics) > 0:
        print(f"Found {len(class_metrics)} unique classes in validation.")
    
    output_dir = log_path.parent / 'training_plots'
    output_dir.mkdir(exist_ok=True)
    print(f"Saving plots to: {output_dir}")
    
    if args.all or args.plot_loss: plot_total_loss(train_metrics, output_dir)
    if args.all or args.plot_components: plot_loss_components(train_metrics, output_dir)
    if args.all or args.plot_iou: plot_matched_iou_step(train_metrics, output_dir)
    if args.all or args.plot_lr: plot_lr_schedule(train_metrics, output_dir)
    if args.all or args.plot_batch: plot_batch_level_metrics(train_metrics, output_dir)
    if args.all or args.plot_dist: plot_epoch_distributions(train_metrics, output_dir)
    if args.all or args.plot_avg: plot_epoch_averages_highlight(train_metrics, output_dir)
    if args.all or args.plot_batch_progress: plot_iou_batch_progress(train_metrics, output_dir)
    if args.all or args.plot_val: plot_validation_metrics(val_metrics, output_dir)
    if args.all or args.plot_class: plot_per_class_accuracy(class_metrics, output_dir)
    if args.all or args.plot_stats: generate_stats_image(train_metrics, output_dir)

    print("\n\u2713 All requested plots generated successfully.")

if __name__ == "__main__":
    main()


# how to use this script:
# python plot_metrics.py work_dirs/sec_train_with_10_epochs/20260130_222502/20260130_222502_2.log --all
# --all will generate all plots
# --plot_loss will generate the loss plot
# --plot_components will generate the loss components plot
# --plot_iou will generate the matched iou plot
# --plot_lr will generate the learning rate plot
# --plot_batch will generate the batch-level details plot
# --plot_dist will generate the epoch distributions plot
# --plot_avg will generate the epoch averages plot
# --plot_batch_progress will generate the iou per batch across epochs plot
# --plot_val will generate the validation accuracy plot
# --plot_class will generate the per-class accuracy plot
# --plot_stats will generate the statistics table