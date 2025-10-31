import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc, classification_report
)
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

from modules import PyramidGFNet, pyramid_gfnet_tiny, pyramid_gfnet_small, pyramid_gfnet_base
from dataset import get_data_loaders, AutoCropBlack
from torchvision import transforms

# Try to import old GFNet for backward compatibility
try:
    from modules import GFNet
except ImportError:
    GFNet = None


def calculate_scan_level_metrics(df_results):
    """Calculate metrics at scan level using majority voting."""
    scan_results = df_results.groupby('scan_id').agg({
        'y_true': 'first',
        'y_pred': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'y_prob': 'mean'
    }).reset_index()
    
    y_true_scan = scan_results['y_true'].values
    y_pred_scan = scan_results['y_pred'].values
    y_prob_scan = scan_results['y_prob'].values
    
    accuracy = accuracy_score(y_true_scan, y_pred_scan)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_scan, y_pred_scan, average='binary', zero_division=0
    )
    
    cm = confusion_matrix(y_true_scan, y_pred_scan)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    try:
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(y_true_scan, y_prob_scan)
    except:
        roc_auc = None
    
    return {
        'scan_accuracy': accuracy,
        'scan_precision': precision,
        'scan_recall': recall,
        'scan_f1': f1,
        'scan_specificity': specificity,
        'scan_auc': roc_auc,
        'scan_confusion_matrix': cm,
        'num_scans': len(scan_results),
        'scan_predictions': scan_results
    }


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    is_pyramid = 'stages.0.blocks.0.gamma_1' in state_dict or 'merges.0.reduction.weight' in state_dict
    is_old_gfnet = 'blocks.0.gamma_1' in state_dict and not is_pyramid
    
    model_config = checkpoint.get('model_config', {})
    img_size = model_config.get('img_size', 224)
    num_classes = model_config.get('num_classes', 2)
    
    if is_pyramid:
        model_size = model_config.get('model_size', 'small')
        print(f"Loading model: Pyramid GFNet {model_size.upper()}")
        print(f"Image size: {img_size}, Classes: {num_classes}")
        
        if model_size == 'tiny':
            model = pyramid_gfnet_tiny(img_size=img_size, num_classes=num_classes)
        elif model_size == 'small':
            model = pyramid_gfnet_small(img_size=img_size, num_classes=num_classes)
        elif model_size == 'base':
            model = pyramid_gfnet_base(img_size=img_size, num_classes=num_classes)
        else:
            model = PyramidGFNet(
                img_size=img_size,
                num_classes=num_classes,
                embed_dims=model_config.get('embed_dims', [64, 128, 256, 512]),
                depths=model_config.get('depths', [3, 4, 6, 3])
            )
    
    elif is_old_gfnet:
        print(f"Loading model: OLD GFNet (non-pyramid)")
        if GFNet is None:
            raise ImportError("Old GFNet model detected but GFNet class not found")
        
        embed_dim = state_dict['norm.weight'].shape[0]
        depth = sum(1 for k in state_dict.keys() if k.startswith('blocks.') and 'gamma_1' in k)
        
        model = GFNet(
            img_size=img_size, patch_size=16, in_chans=1,
            num_classes=num_classes, embed_dim=embed_dim, depth=depth,
            mlp_ratio=3.0, drop_rate=0.3, drop_path_rate=0.4
        )
    else:
        raise ValueError("Unable to detect model architecture from checkpoint")
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Trained for {checkpoint['epoch']+1} epochs")
    if 'val_acc' in checkpoint:
        print(f"  Validation Accuracy: {checkpoint['val_acc']:.4f}")
    
    return model, checkpoint


def visualize_model_complete(model, test_loader, device, save_dir):
    """
    Create ONE comprehensive visualization combining:
    - Frequency filters (learned weights)
    - Feature maps (averaged across test set)
    - Frequency attention (averaged across test set)
    
    No image path needed - uses the entire test set!
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    print("\nGenerating comprehensive model visualization...")
    
    # ==================== COLLECT FILTERS (Model Weights) ====================
    print("  Extracting frequency filters...")
    all_filters = []
    stage_names = []
    
    if hasattr(model, 'stages'):
        for stage_idx in range(len(model.stages)):
            stage = model.stages[stage_idx]
            block = stage.blocks[0]
            complex_weight = block.filter.complex_weight.detach().cpu()
            weight = torch.view_as_complex(complex_weight)
            magnitude = torch.abs(weight).numpy()
            magnitude_avg = magnitude.mean(axis=-1)
            all_filters.append(magnitude_avg)
            stage_names.append(f'Stage {stage_idx + 1}')
    elif hasattr(model, 'blocks'):
        for block_idx in range(min(4, len(model.blocks))):
            block = model.blocks[block_idx]
            complex_weight = block.filter.complex_weight.detach().cpu()
            weight = torch.view_as_complex(complex_weight)
            magnitude = torch.abs(weight).numpy()
            magnitude_avg = magnitude.mean(axis=-1)
            all_filters.append(magnitude_avg)
            stage_names.append(f'Block {block_idx + 1}')
    
    # ==================== COLLECT FEATURES & ATTENTION (Averaged) ====================
    print("  Computing average feature maps and attention from test set...")
    
    all_features_sum = []
    all_attention_sum = []
    sample_count = 0
    max_samples = 50  # Average over first 50 samples for speed
    
    # Hook to capture frequency attention
    freq_features = []
    
    def hook_fn(module, input, output):
        x = input[0]
        B, N, C = x.shape
        a = b = int(np.sqrt(N))
        x_reshaped = x.view(B, a, b, C)
        x_fft = torch.fft.rfft2(x_reshaped, dim=(1, 2), norm='ortho')
        freq_features.append(torch.abs(x_fft).detach().cpu())
    
    # Register hooks
    hooks = []
    if hasattr(model, 'stages'):
        for stage_idx in range(len(model.stages)):
            hook = model.stages[stage_idx].blocks[0].filter.register_forward_hook(hook_fn)
            hooks.append(hook)
    elif hasattr(model, 'blocks'):
        for block_idx in range(min(4, len(model.blocks))):
            hook = model.blocks[block_idx].filter.register_forward_hook(hook_fn)
            hooks.append(hook)
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            if sample_count >= max_samples:
                break
            
            images = batch_data[0].to(device)  # Handle both (img, label) and (img, label, scan_id)
            freq_features.clear()
            
            # Get features at each stage
            if hasattr(model, 'stages'):
                x, (H, W) = model.patch_embed1(images)
                x = x + model.pos_embed1
                x = model.pos_drop(x)
                
                for stage_idx in range(len(model.stages)):
                    x = model.stages[stage_idx](x, H, W)
                    
                    # Extract spatial features (first channel, averaged over batch)
                    B, N, C = x.shape
                    x_grid = x.reshape(B, H, W, C)[:, :, :, 0].cpu().numpy()
                    x_avg = x_grid.mean(axis=0)
                    
                    if len(all_features_sum) <= stage_idx:
                        all_features_sum.append(x_avg)
                    else:
                        all_features_sum[stage_idx] += x_avg
                    
                    # Merge for next stage
                    if stage_idx < len(model.merges):
                        x, H, W = model.merges[stage_idx](x, H, W)
            
            # Forward pass triggers hooks
            _ = model(images)
            
            # Accumulate attention
            for stage_idx, freq_feat in enumerate(freq_features):
                freq_magnitude = freq_feat.mean(dim=0).mean(dim=-1).numpy()
                
                if len(all_attention_sum) <= stage_idx:
                    all_attention_sum.append(freq_magnitude)
                else:
                    all_attention_sum[stage_idx] += freq_magnitude
            
            sample_count += images.size(0)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Average the accumulated features and attention
    all_features_avg = [f / (sample_count / test_loader.batch_size) for f in all_features_sum]
    all_attention_avg = [a / (sample_count / test_loader.batch_size) for a in all_attention_sum]
    
    print(f"  Averaged over {sample_count} samples")
    
    # ==================== CREATE COMBINED PLOT ====================
    num_stages = len(all_filters)
    fig = plt.figure(figsize=(5 * num_stages, 12))
    gs = fig.add_gridspec(3, num_stages, hspace=0.3, wspace=0.3)
    
    # Row 1: Frequency Filters
    for i in range(num_stages):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(all_filters[i], cmap='viridis', aspect='auto')
        ax.set_title(f'{stage_names[i]}\nLearned Filter', fontsize=10, fontweight='bold')
        ax.set_xlabel('Frequency (Width)', fontsize=9)
        ax.set_ylabel('Frequency (Height)', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Row 2: Average Feature Maps
    for i in range(min(num_stages, len(all_features_avg))):
        ax = fig.add_subplot(gs[1, i])
        im = ax.imshow(all_features_avg[i], cmap='viridis')
        ax.set_title(f'{stage_names[i]}\nAvg Features', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Row 3: Average Frequency Attention
    for i in range(min(num_stages, len(all_attention_avg))):
        ax = fig.add_subplot(gs[2, i])
        im = ax.imshow(all_attention_avg[i], cmap='hot', aspect='auto')
        ax.set_title(f'{stage_names[i]}\nAvg Attention', fontsize=10, fontweight='bold')
        ax.set_xlabel('Frequency (Width)', fontsize=9)
        ax.set_ylabel('Frequency (Height)', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Model Frequency Analysis (Averaged over {sample_count} test samples)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    save_path = save_dir / 'model_frequency_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Model frequency analysis saved: {save_path}")
    return str(save_path)


def plot_misclassified_samples(test_loader, all_labels, all_preds, all_probs, all_scan_ids, save_dir, max_samples=20):
    """Plot misclassified SCANS (at scan-level after majority voting)."""
    save_dir = Path(save_dir)
    
    class_names = ['AD', 'NC']
    
    # Create DataFrame for scan-level aggregation
    df_results = pd.DataFrame({
        'scan_id': all_scan_ids,
        'y_true': all_labels,
        'y_pred': all_preds,
        'y_prob': all_probs
    })
    
    # Aggregate to scan level using majority voting
    scan_results = df_results.groupby('scan_id').agg({
        'y_true': 'first',
        'y_pred': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'y_prob': 'mean'
    }).reset_index()
    
    # Find misclassified SCANS
    misclassified_scans = scan_results[scan_results['y_true'] != scan_results['y_pred']]
    
    if len(misclassified_scans) == 0:
        print("\n🎉 No misclassified scans! Perfect scan-level accuracy!")
        return
    
    print(f"\n⚠️  Found {len(misclassified_scans)} misclassified SCANS (out of {len(scan_results)} total scans)")
    
    # Get images from test_loader
    all_images = []
    for batch_data in test_loader:
        images = batch_data[0]
        all_images.append(images)
    all_images = torch.cat(all_images, dim=0)
    
    # For each misclassified scan, find representative slices
    num_scans_to_show = min(max_samples, len(misclassified_scans))
    cols = 5
    rows = (num_scans_to_show + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    axes = axes.flatten() if num_scans_to_show > 1 else [axes]
    
    for idx, ax in enumerate(axes):
        if idx < num_scans_to_show:
            scan_row = misclassified_scans.iloc[idx]
            scan_id = scan_row['scan_id']
            scan_true = int(scan_row['y_true'])
            scan_pred = int(scan_row['y_pred'])
            scan_prob = scan_row['y_prob']
            
            # Find all slices from this scan
            scan_slice_indices = df_results[df_results['scan_id'] == scan_id].index.tolist()
            
            # Get a representative slice (middle slice)
            middle_idx = scan_slice_indices[len(scan_slice_indices) // 2]
            img = all_images[middle_idx].squeeze().cpu().numpy()
            
            # Count how many slices voted for each class
            scan_slices = df_results[df_results['scan_id'] == scan_id]
            votes_ad = (scan_slices['y_pred'] == 0).sum()
            votes_nc = (scan_slices['y_pred'] == 1).sum()
            total_slices = len(scan_slices)
            
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            
            title = f"Scan: {scan_id}\n"
            title += f"True: {class_names[scan_true]} | Pred: {class_names[scan_pred]}\n"
            title += f"Votes: AD={votes_ad}, NC={votes_nc} (of {total_slices})\n"
            title += f"Avg Prob: {scan_prob:.2%}"
            ax.set_title(title, fontsize=7, color='red', fontweight='bold')
        else:
            ax.axis('off')
    
    plt.suptitle(f'Misclassified SCANS (Scan-Level) - Showing {num_scans_to_show}/{len(misclassified_scans)}', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_dir / 'misclassified_samples_scan_level.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Misclassified scans visualization saved: misclassified_samples_scan_level.png")
    
    # Save detailed list of misclassified SCANS
    misclassified_scans_export = misclassified_scans.copy()
    misclassified_scans_export['true_label'] = misclassified_scans_export['y_true'].map(lambda x: class_names[int(x)])
    misclassified_scans_export['pred_label'] = misclassified_scans_export['y_pred'].map(lambda x: class_names[int(x)])
    misclassified_scans_export['avg_confidence'] = misclassified_scans_export['y_prob']
    
    # Add voting breakdown for each scan
    voting_breakdown = []
    for _, row in misclassified_scans.iterrows():
        scan_id = row['scan_id']
        scan_slices = df_results[df_results['scan_id'] == scan_id]
        votes_ad = (scan_slices['y_pred'] == 0).sum()
        votes_nc = (scan_slices['y_pred'] == 1).sum()
        total = len(scan_slices)
        voting_breakdown.append(f"AD:{votes_ad}/{total}, NC:{votes_nc}/{total}")
    
    misclassified_scans_export['voting_breakdown'] = voting_breakdown
    misclassified_scans_export = misclassified_scans_export[['scan_id', 'true_label', 'pred_label', 
                                                               'avg_confidence', 'voting_breakdown']]
    
    misclassified_scans_export.to_csv(save_dir / 'misclassified_scans_list.csv', index=False)
    print(f"✓ Misclassified scans list saved to: misclassified_scans_list.csv")
    print(f"  Details: {len(misclassified_scans)} scans misclassified at scan-level")


def plot_model_complete_analysis(results, save_dir):
    """Create comprehensive model analysis visualization."""
    save_dir = Path(save_dir)
    
    slice_metrics = results['slice_level']
    scan_metrics = results['scan_level']
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Slice-level Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    slice_cm = np.array(slice_metrics['confusion_matrix'])
    sns.heatmap(slice_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['AD', 'NC'], yticklabels=['AD', 'NC'],
                cbar_kws={'label': 'Count'}, ax=ax1)
    ax1.set_title('Slice-Level Confusion Matrix', fontweight='bold', fontsize=12)
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # 2. Scan-level Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    scan_cm = np.array(scan_metrics['confusion_matrix'])
    sns.heatmap(scan_cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['AD', 'NC'], yticklabels=['AD', 'NC'],
                cbar_kws={'label': 'Count'}, ax=ax2)
    ax2.set_title('Scan-Level Confusion Matrix ⭐', fontweight='bold', fontsize=12)
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    # 3. ROC Curve
    ax3 = fig.add_subplot(gs[0, 2])
    fpr = np.array(results['roc_curve']['fpr'])
    tpr = np.array(results['roc_curve']['tpr'])
    ax3.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {slice_metrics["roc_auc"]:.4f}')
    ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curve', fontweight='bold', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Metrics Comparison (Slice vs Scan)
    ax4 = fig.add_subplot(gs[1, :])
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'Specificity']
    slice_values = [
        slice_metrics['accuracy'],
        slice_metrics['precision'],
        slice_metrics['recall'],
        slice_metrics['f1'],
        slice_metrics['specificity']
    ]
    scan_values = [
        scan_metrics['accuracy'],
        scan_metrics['precision'],
        scan_metrics['recall'],
        scan_metrics['f1'],
        scan_metrics['specificity']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, slice_values, width, label='Slice-Level', color='skyblue', alpha=0.8)
    bars2 = ax4.bar(x + width/2, scan_values, width, label='Scan-Level ⭐', color='green', alpha=0.8)
    
    ax4.set_ylabel('Score', fontsize=11)
    ax4.set_title('Performance Metrics: Slice vs Scan Level', fontweight='bold', fontsize=13)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)
    
    # 5. Performance Summary Table
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis('tight')
    ax5.axis('off')
    
    summary_data = [
        ['Metric', 'Slice-Level', 'Scan-Level'],
        ['Accuracy', f"{slice_metrics['accuracy']:.4f}", f"{scan_metrics['accuracy']:.4f}"],
        ['Precision', f"{slice_metrics['precision']:.4f}", f"{scan_metrics['precision']:.4f}"],
        ['Recall', f"{slice_metrics['recall']:.4f}", f"{scan_metrics['recall']:.4f}"],
        ['F1-Score', f"{slice_metrics['f1']:.4f}", f"{scan_metrics['f1']:.4f}"],
        ['Specificity', f"{slice_metrics['specificity']:.4f}", f"{scan_metrics['specificity']:.4f}"],
        ['AUC-ROC', f"{slice_metrics['roc_auc']:.4f}", f"{scan_metrics['roc_auc']:.4f}" if scan_metrics['roc_auc'] else 'N/A']
    ]
    
    table = ax5.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.32, 0.32])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax5.set_title('Performance Summary', fontweight='bold', fontsize=12, pad=20)
    
    # 6. Dataset Information
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('tight')
    ax6.axis('off')
    
    dataset_data = [
        ['Dataset Statistics', ''],
        ['Total Slices', f"{slice_metrics['num_slices']:,}"],
        ['Total Scans', f"{scan_metrics['num_scans']:,}"],
        ['Slices per Scan', f"{slice_metrics['num_slices'] / scan_metrics['num_scans']:.1f}"],
        ['', ''],
        ['Confusion Matrix (Scan)', ''],
        ['True Positives (TP)', f"{scan_cm[1,1]}"],
        ['True Negatives (TN)', f"{scan_cm[0,0]}"],
        ['False Positives (FP)', f"{scan_cm[0,1]}"],
        ['False Negatives (FN)', f"{scan_cm[1,0]}"]
    ]
    
    table2 = ax6.table(cellText=dataset_data, cellLoc='left', loc='center',
                      colWidths=[0.55, 0.45])
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 1.8)
    
    table2[(0, 0)].set_facecolor('#2196F3')
    table2[(0, 0)].set_text_props(weight='bold', color='white')
    table2[(5, 0)].set_facecolor('#2196F3')
    table2[(5, 0)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Dataset & Results Info', fontweight='bold', fontsize=12, pad=20)
    
    # 7. Key Insights
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    scan_acc = scan_metrics['accuracy']
    scan_f1 = scan_metrics['f1']
    
    insights_text = "🎯 KEY INSIGHTS\n\n"
    insights_text += f"✓ Scan-Level Accuracy: {scan_acc*100:.2f}%\n"
    insights_text += f"✓ Scan-Level F1-Score: {scan_f1:.4f}\n\n"
    
    if scan_acc >= 0.90:
        insights_text += "🌟 EXCELLENT Performance!\n"
    elif scan_acc >= 0.80:
        insights_text += "✅ GOOD Performance!\n"
    elif scan_acc >= 0.70:
        insights_text += "⚠️ MODERATE Performance\n"
    else:
        insights_text += "❌ Needs Improvement\n"
    
    insights_text += f"\n📊 Dataset Size:\n"
    insights_text += f"   • {scan_metrics['num_scans']} scans\n"
    insights_text += f"   • {slice_metrics['num_slices']} slices\n"
    
    improvement = (scan_acc - slice_metrics['accuracy']) * 100
    if improvement > 0:
        insights_text += f"\n📈 Scan-level aggregation\n"
        insights_text += f"   improved accuracy by\n"
        insights_text += f"   {improvement:.2f}% vs slice-level"
    
    ax7.text(0.1, 0.9, insights_text, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Overall title
    fig.suptitle('🧠 Complete Model Analysis: Alzheimer\'s Detection (AD vs NC)', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_dir / 'model_complete_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Complete model analysis saved")


@torch.no_grad()
def evaluate_model(model, test_loader, device):
    """Comprehensive evaluation with BOTH slice-level and scan-level metrics."""
    model.eval()

    save_dir = Path('./evaluation_results')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    all_labels = []
    all_preds = []
    all_probs = []
    all_scan_ids = []
    
    print("\n" + "="*60)
    print("Evaluating model (slice-level)...")
    print("="*60)
    
    for images, labels, scan_ids in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
        all_scan_ids.extend(scan_ids)
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # SLICE-LEVEL METRICS
    print("\n" + "="*60)
    print("SLICE-LEVEL RESULTS")
    print("="*60)
    
    slice_accuracy = accuracy_score(all_labels, all_preds)
    slice_precision, slice_recall, slice_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    slice_cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = slice_cm.ravel()
    slice_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    slice_roc_auc = auc(fpr, tpr)
    
    print(f"Slice Accuracy:    {slice_accuracy:.4f} ({slice_accuracy*100:.2f}%)")
    print(f"Slice Precision:   {slice_precision:.4f}")
    print(f"Slice Recall:      {slice_recall:.4f}")
    print(f"Slice F1:          {slice_f1:.4f}")
    print(f"Slice Specificity: {slice_specificity:.4f}")
    print(f"Slice AUC:         {slice_roc_auc:.4f}")
    
    # SCAN-LEVEL METRICS
    print("\n" + "="*60)
    print("SCAN-LEVEL RESULTS (Majority Voting)")
    print("="*60)
    
    df_results = pd.DataFrame({
        'scan_id': all_scan_ids,
        'y_true': all_labels,
        'y_pred': all_preds,
        'y_prob': all_probs
    })
    
    scan_metrics = calculate_scan_level_metrics(df_results)
    
    print(f"Scan Accuracy:    {scan_metrics['scan_accuracy']:.4f} ({scan_metrics['scan_accuracy']*100:.2f}%) ⭐")
    print(f"Scan Precision:   {scan_metrics['scan_precision']:.4f}")
    print(f"Scan Recall:      {scan_metrics['scan_recall']:.4f}")
    print(f"Scan F1:          {scan_metrics['scan_f1']:.4f}")
    print(f"Scan Specificity: {scan_metrics['scan_specificity']:.4f}")
    if scan_metrics['scan_auc']:
        print(f"Scan AUC:         {scan_metrics['scan_auc']:.4f}")
    print(f"Total Scans:      {scan_metrics['num_scans']}")
    
    print("\n" + "="*60)
    print(f"⭐ MAIN METRIC: Scan-Level Accuracy = {scan_metrics['scan_accuracy']*100:.2f}%")
    print("="*60)
    
    # SAVE RESULTS
    results = {
        'slice_level': {
            'accuracy': float(slice_accuracy),
            'precision': float(slice_precision),
            'recall': float(slice_recall),
            'f1': float(slice_f1),
            'specificity': float(slice_specificity),
            'roc_auc': float(slice_roc_auc),
            'confusion_matrix': slice_cm.tolist(),
            'num_slices': len(all_labels)
        },
        'scan_level': {
            'accuracy': float(scan_metrics['scan_accuracy']),
            'precision': float(scan_metrics['scan_precision']),
            'recall': float(scan_metrics['scan_recall']),
            'f1': float(scan_metrics['scan_f1']),
            'specificity': float(scan_metrics['scan_specificity']),
            'roc_auc': float(scan_metrics['scan_auc']) if scan_metrics['scan_auc'] else None,
            'confusion_matrix': scan_metrics['scan_confusion_matrix'].tolist(),
            'num_scans': scan_metrics['num_scans']
        },
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
    }
    
    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save scan predictions
    scan_metrics['scan_predictions'].to_csv(save_dir / 'scan_predictions.csv', index=False)
    print(f"\n✓ Scan predictions saved to: scan_predictions.csv")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Plot misclassified samples
    plot_misclassified_samples(test_loader, all_labels, all_preds, all_probs, 
                               all_scan_ids, save_dir, max_samples=20)
    
    # Plot complete analysis (metrics)
    plot_model_complete_analysis(results, save_dir)
    
    # Plot model frequency analysis
    visualize_model_complete(model, test_loader, device, save_dir)
    
    # Classification report
    class_names = ['AD', 'NC']
    scan_preds = scan_metrics['scan_predictions']
    report = classification_report(scan_preds['y_true'], scan_preds['y_pred'], 
                                   target_names=class_names)
    with open(save_dir / 'classification_report_scan_level.txt', 'w') as f:
        f.write("SCAN-LEVEL CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    
    print(f"\n✓ All results saved to: {save_dir}")
    
    return results, all_labels, all_preds, all_probs

def main():
    """Main evaluation and prediction pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pyramid GFNet - Evaluation with Scan-Level Metrics')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth')
    parser.add_argument('--data_dir', type=str, default='/home/groups/comp3710/ADNI/AD_NC')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_dir', type=str, default='./evaluation_results')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "="*60)
    print("🧠 Pyramid GFNet - Alzheimer's Detection")
    print("   WITH SCAN-LEVEL EVALUATION")
    print("="*60)
    print(f"Device: {device}")
    
    # Load model
    print("\n" + "="*60)
    print("Loading Model...")
    print("="*60)
    model, checkpoint = load_model(args.checkpoint, device)
    
    model_config = checkpoint.get('model_config', {})
    img_size = model_config.get('img_size', 224)
    
    # Load test data
    print("\n" + "="*60)
    print("Loading Test Data...")
    print("="*60)
    _, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=img_size,
        num_workers=4
    )
    
    # Evaluate
    results, labels, preds, probs = evaluate_model(model, test_loader, device)
    
    print("\n" + "="*60)
    print("✓ Complete! Check the following files:")
    print("="*60)
    print("  📊 model_complete_analysis.png - Performance metrics analysis")
    print("  🔬 model_frequency_analysis.png - Learned filters & features")
    print("  🔍 misclassified_samples_scan_level.png - Misclassified SCANS")
    print("  📈 confusion_matrix_*.png - Confusion matrices")
    print("  📉 roc_curve.png - ROC curve")
    print("  📋 test_results.json - Detailed metrics")
    print("  📝 scan_predictions.csv - Per-scan predictions")
    print("  📝 misclassified_scans_list.csv - Misclassified scans details")
    print("="*60)


if __name__ == "__main__":
    main()