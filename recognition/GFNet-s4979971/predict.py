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
    """
    Calculate metrics at scan level using majority voting.
    
    Args:
        df_results: DataFrame with columns ['scan_id', 'y_true', 'y_pred', 'y_prob']
    
    Returns:
        Dictionary of scan-level metrics
    """
    # Group by scan_id and aggregate
    scan_results = df_results.groupby('scan_id').agg({
        'y_true': 'first',  # Same for all slices of a scan
        'y_pred': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],  # Majority vote
        'y_prob': 'mean'  # Average probability
    }).reset_index()
    
    y_true_scan = scan_results['y_true'].values
    y_pred_scan = scan_results['y_pred'].values
    y_prob_scan = scan_results['y_prob'].values
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_scan, y_pred_scan)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_scan, y_pred_scan, average='binary', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true_scan, y_pred_scan)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # ROC AUC
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
    
    # Detect architecture type
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


@torch.no_grad()
def evaluate_model(model, test_loader, device):
    """
    Comprehensive evaluation with BOTH slice-level and scan-level metrics.
    """
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
        all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (NC)
        all_scan_ids.extend(scan_ids)
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # ==================== SLICE-LEVEL METRICS ====================
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
    
    # ==================== SCAN-LEVEL METRICS ====================
    print("\n" + "="*60)
    print("SCAN-LEVEL RESULTS (Majority Voting)")
    print("="*60)
    
    # Create DataFrame for scan-level aggregation
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
    
    # ==================== SAVE RESULTS ====================
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
    
    # Save scan-level predictions
    scan_metrics['scan_predictions'].to_csv(save_dir / 'scan_predictions.csv', index=False)
    print(f"\n✓ Scan predictions saved to: scan_predictions.csv")
    
    # Plot confusion matrices (both levels)
    plot_confusion_matrix(slice_cm, save_dir / 'confusion_matrix_slice_level.png', 
                         title='Slice-Level Confusion Matrix')
    plot_confusion_matrix(scan_metrics['scan_confusion_matrix'], 
                         save_dir / 'confusion_matrix_scan_level.png',
                         title='Scan-Level Confusion Matrix (Majority Voting)')
    
    # Plot ROC curve
    plot_roc_curve(fpr, tpr, slice_roc_auc, save_dir / 'roc_curve.png')
    
    # Save classification report
    class_names = ['AD', 'NC']
    scan_preds = scan_metrics['scan_predictions']
    report = classification_report(scan_preds['y_true'], scan_preds['y_pred'], 
                                   target_names=class_names)
    with open(save_dir / 'classification_report_scan_level.txt', 'w') as f:
        f.write("SCAN-LEVEL CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    
    print(f"\n✓ Results saved to: {save_dir}")
    
    return results, all_labels, all_preds, all_probs


def plot_confusion_matrix(cm, save_path, title='Confusion Matrix'):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Greens',
                xticklabels=['AD', 'NC'], yticklabels=['AD', 'NC'],
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    labels = np.array([['TP', 'FN'], ['FP', 'TN']])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.5, f"{labels[i, j]}\n{cm[i, j]}\n({cm_percent[i, j]:.1f}%)",
                 ha='center', va='center', fontsize=11, 
                 color='white' if cm[i, j] > cm.max() / 2 else 'black')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  ✓ {save_path.name}")


def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    """Plot and save ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: Alzheimer\'s Detection')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  ✓ {save_path.name}")


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
    print("✓ Complete!")
    print("="*60)


if __name__ == "__main__":
    main()