# Example usage of trained model (Print out any results)

import torch
import torch.nn.functional as F
import numpy as np
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


def get_test_transform(img_size=224):
    """Get test-time transforms matching training."""
    return transforms.Compose([
        AutoCropBlack(threshold=10),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2670], std=[0.2657])
    ])


def load_model(checkpoint_path, device='cuda'):
    """
    Load trained model from checkpoint.
    Supports both old GFNet and new PyramidGFNet architectures.
    
    Args:
        checkpoint_path: Path to saved checkpoint
        device: Device to load model on
    
    Returns:
        model: Loaded model in eval mode
        checkpoint: Full checkpoint dict with metadata
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Detect architecture type by checking state_dict keys
    is_pyramid = 'stages.0.blocks.0.gamma_1' in state_dict or 'merges.0.reduction.weight' in state_dict
    is_old_gfnet = 'blocks.0.gamma_1' in state_dict and not is_pyramid
    
    # Extract model configuration
    model_config = checkpoint.get('model_config', {})
    img_size = model_config.get('img_size', 224)
    num_classes = model_config.get('num_classes', 2)
    
    if is_pyramid:
        # New PyramidGFNet architecture
        model_size = model_config.get('model_size', 'small')
        print(f"Loading model: Pyramid GFNet {model_size.upper()}")
        print(f"Image size: {img_size}, Classes: {num_classes}")
        
        # Initialize model based on size
        if model_size == 'tiny':
            model = pyramid_gfnet_tiny(img_size=img_size, num_classes=num_classes)
        elif model_size == 'small':
            model = pyramid_gfnet_small(img_size=img_size, num_classes=num_classes)
        elif model_size == 'base':
            model = pyramid_gfnet_base(img_size=img_size, num_classes=num_classes)
        else:
            # Custom model
            model = PyramidGFNet(
                img_size=img_size,
                num_classes=num_classes,
                embed_dims=model_config.get('embed_dims', [64, 128, 256, 512]),
                depths=model_config.get('depths', [3, 4, 6, 3])
            )
    
    elif is_old_gfnet:
        # Old GFNet architecture (non-pyramid)
        print(f"Loading model: OLD GFNet (non-pyramid)")
        print(f"Image size: {img_size}, Classes: {num_classes}")
        print("WARNING: This is an old model architecture. Consider retraining with PyramidGFNet.")
        
        if GFNet is None:
            raise ImportError(
                "Old GFNet model detected but GFNet class not found in modules.py. "
                "Please add the old GFNet class or retrain with PyramidGFNet."
            )
        
        # Detect parameters from state_dict
        embed_dim = state_dict['norm.weight'].shape[0]
        depth = sum(1 for k in state_dict.keys() if k.startswith('blocks.') and 'gamma_1' in k)
        
        print(f"  Detected: embed_dim={embed_dim}, depth={depth}")
        
        model = GFNet(
            img_size=img_size,
            patch_size=16,
            in_chans=1,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            mlp_ratio=3.0,
            drop_rate=0.3,
            drop_path_rate=0.4
        )
    
    else:
        raise ValueError(
            f"Unable to detect model architecture from checkpoint. "
            f"State dict keys: {list(state_dict.keys())[:5]}..."
        )
    
    # Load weights
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Trained for {checkpoint['epoch']+1} epochs")
    if 'val_acc' in checkpoint:
        print(f"  Validation Accuracy: {checkpoint['val_acc']:.4f}")
    if 'val_auc' in checkpoint:
        print(f"  Validation AUC: {checkpoint['val_auc']:.4f}")
    
    return model, checkpoint


@torch.no_grad()
def evaluate_model(model, test_loader, device):
    """
    Comprehensive evaluation of the model.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
    
    Returns:
        results: Dictionary containing all evaluation metrics
    """
    model.eval()

    save_dir = Path('./evaluation_results')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    print("\n" + "="*60)
    print("Evaluating model...")
    print("="*60)
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = \
        precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Specificity
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Compile results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'per_class': {
            'AD': {
                'precision': precision_per_class[0],
                'recall': recall_per_class[0],
                'f1': f1_per_class[0],
                'support': int(support[0])
            },
            'NC': {
                'precision': precision_per_class[1],
                'recall': recall_per_class[1],
                'f1': f1_per_class[1],
                'support': int(support[1])
            }
        },
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
    }
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy:  {accuracy:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1 Score:          {f1:.4f}")
    print(f"Specificity:       {specificity:.4f}")
    print(f"ROC AUC:           {roc_auc:.4f}")
    print("\nPer-Class Metrics:")
    print(f"  AD - Precision: {precision_per_class[0]:.4f}, Recall: {recall_per_class[0]:.4f}, F1: {f1_per_class[0]:.4f}, Support: {support[0]}")
    print(f"  NC - Precision: {precision_per_class[1]:.4f}, Recall: {recall_per_class[1]:.4f}, F1: {f1_per_class[1]:.4f}, Support: {support[1]}")
    print("\nConfusion Matrix:")
    print(f"  True AD, Pred AD: {cm[0, 0]}")
    print(f"  True AD, Pred NC: {cm[0, 1]}")
    print(f"  True NC, Pred AD: {cm[1, 0]}")
    print(f"  True NC, Pred NC: {cm[1, 1]}")
    print("="*60)
    
    # Save metrics as JSON
    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    # Plot confusion matrix
    plot_confusion_matrix(cm, save_dir / 'confusion_matrix_test.png')
        
    # Plot ROC curve
    plot_roc_curve(fpr, tpr, roc_auc, save_dir / 'roc_curve_test.png')
        
    # Save classification report
    class_names = ['AD', 'NC']
    report = classification_report(all_labels, all_preds, target_names=class_names)
    with open(save_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
    print(f"  Classification report saved")
        
    print(f"\n✓ Results saved to: {save_dir}")
    
    return results, all_labels, all_preds, all_probs


def plot_confusion_matrix(cm, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Greens',
                xticklabels=['AD', 'NC'], yticklabels=['AD', 'NC'],
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix: Alzheimer\'s Detection (AD vs NC)')
    
    # Add percentages and label
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    labels = np.array([['TP', 'FN'],
                   ['FP', 'TN']])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.5, f"{labels[i, j]}\n{cm_percent[i, j]:.1f}%",
                 ha='center', va='center', fontsize=12, color='white' if cm[i, j] > cm.max() / 2 else 'black')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Confusion matrix saved to {save_path.name}")


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
    print(f"  ROC curve saved to {save_path.name}")


def visualize_model_complete(model, test_loader, device, save_dir):
    """
    Create ONE comprehensive visualization combining:
    - Frequency filters (learned weights)
    - Feature maps (averaged across test set)
    - Frequency attention (averaged across test set)
    
    No image path needed - uses the entire test set!
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        save_dir: Directory to save visualization
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
        for batch_idx, (images, labels) in enumerate(test_loader):
            if sample_count >= max_samples:
                break
            
            images = images.to(device)
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
    
    plt.suptitle(f'Complete Model Analysis (Averaged over {sample_count} test samples)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    save_path = save_dir / 'model_complete_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Complete model visualization saved: {save_path}")
    return str(save_path)


def analyze_misclassifications(model, test_loader, device, save_dir, num_samples=10):
    """
    Analyze and visualize misclassified samples.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        save_dir: Directory to save visualizations
        num_samples: Number of misclassified samples to visualize
    """
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    misclassified = []
    class_names = ['AD', 'NC']
    
    print("\nFinding misclassified samples...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Analyzing", leave=False):
            images_device = images.to(device)
            outputs = model(images_device)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Find misclassified samples
            mask = (preds != labels.to(device))
            if mask.any():
                for idx in mask.nonzero(as_tuple=True)[0]:
                    misclassified.append({
                        'image': images[idx].cpu(),
                        'true_label': labels[idx].item(),
                        'pred_label': preds[idx].cpu().item(),
                        'confidence': probs[idx].cpu().numpy()
                    })
    
    if len(misclassified) == 0:
        print("✓ No misclassifications found! Perfect accuracy!")
        return
    
    print(f"Found {len(misclassified)} misclassified samples ({len(misclassified)/len(test_loader.dataset)*100:.1f}%)")
    
    # Visualize some misclassifications
    num_to_plot = min(num_samples, len(misclassified))
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(num_to_plot):
        sample = misclassified[i]
        img = sample['image'].squeeze().numpy()
        
        # Denormalize
        img = img * 0.2657 + 0.2670
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(
            f"True: {class_names[sample['true_label']]}\n"
            f"Pred: {class_names[sample['pred_label']]}\n"
            f"Conf: {sample['confidence'][sample['pred_label']]:.2%}",
            fontsize=9
        )
    
    # Hide unused subplots
    for i in range(num_to_plot, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Misclassified Samples (AD vs NC)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'misclassified_samples.png', dpi=300)
    plt.close()
    print(f"✓ Misclassification analysis saved to {save_dir}")


def main():
    """Main evaluation and prediction pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pyramid GFNet Alzheimer\'s Detection - Evaluation')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='/home/groups/comp3710/ADNI/AD_NC',
                       help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--save_dir', type=str, default='./evaluation_results',
                       help='Directory to save results')
    
    # Visualization options
    parser.add_argument('--visualize_model', action='store_true',
                       help='Generate complete model visualization (filters + features + attention)')
    parser.add_argument('--analyze_errors', action='store_true',
                       help='Analyze misclassified samples')
    parser.add_argument('--visualize_all', action='store_true',
                       help='Generate ALL visualizations')
    
    args = parser.parse_args()

    # If --visualize_all, enable everything
    if args.visualize_all:
        args.visualize_model = True
        args.analyze_errors = True
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "="*60)
    print("🧠 Pyramid GFNet - Alzheimer's Detection")
    print("="*60)
    print(f"Device: {device}")
    
    # Load model
    print("\n" + "="*60)
    print("Loading Model...")
    print("="*60)
    model, checkpoint = load_model(args.checkpoint, device)
    
    # Get model config for transform
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
    results, labels, preds, probs = evaluate_model(
        model, test_loader, device
    )
    
    # Complete model visualization (auto-combined - no image path needed!)
    if args.visualize_model:
        print("\n" + "="*60)
        print("Creating Complete Model Visualization...")
        print("="*60)
        visualize_model_complete(model, test_loader, device, Path(args.save_dir))
    
    # Analyze misclassifications
    if args.analyze_errors:
        print("\n" + "="*60)
        print("Analyzing Misclassifications...")
        print("="*60)
        analyze_misclassifications(
            model, test_loader, device, 
            save_dir=Path(args.save_dir)
        )
    
    print("\n" + "="*60)
    print("✓ Complete!")
    print("="*60)
    print(f"\nGenerated files in {args.save_dir}:")
    print("  - test_results.json")
    print("  - confusion_matrix_test.png")
    print("  - roc_curve_test.png")
    print("  - classification_report.txt")
    if args.visualize_model:
        print("  - model_complete_analysis.png")
    if args.analyze_errors:
        print("  - misclassified_samples.png")


if __name__ == "__main__":
    main()