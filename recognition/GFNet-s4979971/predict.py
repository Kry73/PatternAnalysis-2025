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
def predict_batch(model, images, device):
    """
    Predict on a batch of images.
    
    Args:
        model: Trained model
        images: Batch of images [B, C, H, W]
        device: Device to run prediction on
    
    Returns:
        predictions: Class predictions [B]
        probabilities: Class probabilities [B, num_classes]
    """
    model.eval()
    images = images.to(device)
    
    outputs = model(images)
    probabilities = F.softmax(outputs, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    
    return predictions.cpu().numpy(), probabilities.cpu().numpy()


@torch.no_grad()
def evaluate_model(model, test_loader, device, save_dir=None):
    """
    Comprehensive evaluation of the model.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        save_dir: Directory to save results (optional)
    
    Returns:
        results: Dictionary containing all evaluation metrics
    """
    model.eval()
    
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
    
    # Save results
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['AD', 'NC'], yticklabels=['AD', 'NC'],
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix: Alzheimer\'s Detection (AD vs NC)')
    
    # Add percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j+0.5, i+0.7, f'({cm_percent[i, j]:.1f}%)',
                    ha='center', va='center', fontsize=10, color='gray')
    
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


@torch.no_grad()
def predict_single_image(model, image_path, transform, device, visualize=True, save_path=None):
    """
    Predict on a single image and optionally visualize.
    
    Args:
        model: Trained model
        image_path: Path to image file
        transform: Transform pipeline
        device: Device to run on
        visualize: Whether to create visualization
        save_path: Path to save visualization
    
    Returns:
        Dictionary with prediction results
    """
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('L')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    outputs = model(image_tensor)
    probs = F.softmax(outputs, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred_class].item()
    
    class_names = ['AD (Alzheimer\'s Disease)', 'NC (Normal Control)']
    results = {
        'predicted_class': pred_class,
        'predicted_label': class_names[pred_class],
        'confidence': confidence,
        'probabilities': {
            'AD': probs[0, 0].item(),
            'NC': probs[0, 1].item()
        },
        'image_path': str(image_path)
    }
    
    # Print results
    print(f"\nImage: {Path(image_path).name}")
    print(f"Prediction: {results['predicted_label']}")
    print(f"Confidence: {results['confidence']:.2%}")
    print(f"Probabilities:")
    print(f"  AD: {results['probabilities']['AD']:.2%}")
    print(f"  NC: {results['probabilities']['NC']:.2%}")
    
    # Visualize
    if visualize:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show image
        axes[0].imshow(image, cmap='gray')
        axes[0].axis('off')
        axes[0].set_title(f"Input MRI Scan\n{Path(image_path).name}", fontsize=12)
        
        # Show predictions
        classes = ['AD', 'NC']
        probs_list = [results['probabilities']['AD'], results['probabilities']['NC']]
        colors = ['#e74c3c', '#2ecc71']
        
        bars = axes[1].barh(classes, probs_list, color=colors, alpha=0.7)
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel('Probability', fontsize=11)
        axes[1].set_title('Prediction Probabilities', fontsize=12)
        axes[1].grid(axis='x', alpha=0.3)
        
        # Add probability labels
        for bar, prob in zip(bars, probs_list):
            axes[1].text(prob + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{prob:.2%}', va='center', fontsize=10)
        
        # Add prediction text
        pred_text = (f"Prediction: {results['predicted_label']}\n"
                    f"Confidence: {results['confidence']:.2%}")
        fig.text(0.5, 0.02, pred_text, ha='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    return results


@torch.no_grad()
def predict_directory(model, image_dir, transform, device, save_dir):
    """
    Predict on all images in a directory.
    
    Args:
        model: Trained model
        image_dir: Directory containing images
        transform: Transform pipeline
        device: Device to run on
        save_dir: Directory to save results
    """
    model.eval()
    image_dir = Path(image_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.nii*']:
        image_paths.extend(image_dir.glob(ext))
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return
    
    print(f"\nFound {len(image_paths)} images")
    
    # Predict all images
    all_results = []
    for img_path in tqdm(image_paths, desc="Predicting"):
        try:
            image = Image.open(img_path).convert('L')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            
            results = {
                'image_path': str(img_path.name),
                'predicted_class': pred_class,
                'predicted_label': 'AD' if pred_class == 0 else 'NC',
                'confidence': probs[0, pred_class].item(),
                'prob_AD': probs[0, 0].item(),
                'prob_NC': probs[0, 1].item()
            }
            all_results.append(results)
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
    
    # Save results
    with open(save_dir / 'predictions.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # Generate summary
    total = len(all_results)
    ad_count = sum(1 for r in all_results if r['predicted_class'] == 0)
    nc_count = sum(1 for r in all_results if r['predicted_class'] == 1)
    avg_conf = np.mean([r['confidence'] for r in all_results])
    
    summary = {
        'total_images': total,
        'predictions': {
            'AD': ad_count,
            'NC': nc_count
        },
        'percentages': {
            'AD': f"{ad_count/total*100:.1f}%",
            'NC': f"{nc_count/total*100:.1f}%"
        },
        'average_confidence': f"{avg_conf:.2%}"
    }
    
    with open(save_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Print summary
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"Total images:      {total}")
    print(f"AD predictions:    {ad_count} ({ad_count/total*100:.1f}%)")
    print(f"NC predictions:    {nc_count} ({nc_count/total*100:.1f}%)")
    print(f"Avg confidence:    {avg_conf:.2%}")
    print("="*60)
    
    # Create summary plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Prediction distribution
    axes[0].pie([ad_count, nc_count],
               labels=['AD', 'NC'],
               colors=['#e74c3c', '#2ecc71'],
               autopct='%1.1f%%',
               startangle=90)
    axes[0].set_title('Prediction Distribution', fontsize=12)
    
    # Confidence distribution
    confidences = [r['confidence'] for r in all_results]
    axes[1].hist(confidences, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Confidence', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('Confidence Distribution', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'summary_plot.png', dpi=300)
    plt.close()
    
    print(f"\n✓ All results saved to: {save_dir}")


def main():
    """Main evaluation and prediction pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pyramid GFNet Alzheimer\'s Detection - Prediction & Evaluation')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='evaluate', choices=['evaluate', 'predict_single', 'predict_dir'],
                       help='Operation mode')
    
    # For evaluation mode
    parser.add_argument('--data_dir', type=str, default='/home/groups/comp3710/ADNI/AD_NC',
                       help='Path to dataset (for evaluation)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--analyze_errors', action='store_true',
                       help='Analyze misclassified samples')
    
    # For prediction modes
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image (for predict_single)')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Path to image directory (for predict_dir)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    
    # General
    parser.add_argument('--save_dir', type=str, default='./evaluation_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "="*60)
    print("🧠 Pyramid GFNet - Alzheimer's Detection")
    print("="*60)
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    
    # Load model
    print("\n" + "="*60)
    print("Loading Model...")
    print("="*60)
    model, checkpoint = load_model(args.checkpoint, device)
    
    # Get model config for transform
    model_config = checkpoint.get('model_config', {})
    img_size = model_config.get('img_size', 224)
    transform = get_test_transform(img_size)
    
    # Execute based on mode
    if args.mode == 'evaluate':
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
            model, test_loader, device, save_dir=args.save_dir
        )
        
        # Analyze misclassifications
        if args.analyze_errors:
            print("\n" + "="*60)
            print("Analyzing Misclassifications...")
            print("="*60)
            analyze_misclassifications(
                model, test_loader, device, 
                save_dir=Path(args.save_dir) / 'misclassifications'
            )
    
    elif args.mode == 'predict_single':
        if not args.image:
            print("Error: --image required for predict_single mode")
            return
        
        print("\n" + "="*60)
        print("Single Image Prediction")
        print("="*60)
        
        save_path = Path(args.save_dir) / 'prediction_visualization.png' if args.visualize else None
        results = predict_single_image(
            model, args.image, transform, device,
            visualize=args.visualize, save_path=save_path
        )
    
    elif args.mode == 'predict_dir':
        if not args.image_dir:
            print("Error: --image_dir required for predict_dir mode")
            return
        
        print("\n" + "="*60)
        print("Directory Prediction")
        print("="*60)
        
        predict_directory(
            model, args.image_dir, transform, device, args.save_dir
        )
    
    print("\n" + "="*60)
    print("✓ Complete!")
    print("="*60)


if __name__ == "__main__":
    main()