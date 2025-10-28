# Example usage of trained model (Print out any results)

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc, classification_report
)
from tqdm import tqdm
import json

from modules import GFNet
from dataset import get_data_loaders


def load_model(checkpoint_path, device='cuda'):
    """
    Load trained model from checkpoint.
    
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
    
    # Initialize model with same architecture
    model = GFNet(
        img_size=224,
        patch_size=16,
        in_chans=1,
        num_classes=2,
        embed_dim=384,
        depth=12,
        mlp_ratio=4.0,
        drop_rate=0.1,
        drop_path_rate=0.2
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Trained for {checkpoint['epoch']+1} epochs")
    if 'val_acc' in checkpoint:
        print(f"Validation Accuracy: {checkpoint['val_acc']:.4f}")
    
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
    
    print("Evaluating model...")
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
    
    # ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Compile results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
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
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nPer-Class Metrics:")
    print(f"  AD - Precision: {precision_per_class[0]:.4f}, Recall: {recall_per_class[0]:.4f}, F1: {f1_per_class[0]:.4f}")
    print(f"  NC - Precision: {precision_per_class[1]:.4f}, Recall: {recall_per_class[1]:.4f}, F1: {f1_per_class[1]:.4f}")
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
        plot_confusion_matrix(cm, save_dir / 'confusion_matrix.png')
        
        # Plot ROC curve
        plot_roc_curve(fpr, tpr, roc_auc, save_dir / 'roc_curve.png')
        
        # Save classification report
        class_names = ['AD', 'NC']
        report = classification_report(all_labels, all_preds, target_names=class_names)
        with open(save_dir / 'classification_report.txt', 'w') as f:
            f.write(report)
        
        print(f"\nResults saved to: {save_dir}")
    
    return results, all_labels, all_preds, all_probs


def plot_confusion_matrix(cm, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['AD', 'NC'], yticklabels=['AD', 'NC'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            plt.text(j + 0.5, i + 0.7, f'({cm[i, j]/total*100:.1f}%)',
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    """Plot and save ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"ROC curve saved to {save_path}")


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
    
    print("Finding misclassified samples...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Analyzing"):
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
        print("No misclassifications found!")
        return
    
    print(f"Found {len(misclassified)} misclassified samples")
    
    # Visualize some misclassifications
    num_to_plot = min(num_samples, len(misclassified))
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(num_to_plot):
        sample = misclassified[i]
        img = sample['image'].permute(1, 2, 0).numpy()
        # Denormalize
        img = img * 0.5 + 0.5
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(
            f"True: {class_names[sample['true_label']]}\n"
            f"Pred: {class_names[sample['pred_label']]}\n"
            f"Conf: {sample['confidence'][sample['pred_label']]:.2f}",
            fontsize=9
        )
    
    # Hide unused subplots
    for i in range(num_to_plot, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Misclassified Samples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'misclassified_samples.png', dpi=300)
    plt.close()
    print(f"Misclassification analysis saved to {save_dir}")


def main():
    """Main evaluation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate GFNet on ADNI dataset')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='/home/groups/comp3710/ADNI/AD_NC',
                       help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--save_dir', type=str, default='./evaluation_results',
                       help='Directory to save results')
    parser.add_argument('--analyze_errors', action='store_true',
                       help='Analyze misclassified samples')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\n" + "="*60)
    print("Loading Model...")
    print("="*60)
    model, checkpoint = load_model(args.checkpoint, device)
    
    # Load test data
    print("\n" + "="*60)
    print("Loading Test Data...")
    print("="*60)
    _, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
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
    
    print("\n✓ Evaluation complete!")
    print(f"All results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()