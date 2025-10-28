import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from modules import PyramidGFNet, pyramid_gfnet_tiny, pyramid_gfnet_small, pyramid_gfnet_base
from dataset import get_data_loaders


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=7, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_metric):
        score = -val_metric if self.mode == 'min' else val_metric
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate comprehensive metrics for binary classification."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': None,
        'auc': None
    }
    
    # Calculate specificity (True Negative Rate)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['specificity'] = specificity
    
    # Calculate AUC-ROC if probabilities provided
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except:
            pass
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path, class_names=['AD', 'NC']):
    """Plot and save confusion matrix for AD vs NC."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix: Alzheimer\'s Detection (AD vs NC)')
    
    # Add percentage annotations
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j+0.5, i+0.7, f'({cm_percent[i, j]:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: Alzheimer\'s Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, use_amp=True):
    """Train for one epoch with mixed precision."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    scaler = GradScaler() if use_amp else None
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Track metrics
        running_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())  # Probability of class 1 (NC)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate epoch metrics
    avg_loss = running_loss / len(train_loader)
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = avg_loss
    
    return metrics


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(val_loader, desc='Validation', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    avg_loss = running_loss / len(val_loader)
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = avg_loss
    
    return metrics, all_labels, all_preds, all_probs


def plot_training_history(history, save_dir):
    """Plot training and validation metrics."""
    save_dir = Path(save_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot F1 score
    axes[1, 0].plot(history['train_f1'], label='Train F1', linewidth=2)
    axes[1, 0].plot(history['val_f1'], label='Val F1', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Training and Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot learning rate
    axes[1, 1].plot(history['learning_rates'], linewidth=2, color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=300)
    plt.close()


def train_model(
    # Model selection
    model_size='small',  # 'tiny', 'small', or 'base'
    
    # Model parameters (used if model_size is None)
    img_size=224,
    patch_size=4,
    in_chans=1,
    num_classes=2,  # AD vs NC (binary classification)
    embed_dims=[64, 128, 256, 512],
    depths=[2, 3, 8, 2],
    mlp_ratios=[4, 4, 4, 4],
    drop_rate=0.1,
    drop_path_rate=0.15,
    use_multiscale_fusion=True,
    
    # Training parameters
    batch_size=16,  # Reduced for pyramid architecture (uses more memory)
    num_epochs=100,
    learning_rate=1e-4,  # Slightly higher for pyramid network
    weight_decay=0.05,
    warmup_epochs=10,
    
    # Data parameters
    data_dir="/home/groups/comp3710/ADNI/AD_NC",
    num_workers=4,
    
    # Training options
    use_amp=True,
    early_stopping_patience=20,
    
    # Save options
    save_dir="./checkpoints",
    save_best_only=True
):
    """
    Main training function for Pyramid GFNet on Alzheimer's Detection.
    
    Binary Classification:
    - Class 0: AD (Alzheimer's Disease)
    - Class 1: NC (Normal Control)
    
    Model Sizes:
    - 'tiny': Fast, ~10M params, good for quick experiments
    - 'small': Balanced, ~22M params, recommended for most cases
    - 'base': Large, ~45M params, maximum accuracy
    """
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "="*60)
    print("Loading Alzheimer's MRI Data (AD vs NC)...")
    print("="*60)
    train_loader, test_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(test_loader.dataset)}")
    
    # Initialize model
    print("\n" + "="*60)
    print("Initializing Pyramid GFNet Model...")
    print("="*60)
    
    if model_size == 'tiny':
        print("Using: Pyramid GFNet TINY")
        model = pyramid_gfnet_tiny(img_size=img_size, num_classes=num_classes)
    elif model_size == 'small':
        print("Using: Pyramid GFNet SMALL (Recommended)")
        model = pyramid_gfnet_small(img_size=img_size, num_classes=num_classes)
    elif model_size == 'base':
        print("Using: Pyramid GFNet BASE")
        model = pyramid_gfnet_base(img_size=img_size, num_classes=num_classes)
    else:
        print("Using: Custom Pyramid GFNet")
        model = PyramidGFNet(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dims=embed_dims,
            depths=depths,
            mlp_ratios=mlp_ratios,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_multiscale_fusion=use_multiscale_fusion
        )
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    print(f"Loss function: CrossEntropyLoss with label smoothing (0.1)")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler with warmup and cosine decay
    total_steps = num_epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"Scheduler: Warmup ({warmup_epochs} epochs) + Cosine Annealing")
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_patience, 
        min_delta=0.001, 
        mode='max'
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_auc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': [],
        'val_specificity': [], 'val_precision': [], 'val_recall': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    best_val_auc = 0.0
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training for Alzheimer's Detection...")
    print("="*60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, use_amp
        )
        
        # Validate
        val_metrics, val_labels, val_preds, val_probs = validate(
            model, test_loader, criterion, device
        )
        
        # Calculate train-val gap
        train_val_gap = train_metrics['accuracy'] - val_metrics['accuracy']
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['train_auc'].append(train_metrics['auc'] or 0)
        
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'] or 0)
        history['val_specificity'].append(val_metrics['specificity'] or 0)
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print metrics
        print(f"Train → Loss: {train_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f} | "
              f"F1: {train_metrics['f1']:.4f} | "
              f"AUC: {train_metrics['auc']:.4f}" if train_metrics['auc'] else "")
        
        print(f"Val   → Loss: {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"F1: {val_metrics['f1']:.4f}")
        
        if val_metrics['auc']:
            print(f"        AUC: {val_metrics['auc']:.4f} | "
                  f"Precision: {val_metrics['precision']:.4f} | "
                  f"Recall: {val_metrics['recall']:.4f} | "
                  f"Specificity: {val_metrics['specificity']:.4f}")
        
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Overfitting check
        if train_val_gap > 0.15:
            print(f"⚠️  WARNING: Train-Val gap = {train_val_gap:.4f} (severe overfitting!)")
        elif train_val_gap > 0.10:
            print(f"⚠️  CAUTION: Train-Val gap = {train_val_gap:.4f} (mild overfitting)")
        else:
            print(f"✓ Train-Val gap = {train_val_gap:.4f} (healthy)")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_val_auc = val_metrics['auc'] or 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_auc': best_val_auc,
                'val_metrics': val_metrics,
                'model_config': {
                    'model_size': model_size,
                    'img_size': img_size,
                    'num_classes': num_classes,
                    'embed_dims': embed_dims,
                    'depths': depths
                }
            }, save_dir / 'best_model.pth')
            print(f"✓ Best model saved (Val Acc: {best_val_acc:.4f}, AUC: {best_val_auc:.4f})")
        
        # Save latest checkpoint
        if not save_best_only:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, save_dir / 'latest_checkpoint.pth')
        
        # Early stopping
        early_stopping(val_metrics['accuracy'])
        if early_stopping.early_stop:
            print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
            break
    
    # Final evaluation and visualization
    print("\n" + "="*60)
    print("Training Completed! Generating Final Reports...")
    print("="*60)
    
    # Load best model for final evaluation
    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation
    final_metrics, final_labels, final_preds, final_probs = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\n📊 Final Results on Best Model:")
    print(f"  Accuracy:    {final_metrics['accuracy']:.4f}")
    print(f"  F1 Score:    {final_metrics['f1']:.4f}")
    print(f"  Precision:   {final_metrics['precision']:.4f}")
    print(f"  Recall:      {final_metrics['recall']:.4f}")
    print(f"  Specificity: {final_metrics['specificity']:.4f}")
    if final_metrics['auc']:
        print(f"  AUC-ROC:     {final_metrics['auc']:.4f}")
    
    # Plot training history
    plot_training_history(history, save_dir)
    
    # Plot confusion matrix
    plot_confusion_matrix(final_labels, final_preds, 
                         save_dir / 'confusion_matrix_final.png',
                         class_names=['AD', 'NC'])
    
    # Plot ROC curve
    if final_metrics['auc']:
        plot_roc_curve(final_labels, final_probs, 
                      save_dir / 'roc_curve_final.png')
    
    # Save final metrics
    final_report = {
        'best_val_accuracy': float(best_val_acc),
        'best_val_auc': float(best_val_auc),
        'final_metrics': {k: float(v) if v is not None else None 
                         for k, v in final_metrics.items()},
        'model_config': {
            'model_size': model_size,
            'total_params': total_params,
            'img_size': img_size,
            'num_classes': num_classes
        }
    }
    
    with open(save_dir / 'final_report.json', 'w') as f:
        json.dump(final_report, f, indent=4)
    
    # Save training history
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"\n✓ All results saved to: {save_dir}")
    print("="*60)
    
    return model, history, final_metrics


if __name__ == "__main__":
    # Train with recommended settings for Alzheimer's detection
    print("🧠 Pyramid GFNet for Alzheimer's Detection (AD vs NC)")
    print("="*60)
    
    model, history, metrics = train_model(
        model_size='small',  # Change to 'tiny' or 'base' as needed
        img_size=224,
        batch_size=16,
        num_epochs=100,
        learning_rate=1e-4,
        early_stopping_patience=20,
        use_amp=True
    )
    
    print("\n" + "="*60)
    print("🎉 Training Complete!")
    print("="*60)
    print(f"Best Validation Accuracy: {max(history['val_acc']):.4f}")
    print(f"Best Validation AUC: {max(history['val_auc']):.4f}")
    print(f"Final F1 Score: {metrics['f1']:.4f}")