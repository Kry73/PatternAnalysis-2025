import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
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
import random


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


def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=0.3, device='cuda'):
    """Apply mixup augmentation to a batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixup."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def calculate_scan_level_metrics(df_results):
    """
    Calculate metrics at scan level using majority voting.
    
    Args:
        df_results: DataFrame with columns ['scan_id', 'y_true', 'y_pred', 'y_prob']
    
    Returns:
        Dictionary of scan-level metrics
    """
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
        auc_score = roc_auc_score(y_true_scan, y_prob_scan)
    except:
        auc_score = None
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'auc': auc_score,
        'num_scans': len(scan_results),
        'confusion_matrix': cm
    }


def calculate_metrics(y_true, y_pred, y_prob=None, scan_ids=None):
    """
    Calculate comprehensive metrics for binary classification.
    Now supports BOTH slice-level and scan-level metrics!
    """
    # Slice-level metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    metrics = {
        'slice_accuracy': accuracy,
        'slice_precision': precision,
        'slice_recall': recall,
        'slice_f1': f1,
        'slice_specificity': None,
        'slice_auc': None
    }
    
    # Calculate specificity
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['slice_specificity'] = specificity
    
    # Calculate AUC-ROC
    if y_prob is not None:
        try:
            metrics['slice_auc'] = roc_auc_score(y_true, y_prob)
        except:
            pass
    
    # Scan-level metrics (if scan_ids provided)
    if scan_ids is not None:
        df = pd.DataFrame({
            'scan_id': scan_ids,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob if y_prob is not None else y_pred
        })
        scan_metrics = calculate_scan_level_metrics(df)
        
        # Add scan metrics with prefix
        metrics.update({
            'scan_accuracy': scan_metrics['accuracy'],
            'scan_precision': scan_metrics['precision'],
            'scan_recall': scan_metrics['recall'],
            'scan_f1': scan_metrics['f1'],
            'scan_specificity': scan_metrics['specificity'],
            'scan_auc': scan_metrics['auc'],
            'num_scans': scan_metrics['num_scans'],
            'scan_confusion_matrix': scan_metrics['confusion_matrix']
        })
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path, class_names=['AD', 'NC'], title=None):
    """Plot and save confusion matrix for AD vs NC."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title or 'Confusion Matrix: Alzheimer\'s Detection (AD vs NC)')
    
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
    print(f"✓ Confusion matrix saved to: {save_path}")


def plot_roc_curve(y_true, y_prob, save_path, title=None):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title or 'ROC Curve: Alzheimer\'s Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ ROC Graph saved to: {save_path}")


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, use_amp=True, use_mixup=False, mixup_alpha=0.3):
    """Train for one epoch with mixed precision."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    scaler = GradScaler() if use_amp else None
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for images, labels, scan_ids in pbar: 
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if use_mixup and np.random.rand() > 0.5:
            images, targets_a, targets_b, lam = mixup_data(
                images, labels, alpha=mixup_alpha, device=device
            )
            
            if use_amp:
                with autocast():
                    outputs = model(images)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                loss.backward()
                optimizer.step()
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            if lam >= 0.5:
                approx_labels = targets_a
            else:
                approx_labels = targets_b
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(approx_labels.cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
            
        else:
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
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
        
        if scheduler is not None:
            scheduler.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / len(train_loader)
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = avg_loss
    
    return metrics

@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """
    Validate the model with SCAN-LEVEL metrics.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_scan_ids = []
    
    pbar = tqdm(val_loader, desc='Validation', leave=False)
    for images, labels, scan_ids in pbar: 
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
        all_scan_ids.extend(scan_ids)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / len(val_loader)
    
    # Calculate BOTH slice-level and scan-level metrics
    metrics = calculate_metrics(all_labels, all_preds, all_probs, scan_ids=all_scan_ids)
    metrics['loss'] = avg_loss
    
    return metrics, all_labels, all_preds, all_probs, all_scan_ids


def plot_training_history(history, save_dir):
    """Plot training and validation metrics with SCAN-LEVEL emphasis."""
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
    
    # Plot accuracy (BOTH slice and scan level)
    axes[0, 1].plot(history['train_slice_acc'], label='Train Slice Acc', linewidth=2, linestyle='--', alpha=0.7)
    axes[0, 1].plot(history['val_slice_acc'], label='Val Slice Acc', linewidth=2, linestyle='--', alpha=0.7)
    axes[0, 1].plot(history['val_scan_acc'], label='Val Scan Acc ⭐', linewidth=3, color='green')
    axes[0, 1].axhline(y=0.80, color='red', linestyle=':', linewidth=2, label='Target (80%)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy: Slice vs Scan Level')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot F1 score
    axes[1, 0].plot(history['train_slice_f1'], label='Train F1', linewidth=2)
    axes[1, 0].plot(history['val_slice_f1'], label='Val Slice F1', linewidth=2)
    if 'val_scan_f1' in history:
        axes[1, 0].plot(history['val_scan_f1'], label='Val Scan F1 ⭐', linewidth=3, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score Comparison')
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
    print(f"✓ Training history plot saved")


def train_model(
    # Model selection
    model_size='small',
    
    # Model parameters
    img_size=224,
    patch_size=4,
    in_chans=1,
    num_classes=2,
    embed_dims=[64, 128, 256, 512],
    depths=[2, 3, 8, 2],
    mlp_ratios=[4, 4, 4, 4],
    drop_rate=0.1,
    drop_path_rate=0.15,
    use_multiscale_fusion=True,
    
    # Training parameters
    batch_size=16,
    num_epochs=100,
    learning_rate=1e-4,
    weight_decay=0.05,
    warmup_epochs=10,
    min_lr=1e-7,
    
    # Data parameters
    data_dir="/home/groups/comp3710/ADNI/AD_NC",
    num_workers=4,
    
    # Training options
    use_amp=True,
    early_stopping_patience=20,
    min_delta=0.001,
    use_mixup=False,
    mixup_alpha=0.3,
    
    # Save options
    save_dir="./checkpoints_norm"
):
    """
    Main training function with SCAN-LEVEL evaluation.
    The model is now optimized for scan-level accuracy!
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "="*60)
    print("Loading Alzheimer's MRI Data (AD vs NC)")
    print("WITH SCAN-LEVEL TRACKING")
    print("="*60)
    train_loader, test_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(test_loader.dataset)}")
    
    # CRITICAL: Verify scan IDs are being returned
    print("\n🔍 Verifying scan ID tracking...")
    try:
        test_batch = next(iter(test_loader))
        if len(test_batch) == 3:
            images, labels, scan_ids = test_batch
            print(f"✓ Scan IDs working! ")
            print(f"  Sample scan IDs: {list(scan_ids[:3])}")
        else:
            print(f"✗ ERROR: Batch returns {len(test_batch)} items (expected 3: image, label, scan_id)")
            print("  Your dataset.py is NOT returning scan IDs!")
            raise ValueError("Dataset must return (image, label, scan_id)")
    except Exception as e:
        print(f"✗ CRITICAL ERROR: {e}")
        raise
    
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
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            num_classes=num_classes, embed_dims=embed_dims, depths=depths,
            mlp_ratios=mlp_ratios, drop_rate=drop_rate,
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
    print(f"Loss: Standard CrossEntropyLoss")
    
    if use_mixup:
        print(f"Mixup: ENABLED (alpha={mixup_alpha})")
    else:
        print(f"Mixup: DISABLED")
    
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate,
        weight_decay=weight_decay, betas=(0.9, 0.999)
    )
    
    total_steps = num_epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            return max(min_lr / learning_rate, cosine_decay)
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"Scheduler: Warmup ({warmup_epochs} epochs) + Cosine Annealing")
    
    # Early stopping based on SCAN-LEVEL accuracy
    early_stopping = EarlyStopping(
        patience=early_stopping_patience, 
        min_delta=min_delta, 
        mode='max'
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_slice_acc': [], 'train_slice_f1': [], 'train_slice_auc': [],
        'val_loss': [], 'val_slice_acc': [], 'val_slice_f1': [], 'val_slice_auc': [],
        'val_scan_acc': [], 'val_scan_f1': [], 'val_scan_auc': [],
        'learning_rates': []
    }
    
    best_val_scan_acc = 0.0
    best_val_scan_auc = 0.0
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training (SCAN-LEVEL Optimization)")
    print("="*60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, use_amp,
            use_mixup=use_mixup, mixup_alpha=mixup_alpha
        )
        
        # Validate (with scan-level metrics!)
        val_metrics, val_labels, val_preds, val_probs, val_scan_ids = validate(
            model, test_loader, criterion, device
        )
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['train_slice_acc'].append(train_metrics['slice_accuracy'])
        history['train_slice_f1'].append(train_metrics['slice_f1'])
        history['train_slice_auc'].append(train_metrics['slice_auc'] or 0)
        
        history['val_loss'].append(val_metrics['loss'])
        history['val_slice_acc'].append(val_metrics['slice_accuracy'])
        history['val_slice_f1'].append(val_metrics['slice_f1'])
        history['val_slice_auc'].append(val_metrics['slice_auc'] or 0)
        
        # Scan-level metrics (only for validation)
        history['val_scan_acc'].append(val_metrics.get('scan_accuracy', 0))
        history['val_scan_f1'].append(val_metrics.get('scan_f1', 0))
        history['val_scan_auc'].append(val_metrics.get('scan_auc', 0) or 0)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print metrics
        print(f"Train → Loss: {train_metrics['loss']:.4f} | "
              f"Slice Acc: {train_metrics['slice_accuracy']:.4f}")
        
        print(f"Val   → Loss: {val_metrics['loss']:.4f} | "
              f"Slice Acc: {val_metrics['slice_accuracy']:.4f}")
        
        # Highlight scan-level metrics
        if 'scan_accuracy' in val_metrics:
            scan_acc = val_metrics['scan_accuracy']
            scan_f1 = val_metrics.get('scan_f1', 0)
            num_scans = val_metrics.get('num_scans', 0)
            print(f"⭐ SCAN-LEVEL → Acc: {scan_acc:.4f} ({scan_acc*100:.2f}%) | F1: {scan_f1:.4f} | Scans: {num_scans}")
            
            if scan_acc >= 0.80:
                print(f"🎯 TARGET REACHED! Scan accuracy = {scan_acc*100:.2f}%")
        
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model (based on SCAN-LEVEL accuracy)
        current_scan_acc = val_metrics.get('scan_accuracy', 0)
        if current_scan_acc > best_val_scan_acc:
            best_val_scan_acc = current_scan_acc
            best_val_scan_auc = val_metrics.get('scan_auc', 0) or 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_scan_acc': best_val_scan_acc,
                'val_scan_auc': best_val_scan_auc,
                'val_metrics': val_metrics,
                'model_config': {
                    'model_size': model_size,
                    'img_size': img_size,
                    'num_classes': num_classes,
                    'embed_dims': embed_dims,
                    'depths': depths
                }
            }, save_dir / 'best_model.pth')
            print(f"✓ Best model saved (Scan Acc: {best_val_scan_acc:.4f})")
        
        # Early stopping (based on scan-level accuracy)
        early_stopping(current_scan_acc)
        if early_stopping.early_stop:
            print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
            break
    
    # Final evaluation
    print("\n" + "="*60)
    print("Training Completed! Generating Final Reports...")
    print("="*60)
    
    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_metrics, final_labels, final_preds, final_probs, final_scan_ids = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\n📊 Final Results (Best Model):")
    print(f"  Slice Accuracy:  {final_metrics['slice_accuracy']:.4f}")
    print(f"  ⭐ SCAN Accuracy:  {final_metrics.get('scan_accuracy', 0):.4f} ({final_metrics.get('scan_accuracy', 0)*100:.2f}%)")
    print(f"  Scan F1:         {final_metrics.get('scan_f1', 0):.4f}")
    print(f"  Scan Precision:  {final_metrics.get('scan_precision', 0):.4f}")
    print(f"  Scan Recall:     {final_metrics.get('scan_recall', 0):.4f}")
    print(f"  Scan Specificity:{final_metrics.get('scan_specificity', 0):.4f}")
    if final_metrics.get('scan_auc'):
        print(f"  Scan AUC:        {final_metrics['scan_auc']:.4f}")
    
    # Plot training history
    plot_training_history(history, save_dir)
    
    # Plot confusion matrices
    plot_confusion_matrix(final_labels, final_preds, 
                         save_dir / 'confusion_matrix_slice_level.png',
                         title='Slice-Level Confusion Matrix')
    
    # Scan-level confusion matrix (CRITICAL: Must use aggregated data!)
    if 'scan_confusion_matrix' in final_metrics:
        cm = final_metrics['scan_confusion_matrix']
        
        # Verify this is actually different from slice-level
        slice_cm = confusion_matrix(final_labels, final_preds)
        if np.array_equal(cm, slice_cm):
            print("⚠️  WARNING: Scan-level CM is identical to slice-level! Check aggregation.")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                    xticklabels=['AD', 'NC'], yticklabels=['AD', 'NC'],
                    cbar_kws={'label': 'Count'})
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Scan-Level Confusion Matrix (n={final_metrics.get("num_scans", 0)} scans)')
        
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        labels = np.array([['TP', 'FN'], ['FP', 'TN']])
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j + 0.5, i + 0.5, f"{labels[i, j]}\n{cm[i, j]}\n({cm_percent[i, j]:.1f}%)",
                     ha='center', va='center', fontsize=11, 
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrix_scan_level.png', dpi=300)
        plt.close()
        print(f"✓ Scan-level confusion matrix saved (n={final_metrics.get('num_scans', 0)} scans)")
    
    # Plot ROC curves
    if final_metrics.get('slice_auc'):
        plot_roc_curve(final_labels, final_probs, 
                      save_dir / 'roc_curve_slice_level.png',
                      title='Slice-Level ROC Curve')
    
    if final_metrics.get('scan_auc'):
        # Create scan-level ROC
        df = pd.DataFrame({
            'scan_id': final_scan_ids,
            'y_true': final_labels,
            'y_prob': final_probs
        })
        scan_results = df.groupby('scan_id').agg({
            'y_true': 'first',
            'y_prob': 'mean'
        }).reset_index()
        
        plot_roc_curve(scan_results['y_true'].values, 
                      scan_results['y_prob'].values,
                      save_dir / 'roc_curve_scan_level.png',
                      title='Scan-Level ROC Curve')
    
    # Save final metrics
    final_report = {
        'best_val_scan_accuracy': float(best_val_scan_acc),
        'best_val_scan_auc': float(best_val_scan_auc),
        'final_slice_metrics': {
            k: float(v) if v is not None else None 
            for k, v in final_metrics.items() if 'slice' in k
        },
        'final_scan_metrics': {
            k: float(v) if v is not None and not isinstance(v, np.ndarray) else None 
            for k, v in final_metrics.items() if 'scan' in k and k != 'scan_confusion_matrix'
        },
        'model_config': {
            'model_size': model_size,
            'total_params': total_params,
            'img_size': img_size,
            'num_classes': num_classes
        },
        'training_config': {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'use_mixup': use_mixup,
            'mixup_alpha': mixup_alpha if use_mixup else None
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
        # Model
        model_size='base',
        drop_rate=0.1,
        drop_path_rate=0.15,
        
        # Training
        img_size=224,
        batch_size=24,
        num_epochs=350,
        learning_rate=5e-5,
        min_lr=1e-7,
        weight_decay=0.05,
        warmup_epochs=10,
        
        # Early stopping
        early_stopping_patience=25,
        min_delta=0.001,
        
        # Augmentation 
        use_mixup=True,
        mixup_alpha=0.4, 
        
        # Options
        use_amp=True,
        
        # Save
        save_dir="./checkpoints"
    )
    
    print("\n" + "="*60)
    print("🎉 Training Complete!")
    print("="*60)
    print(f"Best Scan-Level Accuracy: {max(history['val_scan_acc']):.4f} ({max(history['val_scan_acc'])*100:.2f}%)")
    print(f"Best Scan-Level F1: {max(history['val_scan_f1']):.4f}")
    if history['val_scan_auc'] and max(history['val_scan_auc']) > 0:
        print(f"Best Scan-Level AUC: {max(history['val_scan_auc']):.4f}")
    print(f"\nFinal Slice-Level Accuracy: {metrics['slice_accuracy']:.4f}")
    print(f"Final Scan-Level Accuracy: {metrics.get('scan_accuracy', 0):.4f}")
