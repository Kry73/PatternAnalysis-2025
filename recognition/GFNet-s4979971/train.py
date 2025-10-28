# Source code for  training, validating, testing, and saving model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from modules import GFNet
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


def calculate_metrics(y_true, y_pred):
    """Calculate accuracy, precision, recall, F1 score."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['AD', 'NC'], yticklabels=['AD', 'NC'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, use_amp=True):
    """Train for one epoch with mixed precision."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
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
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate epoch metrics
    avg_loss = running_loss / len(train_loader)
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = avg_loss
    
    return metrics


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(val_loader, desc='Validation', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    avg_loss = running_loss / len(val_loader)
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = avg_loss
    
    return metrics, all_labels, all_preds


def plot_training_history(history, save_dir):
    """Plot training and validation metrics."""
    save_dir = Path(save_dir)
    
    # Plot loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=300)
    plt.close()


def train_model(
    # Model parameters
    img_size=224,
    patch_size=16,
    in_chans=1,
    num_classes=2,
    embed_dim=384,  # Increased from 256 for better capacity
    depth=12,
    mlp_ratio=4.0,
    drop_rate=0.1,
    drop_path_rate=0.2,  # Increased for better regularization
    
    # Training parameters
    batch_size=32,
    num_epochs=100,
    learning_rate=1e-4,
    weight_decay=0.05,  # Increased for better regularization
    warmup_epochs=5,
    
    # Data parameters
    data_dir="/home/groups/comp3710/ADNI/AD_NC",
    num_workers=4,
    
    # Training options
    use_amp=True,  # Mixed precision training
    early_stopping_patience=15,
    
    # Save options
    save_dir="./checkpoints",
    save_best_only=True
):

    """
    Main training function with improved practices.
    
    Key improvements:
    - Mixed precision training (faster, less memory)
    - Early stopping
    - Comprehensive metrics tracking
    - Checkpoint saving
    - Better default hyperparameters
    """
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "="*60)
    print("Loading Data...")
    print("="*60)
    train_loader, test_loader= get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers
    )
    
    # Initialize model
    print("\n" + "="*60)
    print("Initializing Model...")
    print("="*60)
    model = GFNet(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler with warmup
    total_steps = num_epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, mode='min')
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, use_amp
        )
        
        # Validate
        val_metrics, val_labels, val_preds = validate(
            model, test_loader, criterion, device
        )
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.4f} | "
              f"Train F1: {train_metrics['f1']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_metrics': val_metrics
            }, save_dir / 'best_model.pth')
            print(f"✓ Best model saved (Val Acc: {best_val_acc:.4f})")
        
        # Save latest checkpoint
        if not save_best_only:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, save_dir / 'latest_checkpoint.pth')
        
        # Early stopping
        early_stopping(val_metrics['loss'])
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Save final results
    print("\n" + "="*60)
    print("Training Completed!")
    print("="*60)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    
    # Plot training history
    plot_training_history(history, save_dir)
    
    # Plot final confusion matrix
    plot_confusion_matrix(val_labels, val_preds, save_dir / 'confusion_matrix.png')
    
    # Save history as JSON
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"\nAll results saved to: {save_dir}")
    
    return model, history


if __name__ == "__main__":
    # Train with default parameters
    model, history = train_model()
    
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Final Train Accuracy: {history['train_acc'][-1]:.4f}")
    print(f"Final Val Accuracy: {history['val_acc'][-1]:.4f}")
    print(f"Best Val Accuracy: {max(history['val_acc']):.4f}")