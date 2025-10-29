# Data loader for loading and preprocessing data
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

class AutoCropBlack:
    """Remove black borders from MRI scans"""
    def __init__(self, threshold=10):
        self.threshold = threshold
    
    def __call__(self, img):
        gray = img.convert('L')
        gray_np = np.array(gray)
        mask = gray_np > self.threshold
        
        if not np.any(mask):
            return img
        
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        
        return img.crop((x0, y0, x1, y1))


def get_data_loaders(data_dir, batch_size=32, img_size=224, num_workers=4):
    """
    Data loaders for GFNet training on ADNI AD/NC dataset.
    Medical imaging-aware preprocessing for brain MRI scans.
    """
    
    # TRAINING: Medical-appropriate augmentation
    train_transform = transforms.Compose([
        AutoCropBlack(threshold=10),              # Remove scanner artifacts
        transforms.Grayscale(num_output_channels=1),  # MRI is grayscale
        transforms.RandomResizedCrop(
            (img_size, img_size), 
            scale=(0.80, 1.0)
        ),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.90, 1.1),
            shear=8
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Intensity variation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2670], std=[0.2657]),  # Your dataset stats
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.25))
    ])
    
    # TESTING: Deterministic only
    test_transform = transforms.Compose([
        AutoCropBlack(threshold=10),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2670], std=[0.2657])
    ])
    
    # Load datasets
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise ValueError(f"Dataset not found at {data_dir}")
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Logging
    print(f"\n{'='*70}")
    print(f"GFNet ADNI Dataset Loaded")
    print(f"{'='*70}")
    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Input: {img_size}×{img_size} grayscale (1 channel)")
    print(f"Preprocessing: AutoCropBlack → Grayscale → Augment → Normalize")
    print(f"{'='*70}\n")
    
    return train_loader, test_loader


# Usage for GFNet
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders(
        data_dir="/home/groups/comp3710/ADNI/AD_NC",
        batch_size=32,
        img_size=224,
        num_workers=4
    )
    
    # Verify data
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")  # Should be [32, 1, 224, 224]
    print(f"Labels: {labels}")
    print(f"Value range: [{images.min():.3f}, {images.max():.3f}]")