import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from pathlib import Path

class AutoCropBlack:
    """Remove black borders from MRI scans"""
    def __init__(self, threshold=10):
        self.threshold = threshold
    
    def __call__(self, img):
        gray = img.convert('L')
        gray_np = np.array(gray)
        mask = gray_np > self.threshold
        
        if not np.any(mask):
            return gray
        
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        
        return gray.crop((x0, y0, x1, y1))


class ADNIDatasetWithScanID(Dataset):
    """
    ADNI Dataset that extracts and returns scan IDs from file paths.
    
    Expected path format: .../AD_NC/train/AD/scanid_slicenumber.jpeg
    Example: .../1031067_85.jpeg -> scan_id = "1031067", slice = "85"
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Collect all image paths and labels
        self.samples = []
        self.scan_ids = []
        self.classes = sorted([d.name for d in Path(root_dir).iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_dir = Path(root_dir) / class_name
            for img_path in list(class_dir.glob('*.jpeg')):
                # Extract scan_id from filename
                filename = img_path.stem  # Remove extension
                
                # Split by underscore and take everything except the last part (slice number)
                parts = filename.split('_')
                if len(parts) >= 2 and parts[-1].isdigit():
                    # Last part is slice number, everything before is scan_id
                    scan_id = '_'.join(parts[:-1])
                else:
                    # Fallback: use entire filename as scan_id
                    scan_id = filename
                
                self.samples.append((str(img_path), self.class_to_idx[class_name]))
                self.scan_ids.append(scan_id)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        scan_id = self.scan_ids[idx]
        
        # Load image
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, scan_id


def get_data_loaders(data_dir, batch_size=32, img_size=224, num_workers=4):
    """
    Data loaders for GFNet training on ADNI AD/NC dataset.
    Medical imaging-aware preprocessing for brain MRI scans.
    Include SCAN IDs for scan-level evaluation
    """
    
    # TRAINING: Medical-appropriate augmentation
    train_transform = transforms.Compose([
        AutoCropBlack(threshold=10),
        transforms.Resize((img_size, img_size)),  # Consistent sizing
        transforms.RandomAffine(
            degrees=8,                
            translate=(0.05, 0.05),   
            scale=(0.95, 1.05),       
            shear=None               
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2670], std=[0.2657]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.25))
    ])
    
    # TESTING: Deterministic only
    test_transform = transforms.Compose([
        AutoCropBlack(threshold=10),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2670], std=[0.2657])
    ])
    
    # Load datasets WITH SCAN IDs
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise ValueError(f"Dataset not found at {data_dir}")
    
    train_dataset = ADNIDatasetWithScanID(train_dir, transform=train_transform)
    test_dataset = ADNIDatasetWithScanID(test_dir, transform=test_transform)
    
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
    
    # Count unique scans
    unique_train_scans = len(set(train_dataset.scan_ids))
    unique_test_scans = len(set(test_dataset.scan_ids))
    
    # Calculate average slices per scan
    from collections import Counter
    avg_train_slices = len(train_dataset) / unique_train_scans if unique_train_scans > 0 else 0
    avg_test_slices = len(test_dataset) / unique_test_scans if unique_test_scans > 0 else 0
    
    # Logging
    print(f"\n{'='*70}")
    print(f"GFNet ADNI Dataset Loaded (WITH SCAN ID TRACKING)")
    print(f"{'='*70}")
    print(f"Train: {len(train_dataset)} slices from {unique_train_scans} unique scans")
    print(f"       (~{avg_train_slices:.1f} slices per scan)")
    print(f"Test:  {len(test_dataset)} slices from {unique_test_scans} unique scans")
    print(f"       (~{avg_test_slices:.1f} slices per scan)")
    print(f"Classes: {train_dataset.classes}")
    print(f"Input: {img_size}×{img_size} grayscale (1 channel)")
    print(f"Augmentation: FIXED (removed RandomResizedCrop, reduced affine)")
    
    # Show sample scan IDs
    if len(test_dataset.scan_ids) > 0:
        sample_scan_id = test_dataset.scan_ids[0]
        same_scan_count = sum(1 for sid in test_dataset.scan_ids if sid == sample_scan_id)
        print(f"\nSample verification:")
        print(f"  Scan ID '{sample_scan_id}' appears {same_scan_count} times")
    
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
    images, labels, scan_ids = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
    print(f"Scan IDs (first 5): {scan_ids[:5]}")
    print(f"Unique scans in batch: {len(set(scan_ids))}")
    print(f"Value range: [{images.min():.3f}, {images.max():.3f}]")