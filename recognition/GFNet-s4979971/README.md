# Global Filter Network (GFNET)

## File Structure
recognition/
└── GFNet-s4979971
    ├── dataset.py      # Data loader and preprocessing
    ├── modules.py      # Model components (GFBlock, GlobalFilter, GFNet)
    ├── train.py        # Training, validation, testing pipeline
    ├── predict.py      # Example usage
    ├── README.md       # This file


## Dependencies

## Installation

## Implementation

### Dataset
This project uses the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset for binary classification of brain MRI scans into two classes:
* AD (Alzheimer's Disease): Diagnosed Alzheimer's patients
* NC (Normal Control): Cognitively normal subjects

** Dataset Statistic **
* Total Scans: 2,189 unique 3D MRI scans
* Total Images: 30,520 2D slices
* Train Set: 21,520 images from 1,076 unique scans
    - AD: 10,400 images (48.3%)
    - NC: 11,120 images (51.7%)
* Test Set: 9,000 images from 450 unique scans
    - AD: 4,460 images (49.6%)
    - NC: 4,540 images (50.4%)
* Average slices per scan: 20.0 slices
* Classes: 2 (AD, NC) - Well balanced at approximately 50/50

** Directory Structure **
ADNI/
├── meta_data_with_label.json
└── AD_NC/
    ├── train/
    │   ├── AD/    # 10,400 images
    │   └── NC/    # 11,120 images
    └── test/
        ├── AD/    # 4,460 images
        └── NC/    # 4,540 images

### dataset.py

get_data_loaders(ddata_dir, batch_size=32, img_size=224, num_workers=4)
Data loaders for GFNet training on ADNI AD/NC dataset.
Medical imaging-aware preprocessing for brain MRI scans.

To discuss:
* AutoCropBlack
* Augmentation Parameters: https://pmc.ncbi.nlm.nih.gov/articles/PMC10157370/




## Results
