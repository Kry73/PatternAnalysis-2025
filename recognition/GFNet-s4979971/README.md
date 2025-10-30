# Global Filter Network (GFNet) for Alzheimer's Disease Classification

## Table of Contents
- [Introduction](#introduction)
- [Alzheimer’s Disease (AD)](#alzheimers-disease-ad)
- [Dataset](#dataset)
- [Model Implementation](#model-implementation)
- [Training & Evaluation](#training--evaluation)
- [Results](#results)
- [Installation & Usage](#installation--usage)
- [File Structure](#file-structure)
- [References](#references)
- [Acknowledgements](#acknowledgements)

## Introduction

### GFNet
GFNet is a frequency-domain neural architecture that replaces self-attention layer in traditional Vision Transformers (ViT) models with a global filter layer.  

![Overall Architecture of the Global Filter Network](images/GFNet_Archi.gif "Overall Architecture of the Global Filter Network")

The Global Filter Layer consists of three main operations, namely:
1. 2D Discrete Fourier Transform (DFT) to convert the input spatial features to the frequency domain
2. Element-wise multiplication between frequency-domain features and the learnable global filter
3. 2D Inverse Fourier Transform (IDFT) to map the features back to the spatial domain

![Pseudocode of Global Filter Layer](images/GFN_Pseudocode.png "Pseudocode of Global Filter Layer")

In relation to other traditional Convolutional Neural Networks (CNN) and transformer-style models, GFNet has the following advantages:
1. **Improved computational efficiency:** the total computational complexity is only O(NlogN) compared the quadratic complexity of ViT
2. **Scalability to higher resolutions:** both Fourier Transforms (FFT and IFFT) being able to process sequences with arbitary length without additional learnable parameters.This enables seamless interpolation across different input resolutions.

### Alzheimer's Disease (AD)
AD is the biological process that begins with the builup of proteins in the form of amyloid plaques and neurofibrillary tangles in the brain. As the result, brain cells die over time and causes the brain to shrink.

Currently, there is no known cure for AD. Nevertheless, medicines may improve symptoms or slow the decline in thinking, making it imperative diagnose AD as accurately and as early as possible. Over the last few decades, neuroimaging has moved from  minor to a central position in diagnosing AD. In particular, Magnetic Resonance Imaging (MRI) has helped in visualising progressive cerebral atrophy, which is a characteristic feature of neurodegeneration. 

Nevertheless, early diagnosis remains one of the current challenges in the field of neuroimaging Traditional CNNs are limited by their local receptive fields, while transformer-based models often demand large datasets to learn long-range dependencies effectively. GFNet offers a promising alternative by efficiently modeling global spatial relationships through frequency-domain filtering. 

In this study, GFNet is adapted to the Alzheimer’s Disease Neuroimaging Initiative (ADNI) MRI dataset to classify subjects into Alzheimer’s Disease (AD), and Normal Control (NC) groups.

## Dataset
This project uses the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset for binary classification of brain MRI scans into two classes:
*AD (Alzheimer's Disease): Diagnosed Alzheimer's patients
*NC (Normal Control): Cognitively normal subjects

**Dataset Statistic**
* Total Scans: 2,189 unique 3D MRI scans
* Total Images: 30,520 2D slices
* Train Set: 21,520 images from 1,076 unique scans
    - AD: 10,400 images (48.3%)
    - NC: 11,120 images (51.7%)
*Test Set: 9,000 images from 450 unique scans
    - AD: 4,460 images (49.6%)
    - NC: 4,540 images (50.4%)
*Average slices per scan: 20.0 slices
*Classes: 2 (AD, NC) - Well balanced at approximately 50/50

**Directory Structure**
```
ADNI/
├── meta_data_with_label.json
└── AD_NC/
    ├── train/
    │   ├── AD/    # 10,400 images
    │   └── NC/    # 11,120 images
    └── test/
        ├── AD/    # 4,460 images
        └── NC/    # 4,540 images
```

### dataset.py

get_data_loaders(ddata_dir, batch_size=32, img_size=224, num_workers=4)
Data loaders for GFNet training on ADNI AD/NC dataset.
Medical imaging-aware preprocessing for brain MRI scans.
To discuss:
*AutoCropBlack
*Augmentation Parameters: https://pmc.ncbi.nlm.nih.gov/articles/PMC10157370/

## Model Implementation

## Training and Evaluation

## Results

## Installation

## Dependencies

## File Structure

```
recognition/
└── GFNet-s4979971
    ├── dataset.py      # Data loader and preprocessing
    ├── modules.py      # Model components (GFBlock, GlobalFilter, GFNet)
    ├── train.py        # Training, validation, testing pipeline
    ├── predict.py      # Example usage
    ├── README.md       # This file
    ├── images          # Folder containing diagrams and visualisation
```

## References - TBC
https://arxiv.org/pdf/2107.00645
https://radiologyassistant.nl/neuroradiology/dementia/role-of-mri
https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/symptoms-causes/syc-20350447
https://pmc.ncbi.nlm.nih.gov/articles/PMC3312396/

## Acknowledgement
