# Global Filter Network (GFNet) for Alzheimer's Disease Classification

## Table of Contents
- [Introduction](#introduction)
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
* AD (Alzheimer's Disease): Diagnosed Alzheimer's patients
* NC (Normal Control): Cognitively normal subjects

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
In this file, the datasets are loaded and data augmentations are applied for GFnet training on ADNI AD/NC dataset.

In the medical domain, data augmentation is important in improving a model robustness, especially in the case of low volume of datasets due to privacy issue or rarity of diseases. In this case, the size of the dataset is moderate, and thus appropriate data augmentation is needed to increase the effective variability of the training data, reduce overfitting and improve the model's generalisation to unseen MRI scans.

The training augmentation pipeline includes:
```ruby
train_transform = transforms.Compose([
        AutoCropBlack(threshold=10),
        transforms.Resize((img_size, img_size)),
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
```
**Cropping & Scaling**
```ruby
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
```
AutoCropBlack is a custom transform that removes non-informative black borders from MRI scans. Many MRI images include large black regions outside the brain due to scanner padding or acquisition settings. AutoCropBlack converts the image to grayscale, identifies all pixels above a specified intensity threshold, and crops the image to the smallest rectangle containing these pixels. This ensures the model focuses on relevant brain structures, reduces background noise, and saves memory and computation during training

After cropping, each scan is scaled back up to consistent size to standardise the input dimensions across the dataset.

Notably, in A Comparative Analysis of Data Augmentation[^1] study found that crop & scale approach achieved the second-best classification accuracy, supporting the effectiveness of this technique for medical imaging tasks.

**Random Affine**
RandomAffine applies small random rotations, translations, and scaling to each MRI slice, simulating natural variations that occur during patient positioning or image acquisition. In medical imaging, even slight head movements or scanner alignment differences can lead to spatial inconsistencies between scans. 

Hence, by introducing controlled geometric perturbations, this transformation enhances the model’s ability to generalize and accurately recognize brain structures under varying spatial conditions. The magnitude of each perturbation is adapted from studies on brain tumor detection, where morphological variations are more pronounced. In contrast, since Alzheimer’s-related structural changes are subtler, our chosen parameters are intentionally more conservative to preserve anatomical integrity and avoid excessive distortion.

**Others**
Horizontal flipping is applied with a 50% probability to introduce left–right symmetry variations in the training data. Although brain structures are largely symmetrical, subtle asymmetries can occur due to individual anatomy or disease progression. Incorporating horizontal flips helps the model remain invariant to spatial orientation while still learning relevant lateralized features.

Color jittering adjusts brightness and contrast within controlled limits (±0.2), simulating natural intensity variations that may arise from different MRI scanners or acquisition parameters. This enhances the robustness of the model against scanner-dependent artifacts and illumination inconsistencies.

Following these augmentations, images are converted into tensors and normalized using the dataset’s mean and standard deviation, ensuring consistent input scaling across all batches. Finally, RandomErasing is applied with a probability of 0.5, randomly masking small regions of the image. This technique acts as a form of regularization, preventing the model from over-relying on specific local features and encouraging more distributed, context-aware learning of brain morphology.

![Augmentation Diversity](images/augmentation_sample_1_AD.png "Augmentation Diversity")

For testing and evaluation, only deterministic preprocessing steps are applied to ensure consistent result:
```ruby
test_transform = transforms.Compose([
        AutoCropBlack(threshold=10),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2670], std=[0.2657])
    ])
```

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
## Footnotes
[^1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7085309/
## References - TBC
https://arxiv.org/pdf/2107.00645
https://radiologyassistant.nl/neuroradiology/dementia/role-of-mri
https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/symptoms-causes/syc-20350447
https://pmc.ncbi.nlm.nih.gov/articles/PMC3312396/
https://www.sciencedirect.com/science/article/pii/S277244252400042X#b111
https://pmc.ncbi.nlm.nih.gov/articles/PMC10157370/

## Acknowledgement
