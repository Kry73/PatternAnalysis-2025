# Global Filter Network (GFNet) for Alzheimer's Disease Classification

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Implementation](#model-implementation)
- [Training & Evaluation](#training--evaluation)
- [Results](#results)
- [Usage](#usage)
- [File Structure](#file-structure)
- [References](#references)

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
In this file, the datasets are loaded and data augmentations are applied for GFnet training on ADNI AD/NC dataset. It also stores ADNIDatasetWithScanID class

### Data Augmentation
In the medical domain, data augmentation is important in improving a model robustness, especially in the case of low volume of datasets due to privacy issue or rarity of diseases. In this case, the size of the dataset is moderate, and thus appropriate data augmentation is needed to increase the effective variability of the training data, reduce overfitting and improve the model's generalisation to unseen MRI scans.

**Training**\
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
<ins>Cropping & Scaling</ins>

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

<ins>Random Affine</ins>\
RandomAffine applies small random rotations, translations, and scaling to each MRI slice, simulating natural variations that occur during patient positioning or image acquisition. In medical imaging, even slight head movements or scanner alignment differences can lead to spatial inconsistencies between scans. 

Hence, by introducing controlled geometric perturbations, this transformation enhances the model’s ability to generalize and accurately recognize brain structures under varying spatial conditions. The magnitude of each perturbation is adapted from studies on brain tumor detection, where morphological variations are more pronounced. In contrast, since Alzheimer’s-related structural changes are subtler, our chosen parameters are intentionally more conservative to preserve anatomical integrity and avoid excessive distortion.

<ins>Others</ins>\
Horizontal flipping is applied with a 50% probability to introduce left–right symmetry variations in the training data. Although brain structures are largely symmetrical, subtle asymmetries can occur due to individual anatomy or disease progression. Incorporating horizontal flips helps the model remain invariant to spatial orientation while still learning relevant lateralized features.

Color jittering adjusts brightness and contrast within controlled limits (±0.2), simulating natural intensity variations that may arise from different MRI scanners or acquisition parameters. This enhances the robustness of the model against scanner-dependent artifacts and illumination inconsistencies.

Following these augmentations, images are converted into tensors and normalized using the dataset’s mean and standard deviation, ensuring consistent input scaling across all batches. Finally, RandomErasing is applied with a probability of 0.5, randomly masking small regions of the image. This technique acts as a form of regularization, preventing the model from over-relying on specific local features and encouraging more distributed, context-aware learning of brain morphology.

![Augmentation Sample](images/augmentation_sample_1_AD.png "Augmentation Sample")

**Testing**\
For testing and evaluation, only deterministic preprocessing steps are applied to ensure consistent result:
```ruby
test_transform = transforms.Compose([
        AutoCropBlack(threshold=10),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2670], std=[0.2657])
    ])
```
### ADNIDDatasetWithScanID
```ruby
class ADNIDatasetWithScanID(Dataset):
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
```
This class extends the standard PyTorch dataset to include scan-level metadata extraction from the file names of the ADNI dataset. Each MRI slice in the dataset is named following the format scanID_slicenumber.jpeg (e.g., 1031067_85.jpeg). This allows the dataset loader to identify not just the class label (e.g., AD or NC), but also the unique scan ID corresponding to the MRI volume from which each 2D slice originates.

This is crucial to implement scan-level

## Model Implementation
This project implements PyramidGFNet (PGFNet), a hierarchical extension of GFNet, which has 4 stages, each marked with varying fearure map resoultion. This design enables the network to learn both fine-grained local textures and high-level semantic representation, similar to vision transformers but with substantially lower computational overhead.

![Pyramid GFNet](images/PGFNet.jpg "Pyramid GFNet")

A study[^2] has shown that PGFNet achieved superior performance, rivaling the SOTA EfficientNet. Although the cited study evaluated PGFNet on a coal ash content estimation task, its relevance extends to MRI brain imaging because both involve analyzing fine-grained grayscale textures. In both coal imaging and MRI scans, the model must detect subtle intensity and structural variations rather than color cues. 

Furthermore SOTA EfficientNet is well known for capturing hierarchical spatial features effectively even in single-channel data. Hence, the fact that PGFNet rivals SOTA EfficientNet in such a grayscale-dependent and texture-sensitive task suggests that PGFNet can similarly excel in MRI-based classification, where discriminative patterns of brain tissue, rather than color differences, are crucial.

In this implementation, PGFNet architecture is implemented in three main variants:
| Variant  | Parameters |Depth |
| ------------- | ------------- |------------- |
| Tiny | ~13M  |[2, 2, 6, 2] |
| Small | ~26M  |[3, 4, 6, 3] |
| Base | ~44M  |[3, 4, 18, 3] |

## Training and Evaluation

### Training
training.py contains the main training loop for the model.

**Final Parameter**
```r
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

```
The model was trained using the PGFNet (Base) variant after the smaller variant plateaued at around 76–77% validation accuracy, suggesting that a deeper and higher-dimensional model was necessary to better capture the subtle morphological differences in Alzheimer’s pathology. The base configuration increases both embedding dimensions and hierarchical depth, enabling richer spatial-frequency representation of MRI features while remaining computationally efficient when trained with Automatic Mixed Precision (AMP).

Several key mechanisms are implemented to enhance generalisation and stability during training.

**Cross-Entropy Loss with Label Smoothing**\
A cross-entropy loss with label smoothing (ε = 0.1) is employed to prevent overconfidence and improve generalization. Label smoothing introduces controlled uncertainty in the target distribution, preventing the model from becoming overly certain about its predictions. This is particularly valuable in Alzheimer’s Disease (AD) classification, where MRI slices often contain ambiguous or subtle anatomical differences between AD and normal control (NC) subjects.

**Mixup Augmentation**\
Mixup is a technique where the learner combines pairs of training instances to produce a virtual third instance that is a linear combination of the two instances and their labels. Following the implementation of this approach in pyschosis fMRI classification[^3], the authors found that an optimal coefficient of α = 0.2 yielded the best performance. Psychosis and Alzheimer’s disease share similarities in terms of subtle, diffuse structural and functional brain changes, making Mixup a suitable augmentation for both.

However, in our study, we use α = 0.4, higher than the psychosis study. The reason is the fundamental difference in scan types. This project's dataset consists of structural MRI slices of the brain, each representing a discrete anatomical section, whereas fMRI captures dynamic functional activity over time. Slices in structural MRI can be more heterogeneous, and larger interpolation (higher α) allows the model to better generalize across patients and slice variability, without introducing unrealistic images. This adjustment helps the model learn robust representations for scan-level classification, mitigating overfitting and improving performance compared to using the lower 

**Early Stop Loss**\
Early stopping (patience = 25, min Δ = 0.001) was incorporated to prevent overtraining, particularly given the moderate dataset size. The patience value and small delta thresholds are chosen to allow the model a few epochs to recover from minor fluctuations in validation performance. This reduces the risk of stopping too early due to random noise in the validation set. Additionally, as training deep models like PGFNet can be time-intensive, early stopping allows us to halt training once performance plateaus, avoiding unnecessary epochs and reducing GPU hours.

## Results
⚠️ Note: Results are not fully reproducible due to missing random seed control. Running the same training may produce results varying by ±1-3%.

## Usage
### Dependencies
```ruby
# Deep Learning Framework
torch>=1.8.0              # Required for torch.fft (rfft2/irfft2)
torchvision>=0.15.0

# Model Components
timm>=0.9.0               # PyTorch Image Models (for DropPath)

# Numerical & Image Processing
numpy>=1.24.0
Pillow>=9.0.0             # PIL for image loading

# Data Analysis
pandas>=1.5.0             # For CSV handling and predictions

# Visualization & Metrics
matplotlib>=3.7.0
seaborn>=0.12.0           # Enhanced plotting for confusion matrices
scikit-learn>=1.2.0       # Metrics and evaluation

# Progress & Utilities
tqdm>=4.65.0              # Progress bars

# Standard Library (Built-in)
os, pathlib, collections, functools, json, math, warnings, random
```

### Training
**Basic usage**
```
python train.py
```
**Custom parameters**\
Modify the default parameter
```ruby
# Model parameters
model_size='small'              # Options: 'tiny', 'small', 'base'
drop_rate=0.1
drop_path_rate=0.15

# Training parameters
img_size=224
batch_size=24
num_epochs=250
learning_rate=1e-4
min_lr=1e-7
weight_decay=0.05
warmup_epochs=10

# Early stopping
early_stopping_patience=25
min_delta=0.001

# Augmentation
use_mixup=True
mixup_alpha=0.2

# Options
use_amp=True                    # Automatic Mixed Precision
use_class_weights=False

# Data
data_dir="/home/groups/comp3710/ADNI/AD_NC"
num_workers=4

# Save
save_dir="./checkpoints_scan_level"
```

### Evaluation
**Basic usage**
```
python predict.py
```

## File Structure

```
recognition/
└── GFNet-s4979971
    ├── dataset.py      # Data loader and preprocessing
    ├── modules.py      # Model components (GFBlock, GlobalFilter, GFNet)
    ├── train.py        # Training and validation
    ├── predict.py      # Evaluating the accuracy of a given model
    ├── README.md       # This file
    ├── images          # Folder containing diagrams and visualisation
```

## References - TBC
1. Barkhof, F., Hazewinkel, M., Binnewijzend, M., & Smithuis, R. (2022, March 3). Dementia - role of MRI. Radiology Assistant. https://radiologyassistant.nl/neuroradiology/dementia/role-of-mri 
2. Islam, T., Hafiz, Md. S., Jim, J. R., Kabir, Md. M., & Mridha, M. F. (2024, June 5). Https://www.sciencedirect.com/science/article/abs/pii/S1047847720300046?via=ihub. Science Direct. https://www.med.upenn.edu/pmi/events/https-www-sciencedirect-com-science-article-abs-pii-s1047847720300046-via-3dihub 
3. Johnson, K. A., Fox, N. C., Sperling, R. A., & Klunk, W. E. (2012, April). Brain Imaging in alzheimer disease. Cold Spring Harbor perspectives in medicine. https://pmc.ncbi.nlm.nih.gov/articles/PMC3312396 
4. Krishnapriya, S., & Karuna, Y. (2023, April 20). Pre-trained deep learning models for brain MRI image classification. Frontiers in human neuroscience. https://pmc.ncbi.nlm.nih.gov/articles/PMC10157370/ 
5. Mayo Foundation for Medical Education and Research. (2024, November 8). Alzheimer’s disease. Mayo Clinic. https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/symptoms-causes/syc-20350447 
6. Rao, Y., Zhao, W., Zhu, Z., Lu, J., & Zhou, J. (2021, October 26). Global Filter Networks for Image Classification. arXiv.org. https://arxiv.org/abs/2107.00645 
7. Safdar, M. F., Alkobaisi, S. S., & Zahra, F. T. (2020, March). A comparative analysis of data augmentation approaches for Magnetic Resonance Imaging (MRI) scan images of brain tumor. PubMed Central. https://pmc.ncbi.nlm.nih.gov/articles/PMC7085309/ 
8. Zhang, K., Wang,  eidong, Cui, Y., LV, Z., & Fan, Y. (2024, January). GFNet: A pioneering approach for precisely estimating ash content in coal through the fusion of graph convolution and feedforward network. Science Direct. https://www.med.upenn.edu/pmi/events/https-www-sciencedirect-com-science-article-abs-pii-s1047847720300046-via-3dihub 

[^1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7085309/
[^2]: https://www.sciencedirect.com/science/article/pii/S0952197623014859
[^3]: https://www.sciencedirect.com/science/article/pii/S2213158222002790
