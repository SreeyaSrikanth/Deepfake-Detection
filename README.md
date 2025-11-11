[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SreeyaSrikanth/Deepfake-Detection/blob/main/fds-da.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Deepfake Detection using Microexpression Analysis

A novel deepfake detection system that analyzes microexpressions using **Framewise Optical Flow** and **Facial Action Units (AUs)** with a hybrid CNN + Bi-LSTM architecture.

## 🎯 Overview

Deepfakes can replicate facial features accurately but struggle to mimic involuntary microexpressions. This project exploits that limitation by:
- Extracting facial dynamics using framewise optical flow between consecutive frames
- Analyzing 46 Facial Action Units based on the Facial Action Coding System (FACS)
- Fusing spatial and behavioral features for robust detection

**Key Results:** 97.64% accuracy | 0.98 F1-score | 0.99 AUC-ROC

## 🏗️ Architecture

<img width="948" height="633" alt="image" src="https://github.com/user-attachments/assets/5afca992-14d2-4a37-b556-97974b8e2728" />

### Model Components

**Dual-Branch Architecture:**
1. **CNN Branch (ResNet-18)**: Extracts 512-dim spatial features from Optical Flow images
2. **MLP Branch**: Processes 17 AU features → 128-dim semantic vectors
3. **Fusion Layer**: Concatenates features (640-dim)
4. **Bi-LSTM Layer**: Processes fused features
5. **Classification**: Fully connected layer → Binary output (Real/Fake)

### Algorithm Pipeline
```
Input: RGB Images (B×3×H×W) + AU Features (B×n)

1. X_img ← ResNet18(I)                  // 512-dim image features
2. X_au ← ReLU(W_au · A + b_au)         // 128-dim AU features
3. X_fused ← Concat(X_img, X_au)        // 640-dim fusion
4. H ← Bi-LSTM(X_fused)                 // Bi-directional processing
5. P ← W_cls · H[:,-1,:] + b_cls        // Classification logits
6. Loss ← CrossEntropy(P, Y)            // Training loss

Output: Real vs Fake (with confidence score)
```

## 📊 Results

### Performance Metrics

| Metric     | Score     |
|------------|-----------|
| Accuracy   | 97.64%    |
| Precision  | 0.98      |
| Recall     | 0.99      |
| F1-Score   | 0.98      |
| AUC-ROC    | 0.99      |

### Cross-Validation (5-Fold)
- Mean Accuracy: 94.87%
- Mean F1-Score: 0.9475

### Confusion Matrix
<img width="646" height="574" alt="image" src="https://github.com/user-attachments/assets/b24c314f-b20e-4274-96f7-01f877912afb" />

## 🔬 Methodology

### Feature Extraction

**1. Framewise Optical Flow**
- Dense optical flow using RAFT algorithm
- Captures pixel-wise motion patterns between consecutive frames
- Brightness constancy constraint: `I(x+Δx, y+Δy, t+Δt) ≈ I(x, y, t)`
- Output: Flow field representing dynamic facial deformations

**2. Action Units (AU) via OpenFace 2.0**
- 46 AUs based on FACS (Facial Action Coding System)
- Features extracted per video:
  - **Intensity statistics**: mean, max, std, range
  - **Activation ratio**: proportion of frames where AU is active
  - **Frequency**: number of frames with AU activation
  - **Duration patterns**: mean and max length of continuous activation
  - **Co-activation patterns**: pairwise co-occurrence (e.g., AU06 + AU12)

### Training Details
- **Dataset**: DFDC (DeepFake Detection Challenge) - 400 videos
- **Split**: 80/20 train-validation
- **Preprocessing**: Frame extraction, resize to 224×224, normalization
- **Batch Size**: 32
- **Optimizer**: Adam (lr=1e-4, weight decay=1e-5)
- **Epochs**: 7
- **Loss**: Cross-Entropy
- **Regularization**: Dropout + L2

## 🛠️ Technologies

- **PyTorch**: Deep learning framework
- **ResNet-18**: Pre-trained on ImageNet
- **RAFT**: Optical flow estimation
- **OpenFace 2.0**: AU extraction
- **OpenCV**: Video processing
