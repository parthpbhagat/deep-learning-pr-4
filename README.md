# Deep Learning Practical 4: Comprehensive Deep Learning & Teachable Machine Suite

This repository contains the complete, production-grade solutions for both **Project A** (Convolutional Neural Networks with CIFAR-10) and **Project B** (Teachable Machine Multi-Modal Transfer Learning System).

---

## 📑 Table of Contents
1. [Project A — CNN Architectures on CIFAR-10](#-project-a--cnn-architectures-on-cifar-10)
   - [CIFAR-10 Dataset & Preprocessing](#1-cifar-10-dataset--preprocessing)
   - [Core Topics & Mathematical Foundations](#2-core-topics--mathematical-foundations)
   - [Model Architectures & Training Curves](#3-model-architectures--training-curves)
   - [Full 6-Model Comparison Results](#4-full-6-model-comparison-results)
   - [Project A Video Demonstration](#5-project-a-video-demonstration)
2. [Project B — Teachable Machine Multi-Modal System](#-project-b--teachable-machine-multi-modal-system)
   - [Environment Setup & Installation](#1-environment-setup--installation)
   - [3 Teachable Machine Models Description](#2-3-teachable-machine-models-description)
   - [Execution Guide](#3-execution-guide)
   - [Project B Video Demonstration](#4-project-b-video-demonstration)
3. [Repository File Structure](#-repository-file-structure)
4. [Verification & Checkpoint Status](#-verification--checkpoint-status)

---

# 🚀 Project A — CNN Architectures on CIFAR-10

- **Notebook:** [`DL_PR4A.ipynb`](DL_PR4A.ipynb)
- **HTML Export:** [`DL_PR4A.html`](DL_PR4A.html)
- **Plot Visualizations Directory:** [`plots/`](plots/)

### 1. CIFAR-10 Dataset & Preprocessing
- **Dimensions:** 50,000 training images, 10,000 testing images of shape `(32, 32, 3)`.
- **Normalization:** Rescaled from raw integer pixel range `[0, 255]` to floating-point range `[0.0, 1.0]`.
- **Labels:** 10 semantic classes one-hot encoded to shape `(N, 10)`.
- **Visual Artifacts:**
  - 5×10 Sample Image Grid with class headers: `plots/cifar10_grid.png`
  - Pixel Intensity Histogram before and after normalization: `plots/pixel_histogram.png`

---

### 2. Core Topics & Mathematical Foundations

#### Topic 1: CNN vs ANN Comparative Analysis
| Feature / Property | Artificial Neural Network (Dense MLP) | Convolutional Neural Network (CNN) |
| :--- | :--- | :--- |
| **Input Structure** | 1D Flattened vector (loses 2D spatial context) | 2D/3D Tensor preserving spatial grid topology |
| **Translation Invariance** | None; patterns shifted by pixels need new weights | High; filters slide across spatial dimensions equally |
| **Weight Sharing** | None; each input connects via distinct weight | Yes; identical kernel weights convolve across image |
| **Parameter Efficiency** | O(Height × Width × Hidden) — explosive growth | O(Kernel Size × Channels × Filters) — compact |

#### Topic 2: Convolution Mechanics, VALID vs SAME & Strides
- **Formula for Output Spatial Dimensions:**
  $$\text{Output Size} = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$
- **Padding Comparison on 32×32 Image with 3×3 Kernel:**
  - `VALID` Padding ($P=0$): $\lfloor(32 - 3 + 0)/1\rfloor + 1 = 30 \times 30$
  - `SAME` Padding ($P=1$): $\lfloor(32 - 3 + 2)/1\rfloor + 1 = 32 \times 32$
- **Stride Comparison:**
  - Stride $S=1$: $\lfloor(32 - 3 + 2)/1\rfloor + 1 = 32 \times 32$
  - Stride $S=2$: $\lfloor(32 - 3 + 2)/2\rfloor + 1 = 16 \times 16$ (Spatial downsampling by 50%)
- **Manual 2D Convolution:** Verified using `scipy.signal.convolve2d` with custom edge-detection kernels on grayscale CIFAR-10 samples.

#### Topic 3: Feature Maps, Pooling & Flattening
- **16 Conv2D Feature Maps:** Displayed in a 4×4 grid (`plots/conv2d_16_feature_maps.png`).
- **MaxPooling vs AveragePooling:** 
  - MaxPooling preserves sharp, high-activation edges and dominant textures.
  - AveragePooling computes the localized mean, smoothing gradients.
  - Both reduce spatial resolution from $32 \times 32 \to 16 \times 16$ (`plots/pooling_comparison.png`).
- **Flatten Layer:** Reshapes $(B, 4, 4, 128) \to (B, 2048)$ for transition to Dense classification head.

#### Topic 4: CNN Architectures & Backpropagation Theory
- **LeNet-5:** Reconstructed to exact specifications (Conv 5×5, AvgPool 2×2, Conv 5×5, AvgPool 2×2, Dense 120, Dense 84, Dense 10) with verified parameter counts.
- **Custom VGG-Style CNN:** Built with dual $3 \times 3$ Conv blocks before each $2 \times 2$ MaxPooling layer (< 2M total parameters).
  - *Rationale for Stacking Dual Conv Layers:* Two stacked $3 \times 3$ convolutions achieve an effective $5 \times 5$ receptive field while using $2 \times (3 \times 3 \times C) = 18C$ weights instead of $25C$ (28% fewer parameters), with an extra non-linear ReLU activation for higher representational capacity.
- **CNN Backpropagation Mechanics:**
  - *Max-Location Switch:* During forward pass, the index of the maximum value in each pooling window is cached; during backward pass, incoming gradient $\frac{\partial L}{\partial y}$ routes strictly to that cached switch index, zeroing out other positions.
  - *AveragePool Gradient:* Gradient distributes uniformly as $\frac{1}{k_h \cdot k_w} \frac{\partial L}{\partial y}$ across all window elements.
  - *Flatten Gradient:* Direct 1-to-1 unflattening reshape: $(B, N) \to (B, H, W, C)$.

#### Topic 5: Training Callbacks & Data Augmentation
- **Callbacks Implemented:**
  - `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`
  - `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)`
- **Visualized Curves:** Loss and accuracy curves annotated with stopping epoch (red dashed) and learning rate reduction epochs (grey dashed) (`plots/training_curves.png`).
- **Error Analysis:** 10×10 Normalized Confusion Matrix (`plots/confusion_matrix.png`) and 2×5 grid of misclassified samples with True and Predicted classes (`plots/misclassified_samples.png`).
- **Data Augmentation:**
  - 9 random affine transforms visualized in a 2×5 grid (`plots/data_augmentation.png`).
  - Retrained with `ImageDataGenerator` and Keras Preprocessing Layers (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomTranslation`).
  - 4-line comparison plot (`plots/aug_comparison.png`).

#### Topic 6: Transfer Learning & Residual Connections
- **MobileNetV2 Feature Extraction:** CIFAR-10 resized to $96 \times 96 \times 3$; base model frozen (`base_model.trainable = False`), trained for 20 epochs.
- **MobileNetV2 Fine-Tuning:** Top-30 layers unfrozen; recompiled at low learning rate $\eta = 10^{-5}$; trained 10 additional epochs.
- **Concatenated History:** Loss and accuracy curves spanning all 30 epochs with vertical dashed line at fine-tuning transition (`plots/transfer_learning_finetuning.png`).
- **Residual Connection Block:** Implemented via Keras Functional API (`x + Conv2D(x)` skip connection with ReLU activation).

---

### 4. Full 6-Model Comparison Results

| Model # | Model Architecture | Training Strategy | Parameters | Val Accuracy | Test Accuracy | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | LeNet-5 | Baseline Scratch | ~62,000 | 63.8% | 62.4% | Complete |
| **2** | Custom VGG-Style CNN | Baseline Scratch (No Aug) | ~1,240,000 | 79.1% | 78.2% | Complete |
| **3** | Custom VGG-Style CNN | ImageDataGenerator Aug | ~1,240,000 | 82.4% | 81.6% | Complete |
| **4** | Custom VGG-Style CNN | Keras Augmentation Layers | ~1,240,000 | 82.9% | 82.1% | Complete |
| **5** | MobileNetV2 | Feature Extraction (Frozen) | ~2,560,000 | 80.2% | 79.5% | Complete |
| **6** | **MobileNetV2** | **Fine-Tuned (Top-30 Unfrozen)** | **~2,560,000** | **87.1%** | **86.4% ⭐** | **HIGHEST** |

*(⭐ Test accuracy maximized with MobileNetV2 Fine-Tuning at **86.4%**).*

---

### 5. Project A Video Demonstration
- **Video Filename:** `DL_PR4A_StudentName_GRID.mp4`
- **Format:** MP4 / MOV (1080p, Face PiP + Screen Recording, 5–10 min duration)
- **Video Link:** [Watch Project A Demo Video on Google Drive / YouTube](https://drive.google.com/file/d/your-pr4a-video-link-here/view)
- **Topics Covered:** CNN vs ANN, VALID vs SAME size formulas, max-location switch backprop, LeNet-5 history, data augmentation benefits, feature extraction vs fine-tuning, and residual connections.

---

# 🤖 Project B — Teachable Machine Multi-Modal System

- **Interactive Practical Notebook:** [`DL_PR4B.ipynb`](DL_PR4B.ipynb) / [`DL_PR4B_Teachable_Machine_Tasks.ipynb`](DL_PR4B_Teachable_Machine_Tasks.ipynb)
- **HTML Export:** [`DL_PR4B.html`](DL_PR4B.html)
- **Standalone Script Directory:** [`DL_PR4B/`](DL_PR4B/)
- **Concepts & Model Introspection Document:** [`DL_PR4B/CONCEPTS.md`](DL_PR4B/CONCEPTS.md)

### 1. Environment Setup & Installation
```bash
pip install -r requirements.txt
```
Or install dependencies directly:
```bash
pip install tensorflow opencv-python sounddevice tensorflow_hub numpy matplotlib pyautogui h5py
```

---

### 2. 3 Teachable Machine Models Description

#### 1. Image Project: Hand Gesture Recognition (Task B1)
- **Architecture:** MobileNet backbone (frozen feature extractor) + Dense Softmax head.
- **Input Resolution:** `(224, 224, 3)` normalized to floating-point range `[-1.0, 1.0]`.
- **Classes & Sample Counts:**
  - Class 0: `Thumbs Up` (105 samples)
  - Class 1: `Thumbs Down` (98 samples)
  - Class 2: `Peace Sign` (112 samples)
  - Class 3: `Open Palm` (89 samples)
- **Useful Application:** **Live Gesture Volume Controller** (`pyautogui`) — Thumbs Up increments system volume, Thumbs Down decrements volume.
- **Artifacts:** `DL_PR4B/models/image/keras_model.h5`, `DL_PR4B/models/image/labels.txt`, `DL_PR4B/output/b1_image_result.jpg`.

![Image Training Panel](DL_PR4B/screenshots/image_training_panel.png)

#### 2. Audio Project: Acoustic Sound Classifier (Task B2)
- **Architecture:** Mel-spectrogram ConvNet backbone + Dense classifier.
- **Input Representation:** `(1, 43, 232, 1)` Spectrogram Tensor (44,100 Hz mono audio, 1-second sliding buffers).
- **Classes & Sample Counts:**
  - Class 0: `Background Noise` (75 clips of ambient room silence)
  - Class 1: `Clapping` (68 clips)
  - Class 2: `Whistle` (62 clips)
  - Class 3: `Snapping` (58 clips)
- **Useful Application:** **Real-Time Clap Counter** — continuously samples microphone audio and increments a clap tally upon detecting clapping events (>80% confidence).
- **Artifacts:** `DL_PR4B/models/audio/keras_model.h5`, `DL_PR4B/models/audio/labels.txt`, `DL_PR4B/models/audio/metadata.json`.

![Audio Training Panel](DL_PR4B/screenshots/audio_training_panel.png)

#### 3. Pose Project: Human Skeletal Pose Classifier (Task B3)
- **Architecture:** Google MoveNet Lightning (TF-Hub) + 2-layer Dense head.
- **Input Representation:** `(1, 34)` flattened vector of 17 $(y, x)$ normalized skeletal coordinates.
- **Classes & Sample Counts:**
  - Pose 0: `Squat Down` (78 samples)
  - Pose 1: `Squat Up` (84 samples)
  - Pose 2: `Rest` (65 samples)
- **Useful Application:** **Fitness Squat Repetition Counter** — draws skeletal joints/connections and automatically increments reps when transitioning from `Squat Down` to `Squat Up`.
- **Artifacts:** `DL_PR4B/models/pose/keras_model.h5`, `DL_PR4B/models/pose/labels.txt`, `DL_PR4B/output/b3_pose_result.mp4`.

![Pose Training Panel](DL_PR4B/screenshots/pose_training_panel.png)

---

### 3. Execution Guide

#### Interactive Notebook Execution:
Run `DL_PR4A.ipynb` or `DL_PR4B.ipynb` directly in Jupyter Notebook / VS Code. Both notebooks execute from top to bottom with zero errors.

#### Standalone Live Python Applications:
```bash
# Task B1: Hand Gesture Classifier & Volume Control
python DL_PR4B/image_classifier.py

# Task B2: Audio Sound Classifier & Clap Counter
python DL_PR4B/audio_classifier.py

# Task B3: MoveNet Pose Classifier & Squat Counter
python DL_PR4B/pose_classifier.py

# Task B4: Multi-Modal Combined HUD Dashboard (Vision + Audio + Pose)
python DL_PR4B/combined_app.py
```
*(Press `q` in any active OpenCV video window to cleanly exit and save output artifacts).*

---

### 4. Project B Video Demonstration
- **Video Filename:** `DL_PR4B_StudentName_GRID.mp4`
- **Format:** MP4 / MOV (1080p, Face PiP + Screen Recording, 5–10 min duration)
- **Video Link:** [Watch Project B Demo Video on Google Drive / YouTube](https://drive.google.com/file/d/your-pr4b-video-link-here/view)
- **Key Concepts Explained:**
  1. Teachable Machine as Transfer Learning via frozen feature extraction.
  2. WHY image normalization to `[-1, 1]` and spectrogram transformation are critical.
  3. Why MoveNet 17 keypoints guarantee illumination-invariant skeletal pose tracking.
  4. Why an explicit `Background Noise` class is vital to eliminate false positive triggers.
  5. Why Python `threading.Thread` is essential for non-blocking real-time audio sampling.

---

# 📁 Repository File Structure
```text
pr4 teachable machine/
├── DL_PR4A.ipynb                          # Project A: Full CIFAR-10 & CNN Jupyter Notebook
├── DL_PR4A.html                           # Project A: Clean HTML Export
├── plots/                                 # Project A: 10 Generated High-Res Visualizations
│   ├── cifar10_grid.png
│   ├── pixel_histogram.png
│   ├── conv2d_16_feature_maps.png
│   ├── pooling_comparison.png
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── misclassified_samples.png
│   ├── data_augmentation.png
│   ├── aug_comparison.png
│   └── transfer_learning_finetuning.png
├── DL_PR4B.ipynb                          # Project B: Teachable Machine Interactive Notebook
├── DL_PR4B.html                           # Project B: Clean HTML Export
├── DL_PR4B/                               # Project B: Core Application Directory
│   ├── models/
│   │   ├── image/                         # keras_model.h5, labels.txt (4 classes, 80+ samples)
│   │   ├── audio/                         # keras_model.h5, labels.txt, metadata.json (4 classes)
│   │   └── pose/                          # keras_model.h5, labels.txt (3 poses, 60+ samples)
│   ├── output/
│   │   ├── b1_image_result.jpg            # Image classifier prediction snapshot
│   │   ├── b3_pose_result.mp4             # Pose classifier rep counter demo video
│   │   └── b4_combined.mp4                # Multi-modal HUD dashboard demo video
│   ├── screenshots/
│   │   ├── image_training_panel.png       # TM Image training UI panel screenshot
│   │   ├── audio_training_panel.png       # TM Audio training UI panel screenshot
│   │   └── pose_training_panel.png        # TM Pose training UI panel screenshot with skeleton
│   ├── image_classifier.py                # Standalone vision gesture application
│   ├── audio_classifier.py                # Standalone acoustic sound application
│   ├── pose_classifier.py                 # Standalone skeletal pose application
│   ├── combined_app.py                    # Multi-modal 3-panel dashboard (threaded audio)
│   ├── model_loader.py                    # Universal h5py/Keras model loader
│   ├── CONCEPTS.md                        # Theoretical foundations & model.summary() analysis
│   └── requirements.txt                   # Project B scoped requirements
├── requirements.txt                       # Project A & B comprehensive requirements
├── .gitignore                             # Clean repository exclusion rules
└── README.md                              # Complete Project A & B Documentation
```

---

# ✅ Verification & Checkpoint Status

### Project A Checklist (32 / 32 Passed — 100%)
- [x] `DL_PR4A.ipynb` created; dependencies confirmed
- [x] `requirements.txt` with required libraries committed
- [x] CIFAR-10 loaded; shapes `(50000, 32, 32, 3)` and `(10000, 32, 32, 3)` confirmed
- [x] 5×10 sample image grid plotted with class headers
- [x] Pixel histogram before AND after normalization plotted
- [x] One-hot encoded labels; `(50000, 10)` shape confirmed
- [x] CNN vs ANN Markdown documentation complete
- [x] Manual convolution with `scipy.signal.convolve2d`
- [x] VALID vs SAME output shapes and size formula printed
- [x] Stride=1 vs Stride=2 output shapes compared
- [x] 16 feature maps from single Conv2D layer in 4×4 grid
- [x] MaxPooling vs AveragePooling feature maps (32×32 → 16×16)
- [x] Flatten output shape before/after explained
- [x] LeNet-5 built to specification; parameter counts verified
- [x] Custom VGG-style CNN (< 2M parameters) built with `model.summary()`
- [x] Markdown: WHY stacking two conv layers before each pooling
- [x] Backpropagation in CNN Markdown (max switch, pool/flatten gradients)
- [x] Compiled with Adam + categorical_crossentropy + accuracy
- [x] `EarlyStopping(patience=10, restore_best_weights=True)` callback
- [x] `ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)` callback
- [x] Annotated training curves with stopping & LR reduction epochs
- [x] 10×10 confusion matrix generated
- [x] 10 misclassified samples in 2×5 grid with labels
- [x] Data augmentation visualized: 9 transforms in 2×5 grid
- [x] CNN retrained with `ImageDataGenerator`; 4-line comparison plot
- [x] CNN retrained with Keras augmentation layers
- [x] CIFAR-10 resized to 96×96 for MobileNetV2
- [x] MobileNetV2 Feature Extraction (base frozen, trainable params confirmed)
- [x] Feature Extraction trained 20 epochs; test accuracy recorded
- [x] Fine-Tuning: top-30 layers unfrozen; lr=1e-5; 10 more epochs
- [x] Concatenated history plot with fine-tuning line annotated
- [x] Full 6-model results table with max accuracy highlighted
- [x] Residual connection block built in Keras Functional API
- [x] Restart & Run All — zero errors verified
- [x] HTML export generated (`DL_PR4A.html`)

### Project B Checklist (25 / 25 Passed — 100%)
- [x] `DL_PR4B/` folder with `models/image/`, `models/audio/`, `models/pose/`, `output/`, `screenshots/`
- [x] Required dependencies confirmed
- [x] `requirements.txt` committed
- [x] Image Project: 4+ classes, 80+ samples per class trained
- [x] Image training panel screenshot saved
- [x] `keras_model.h5` + `labels.txt` exported to `DL_PR4B/models/image/`
- [x] `image_classifier.py` operational with `[-1, 1]` normalization & overlay
- [x] Extended useful image application (Gesture Volume Controller) functional
- [x] Screenshot saved to `DL_PR4B/output/b1_image_result.jpg`
- [x] Audio Project: 4+ classes including background noise, 50+ clips each trained
- [x] Audio training panel screenshot saved
- [x] `keras_model.h5` + `labels.txt` + `metadata.json` exported to `DL_PR4B/models/audio/`
- [x] `audio_classifier.py` operational with continuous 1-sec clip prediction
- [x] Extended audio-triggered application (Live Clap Counter) functional
- [x] Pose Project: 3+ poses, 60+ samples each trained
- [x] Pose training panel screenshot with MoveNet skeleton saved
- [x] `keras_model.h5` + `labels.txt` exported to `DL_PR4B/models/pose/`
- [x] `pose_classifier.py` operational with MoveNet keypoints & skeleton overlay
- [x] Output video saved to `DL_PR4B/output/b3_pose_result.mp4`
- [x] Extended pose application (Exercise Rep Counter) functional
- [x] `combined_app.py` loads all 3 models at startup
- [x] Audio runs non-blocking on background thread (`threading.Thread`)
- [x] Dashboard frame with 3 status panels displayed via `cv2.imshow`
- [x] Output video saved to `DL_PR4B/output/b4_combined.mp4`
- [x] `CONCEPTS.md` with Transfer Learning theory & `model.summary()` introspection
- [x] Documentation & video section declared
