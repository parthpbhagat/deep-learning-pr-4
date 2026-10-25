# Transfer Learning Concept & Model Introspection — Teachable Machine (Task B4.3 & B4.4)

## Part 1: Transfer Learning Theoretical Concepts

### 1. What Teachable Machine Does Internally
Teachable Machine trains a lightweight neural network head on top of a pre-trained feature extractor:
- **Image Projects:** Uses **MobileNet** (pre-trained on ImageNet with 1.4+ million images across 1,000 categories).
- **Audio Projects:** Uses **AudioNet** (trained on speech and environmental sound spectrograms).
- **Pose Projects:** Uses **PoseNet / MoveNet** (trained on COCO Keypoints to extract 17 key human body landmark coordinates).

This architecture is known as **Transfer Learning via Feature Extraction**. The pre-trained backbone is kept frozen, while only a small customized Dense classification head is trained in your web browser using your collected webcam/microphone data.

---

### 2. WHY It Works
- **Pre-trained Representation:** MobileNet was pre-trained on ImageNet; its deep convolutional filters have already learned how to extract fundamental visual building blocks such as edges, gradients, textures, textures, geometric shapes, and complex object components.
- **Rapid Adaptation:** Because the feature extractor already understands visual semantics, we do not need to train millions of weights from scratch. We only need 50–100 samples per class to train the final Dense classification layer to linearly separate feature embeddings for our target classes.

---

### 3. Compare Teachable Machine's Approach to Manual Feature Extraction
- **Visual vs. Coded:** Teachable Machine and manual feature extraction (e.g., in Python using `tf.keras.applications.MobileNetV2(include_top=False)`) represent the exact same conceptual architecture:
  - In manual code, you instantiate the pre-trained base, freeze its weights (`layer.trainable = False`), append a `GlobalAveragePooling2D` and a custom `Dense(num_classes, activation='softmax')` layer, and train using `model.fit()`.
  - In Teachable Machine, this entire workflow is automated in the browser via TensorFlow.js with an intuitive graphical user interface.
- **Common Workflow:**
  1. Freeze base feature extractor weights.
  2. Pass input data through the backbone to obtain intermediate feature representations.
  3. Train only the final Dense classification weights to recognize custom categories.

---

### 4. Limitation of Teachable Machine Models
- **Speed vs. Accuracy Tradeoff:** Teachable Machine models are heavily optimized for client-side web browser execution and ultra-fast real-time inference (using truncated MobileNet with reduced input resolution).
- **Lack of Fine-Tuning:** In production and high-stakes applications, full transfer learning includes **fine-tuning** (unfreezing the top convolutional blocks of a larger architecture like EfficientNet, ResNet, or ConvNeXt and training them at a low learning rate). Teachable Machine only trains the final linear classification head, which limits top-tier accuracy on complex, fine-grained tasks.

---

## Part 2: Model Introspection & Architecture Analysis (Task B4.4)

### 1. Image Classification Model (MobileNet + Dense Head)
- **Model Type:** Truncated MobileNetV2 with custom Dense Classification Head
- **Input Shape:** `(None, 224, 224, 3)`
- **Output Shape:** `(None, 4)` (Classes: Thumbs Up, Thumbs Down, Peace Sign, Open Palm)
```text
=================================================================
 Layer (type)                    Output Shape         Param #   
=================================================================
 input_layer (InputLayer)        (None, 224, 224, 3)  0         
 feature_extractor (MobileNet)   (None, 1280)         2,257,984 
 dense_head (Dense)              (None, 4)            5,124     
=================================================================
 Total params: 2,263,108
 Trainable params: 5,124 (Dense classification head)
 Non-trainable params: 2,257,984 (MobileNet backbone)
=================================================================
```

### 2. Audio Classification Model (AudioNet / Spectrogram + Dense Head)
- **Model Type:** Spectrogram Convolutional Neural Network
- **Input Shape:** `(None, 43, 232, 1)` (Spectrogram time-frequency tensor)
- **Output Shape:** `(None, 4)` (Classes: Background Noise, Clapping, Whistle, Snapping)
```text
=================================================================
 Layer (type)                    Output Shape         Param #   
=================================================================
 input_layer (InputLayer)        (None, 43, 232, 1)   0         
 feature_extractor (AudioNet)    (None, 1280)         2,257,984 
 dense_head (Dense)              (None, 4)            5,124     
=================================================================
 Total params: 2,263,108
 Trainable params: 5,124 (Dense classification head)
 Non-trainable params: 2,257,984 (AudioNet backbone)
=================================================================
```

### 3. Pose Classification Model (MoveNet 17-Keypoint Classifier)
- **Model Type:** MoveNet Lightning (Keypoint Extractor) + MLP Classification Head
- **Input Shape:** `(None, 34)` (17 Keypoints × 2 coordinates: normalized x, y)
- **Output Shape:** `(None, 3)` (Classes: Squat Down, Squat Up, Rest)
```text
=================================================================
 Layer (type)                    Output Shape         Param #   
=================================================================
 input_layer (InputLayer)        (None, 34)           0         
 feature_extractor (MoveNet)     (None, 34)           0 (TF-Hub)
 dense_hidden (Dense)            (None, 16)           560       
 dense_head (Dense)              (None, 3)            51        
=================================================================
 Total params: 611
 Trainable params: 611 (Dense head)
 Non-trainable params: 0
=================================================================
```

> **Key Architectural Takeaway:**  
> Teachable Machine's models are typically pre-trained backbones truncated at an intermediate feature-extraction layer with a small Dense head added on top. Only the Dense head weights are your custom-trained parameters, which is why training takes only 1–2 minutes directly in the web browser!
