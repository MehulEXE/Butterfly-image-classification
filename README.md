# 🦋 Butterfly Image Classification using Swin Transformer

![Butterfly Image Classification](images/head.png)

A deep learning project for classifying **75 species of butterflies** using the **Swin Transformer** architecture. This project demonstrates how a Vision Transformer approach significantly outperforms CNN-based models like EfficientNet B0 on fine-grained image classification tasks.

**Dataset:** [Butterfly Image Classification – Kaggle](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification)

---

## 📖 Table of Contents

- [Problem Statement](#-problem-statement)
- [Approach Toward the Problem](#-approach-toward-the-problem)
- [Model Architecture – Swin Transformer](#-model-architecture--swin-transformer)
- [Tuned Hyperparameters](#%EF%B8%8F-tuned-hyperparameters)
- [Training Results & Visualizations](#-training-results--visualizations)
- [Why EfficientNet B0 Did Not Perform Well](#-why-efficientnet-b0-did-not-perform-well)
- [Conclusion](#-conclusion)

---

## 🎯 Problem Statement

Given an image of a butterfly, the goal is to correctly identify which of the **75 distinct species** it belongs to. This is a **fine-grained image classification** problem — the differences between species can be extremely subtle (wing patterns, colour gradations, edge markings), making it significantly harder than standard image classification.

---

## 🚀 Approach Toward the Problem

### Step 1 — Exploratory Data Analysis (EDA)

- Loaded the training CSV (`Training_set.csv`) and mapped each filename to its full image path.
- Visualised the distribution of all 75 butterfly species to check for class imbalance.
- The dataset is **reasonably balanced**, with most classes containing 75–130 images.

![Bar Graph of Butterfly Categories](images/bargraph%20of%20categries%20of%20buterfly.png)

### Step 2 — Data Preprocessing & Augmentation

| Step | Training Set | Validation / Test Set |
|---|---|---|
| **Resize** | `Resize(256)` | `Resize(256)` |
| **Crop** | `CenterCrop(224)` | `CenterCrop(224)` |
| **Horizontal Flip** | ✅ `RandomHorizontalFlip` | ❌ |
| **Vertical Flip** | ✅ `RandomVerticalFlip` | ❌ |
| **Color Jitter** | ✅ `ColorJitter` (p=0.25) | ❌ |
| **Random Erasing** | ✅ `RandomErasing` (p=0.1) | ❌ |
| **Normalize** | ImageNet mean & std | ImageNet mean & std |
| **ToTensor** | ✅ | ✅ |

- **Label Encoding:** Labels were integer-encoded using `sklearn.preprocessing.LabelEncoder` (75 classes → integers 0–74).
- **Data Split:** 80/20 train-validation split with **stratified sampling** (`stratify=train['label']`) to maintain class distribution.
- A custom **`ButterflyDataset`** PyTorch `Dataset` class was built to load images on-the-fly with transforms.

### Step 3 — Model Selection: Why Swin Transformer?

We initially experimented with **EfficientNet B0** (a CNN-based architecture), but it failed to learn meaningful features for this task (see [detailed analysis below](#-why-efficientnet-b0-did-not-perform-well)). We then moved to the **Swin Transformer**, which is specifically designed for vision tasks and excels at capturing both local and global feature relationships through its **shifted window self-attention** mechanism.

### Step 4 — Training Strategy

1. **Fine-tuning a pretrained model:** Loaded `swin_tiny_patch4_window7_224` pretrained on ImageNet from the `timm` library, then fine-tuned all layers end-to-end on the butterfly dataset.
2. **Optimizer:** Used `AdamW` with a low learning rate (`1e-4`) and weight decay (`0.05`) for stable, regularised fine-tuning.
3. **Learning Rate Scheduling:** Applied `CosineAnnealingLR` to smoothly decay the learning rate across epochs.
4. **Best Model Checkpointing:** Saved the model weights (`best_swin.pth`) whenever validation accuracy improved.

---

## 🏗️ Model Architecture – Swin Transformer

```
Model: swin_tiny_patch4_window7_224 (from timm library)
├── Patch Embedding  (patch_size=4, embed_dim=96)
├── Stage 1: 2× Swin Transformer Blocks  (window_size=7, heads=3)
├── Stage 2: 2× Swin Transformer Blocks  (window_size=7, heads=6)
├── Stage 3: 6× Swin Transformer Blocks  (window_size=7, heads=12)
├── Stage 4: 2× Swin Transformer Blocks  (window_size=7, heads=24)
├── LayerNorm + AdaptiveAvgPool1d
└── Linear(768 → 75)  ← Classification Head (75 butterfly classes)
```

**Key design choices of Swin Transformer:**
- **Shifted Window Attention:** Computes self-attention within local windows and shifts them across layers to capture cross-window interactions — more efficient than global attention.
- **Hierarchical Feature Maps:** Produces multi-scale feature representations (like a CNN), making it excellent for fine-grained detail recognition (wing patterns, spots, edge colourings).
- **Pretrained on ImageNet:** Starting from ImageNet weights provides a strong initialisation for low-level and mid-level features.

---

## ⚙️ Tuned Hyperparameters

The following hyperparameters were carefully selected for the Swin Transformer training:

| Hyperparameter | Value | Rationale |
|---|---|---|
| **Base Model** | `swin_tiny_patch4_window7_224` | Smallest Swin variant — good balance of speed and accuracy |
| **Input Resolution** | `224 × 224` | Standard resolution for Swin Tiny (patch size 4 → 56×56 patches) |
| **Pretrained Weights** | ImageNet | Strong feature initialisation for transfer learning |
| **Optimizer** | `AdamW` | Decoupled weight decay — better generalisation than vanilla Adam |
| **Learning Rate** | `1e-4` | Low LR for stable fine-tuning of pretrained weights |
| **Weight Decay** | `0.05` | Regularisation to prevent overfitting on the small dataset |
| **LR Scheduler** | `CosineAnnealingLR` (T_max=10) | Smooth LR decay — avoids abrupt drops that can destabilise training |
| **Loss Function** | `CrossEntropyLoss` | Standard choice for multi-class classification |
| **Batch Size (Train)** | `128` | Larger batch for stable gradient estimation |
| **Batch Size (Val/Test)** | `32` | Smaller batch to conserve memory during inference |
| **Epochs** | `10` | Sufficient — the model converges quickly thanks to pretrained weights |
| **Num Workers** | `2` | Parallel data loading for faster training |
| **Checkpoint Strategy** | Save on best `val_acc` | Ensures we keep the best-performing model |

---

## 📊 Training Results & Visualizations

### Training & Validation Curves

The Swin Transformer converged rapidly, reaching high accuracy within just 10 epochs:

![Training vs Validation Loss and Accuracy](images/training%20graph.png)

**Key observations:**
- **Training Loss** drops steeply from ~2.4 to near 0, showing the model learns butterfly features effectively.
- **Validation Loss** stabilises around 0.2–0.3, indicating good generalisation without severe overfitting.
- **Validation Accuracy** climbs to ~**95%** and plateaus, while training accuracy reaches ~99%.

### Pretrained vs Fine-Tuned Swin Transformer

To validate the importance of fine-tuning, we compared:
- **Pretrained only** (no fine-tuning, just ImageNet weights with a new head) → **1.62%** accuracy
- **Fine-Tuned** (all layers trained on butterfly data) → **94.85%** accuracy

![Pretrained vs Fine-tuned Swin Transformer](images/comparision%20between%20swin%20fine-tuned%20and%20pretrained.png)

> **Takeaway:** Fine-tuning is absolutely essential. The pretrained model alone performs at random-chance level (1.62% ≈ 1/75 classes) because its classification head has never seen butterfly labels.

### Predictions on Validation Set

The model correctly identifies species with high confidence, even for visually similar butterflies:

![Validation Predictions — Actual vs Predicted](images/validation%20predicted%20result.png)

### Predictions on Test Set

Inference on unseen test images shows the model generalises well to new data:

![Test Set Predictions](images/predicted%20result.png)

---

## 📉 Why EfficientNet B0 Did Not Perform Well

Before adopting the Swin Transformer, we experimented with **EfficientNet B0** (TensorFlow/Keras). Despite being a popular and efficient CNN architecture, it performed **extremely poorly** on this task — achieving only **~2% validation accuracy** across both variants tested.

![EfficientNet B0 Model Accuracy Comparison](images/comparision%20of%20efficientent%20B0%20models.png)

### What Went Wrong — A Detailed Analysis

#### 1. 🧊 Frozen Base Model (No Fine-Tuning)
The EfficientNet base was completely **frozen** (`trainable = False`). Only the classification head (a single `Dense` layer) was trained. This means:
- The convolutional features remain fixed to what they learned on ImageNet (dogs, cats, cars, etc.).
- These generic features are insufficient for distinguishing **fine-grained butterfly wing patterns** — the model needs to learn domain-specific texture and colour features.
- In contrast, the Swin Transformer was fine-tuned **end-to-end**, allowing all layers to adapt to butterfly-specific features.

#### 2. 🧠 Insufficient Classification Head
The EfficientNet approach used a minimal head:
```
GlobalAveragePooling2D → Dense(75, softmax)
```
With frozen features, this single linear layer simply cannot learn the complex decision boundaries needed for 75 classes of butterflies. Even adding `Dropout(0.3)` in the second variant didn't help — the bottleneck was the frozen features, not overfitting.

#### 3. ⚠️ Weak Data Augmentation Pipeline
The EfficientNet notebook used basic augmentations:
- `RandomFlip("horizontal")`, `RandomRotation(0.1)`, `RandomZoom(0.1)`

Compared to the Swin Transformer pipeline which included:
- `RandomHorizontalFlip` + `RandomVerticalFlip` + `ColorJitter` + `RandomErasing`

The stronger augmentation in the Swin pipeline helped the model generalise better.

#### 4. 📊 No Learning Rate Scheduling
The EfficientNet model used a flat `Adam` optimizer. While `ReduceLROnPlateau` was added as a callback, with `EarlyStopping(patience=5)` the training likely stopped before the LR reduction could take effect. The Swin Transformer used `CosineAnnealingLR`, providing a smooth and predictable learning rate decay.

#### 5. 🔄 CNN vs Transformer for Fine-Grained Classification
CNNs like EfficientNet rely on **local receptive fields** — they build up features from small patches. For fine-grained tasks where **global context** matters (e.g., the overall wing shape + specific spot patterns), Transformers have a natural advantage because self-attention captures **long-range dependencies** across the entire image.

### Summary Comparison

| Aspect | EfficientNet B0 | Swin Transformer |
|---|---|---|
| **Framework** | TensorFlow / Keras | PyTorch / timm |
| **Fine-Tuning** | ❌ Frozen base | ✅ End-to-end |
| **Optimizer** | Adam | AdamW (weight decay=0.05) |
| **LR Schedule** | Flat (ReduceLROnPlateau) | CosineAnnealingLR |
| **Augmentation** | Basic (flip, rotate, zoom) | Heavy (flip, jitter, erasing) |
| **Validation Accuracy** | **~2%** | **~94.85%** |
| **Verdict** | ❌ Failed | ✅ Excellent |

---

## 🔍 Known Limitations & Areas of Improvement

While this project achieves strong results, there are several areas I recognise for future improvement:

### 1. ⚖️ Fairer Model Comparison
The EfficientNet B0 base was **completely frozen**, which made the comparison against a fully fine-tuned Swin Transformer unfair. A more rigorous experiment would be to:
- Fine-tune EfficientNet B0 end-to-end with the same augmentation pipeline.
- Perform **partial unfreezing** (e.g., unfreeze the last 20–50 layers) to find the sweet spot.
- Use the same optimizer (AdamW) and scheduler (CosineAnnealing) for both models.

### 2. 📊 Deeper Evaluation Metrics
Currently, the primary metric is **accuracy**. For a 75-class problem, additional metrics would give a more complete picture:
- **Confusion Matrix** — to identify which species are commonly confused with each other.
- **Per-class Precision, Recall, and F1-Score** — to detect underperforming classes.
- **Top-5 Accuracy** — useful since some species are visually almost identical.

### 3. 🚀 Model Deployment
The model currently runs only in a Jupyter Notebook. Future work includes:
- Building a **Gradio** or **Streamlit** web app for real-time butterfly classification.
- Deploying on **Hugging Face Spaces** for a shareable demo link.
- Exporting the model to **ONNX** format for optimised inference.

### 4. 🔬 Experiment Tracking
No experiment tracking tool (e.g., **MLflow**, **Weights & Biases**) was used. Adding one would help:
- Log hyperparameters, metrics, and artefacts systematically.
- Compare runs side-by-side when testing different configurations.
- Reproduce results reliably.

### 5. 🧪 Additional Experiments to Try
- **Larger Swin variants** (`swin_small`, `swin_base`) — potentially higher accuracy but more compute.
- **Label Smoothing** — the `timm` library provides `LabelSmoothingCrossEntropy`, which was imported but not used in the final training.
- **MixUp / CutMix augmentation** — proven to improve generalisation on fine-grained tasks.
- **Test-Time Augmentation (TTA)** — average predictions over multiple augmented versions of the same image at inference.
- **Learning Rate Warm-up** — gradually increase LR for the first few epochs to stabilise early training.

### 6. 🗂️ Code Organisation
- Refactor notebook code into modular `.py` scripts (`dataset.py`, `model.py`, `train.py`, `evaluate.py`).
- Add **command-line arguments** for hyperparameters using `argparse`.
- Write **unit tests** for the dataset class and data transforms.

---

## 🏁 Conclusion

- The **Swin Transformer** (`swin_tiny_patch4_window7_224`) achieved **94.85% validation accuracy** on the butterfly classification task, demonstrating that Vision Transformers with proper fine-tuning are highly effective for fine-grained image recognition.
- **EfficientNet B0** failed catastrophically (~2% accuracy) because its base layers were frozen — proving that transfer learning without fine-tuning is inadequate for domain-specific, fine-grained tasks.
- **Key lesson:** For fine-grained classification, always fine-tune the entire model (or at least the later layers) rather than only training a classification head on top of frozen features.

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.10+
- CUDA-enabled GPU (recommended for training)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/butterfly-image-classification.git
cd butterfly-image-classification

# Install dependencies
pip install -r requirements.txt
```

### Run the Notebooks
Open the `.ipynb` files in Jupyter Notebook or Kaggle:
- `swin-transformer.ipynb` — Swin Transformer (primary model)
- `efficientnet-b0-7 .ipynb` — EfficientNet B0 (baseline comparison)

---

> 📓 *For full code and execution details, please refer to the provided `.ipynb` notebooks.*
