# BeyondWords — ASL Alphabet Recognition

A deep learning pipeline that classifies American Sign Language (ASL) hand signs from images. The model identifies which of 29 ASL classes is being signed — the 26 letters A through Z plus `del`, `nothing`, and `space` — and is designed to accept personally-submitted photos.

---

## Team

**BeyondWords**

---

## Project Overview

This project trains an image classification model on the [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) using transfer learning on MobileNetV2. The full pipeline covers exploratory data analysis, SQL-backed metadata storage, dataset cleaning, PyTorch DataLoader construction, model training and hyperparameter experimentation, and final evaluation.

**Final test accuracy: 90.69%** — achieved by freezing MobileNetV2's pretrained ImageNet feature extractor and fine-tuning only the 29-class classifier head.

---

## Repository Structure

```
.
├── data/
│   ├── asl_alphabet_train/     # Training images (not tracked by git)
│   ├── asl_alphabet_test/      # Kaggle test images (not tracked by git)
│   └── cleaned/                # Validated images organized by class
├── models/
│   └── baseline_mobilenetv2.pth  # Saved model checkpoint
├── eda.ipynb              # Exploratory data analysis
├── sprint_2.ipynb              # SQL pipeline, cleaning, and preprocessing
├── dataset_dataloader_verification.ipynb  # Custom Dataset and DataLoader
├── sprint_3.ipynb              # Model architecture, training, and evaluation
├── asl_metadata.db             # SQLite metadata database (not tracked by git)
└── README.md
```

> `data/` and `asl_metadata.db` are listed in `.gitignore` — download the dataset separately (see Setup below). The selected Sprint 4 checkpoint is tracked at `models/baseline_mobilenetv2.pth`.

---

## Dataset

- **Source:** [Kaggle — ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Classes:** 29 (A–Z, `del`, `nothing`, `space`)
- **Total images:** 87,000 (3,000 per class)
- **Image size:** 200 × 200 px (original); resized to 224 × 224 for the model
- **Split:** stratified 70 / 15 / 15 (train / val / test)

| Split | Images |
|-------|--------|
| Train | 60,900 |
| Val   | 13,050 |
| Test  | 13,050 |

Each split has exactly 450 test images and 2,100 training images per class, ensuring balanced class representation throughout.

---

## Pipeline

### Sprint 1 — Exploratory Data Analysis (`eda.ipynb`)

Initial exploration of the dataset: class distribution, image size verification, and sample visualization.

### Sprint 2 — Storage, Cleaning, and Preprocessing (`sprint_2.ipynb`)

Builds the SQL metadata database and prepares data for the model pipeline.

- Scans training folders and records relative image paths, class labels, and image dimensions in a SQLite database (`asl_metadata.db`)
- Validates each image with PIL to catch corrupt or unloadable files
- Assigns stratified 70/15/15 train/val/test splits and stores the `split` column in SQL
- Exposes three query-ready DataFrames (`train_df`, `val_df`, `test_df`) for downstream use

All 87,000 images passed validation. Image paths are stored as relative paths so the notebook runs consistently across team members' machines.

### Dataset & DataLoader (`dataset_dataloader_verification.ipynb`)

Custom PyTorch `Dataset` class and `DataLoader` configuration (Nick).

- Loads split metadata from SQL
- Applies normalization and resizing transforms (224 × 224)
- Creates `train_loader`, `val_loader`, and `test_loader` with batch size 32
- Verifies zero overlap between splits

### Sprint 3 — Model Architecture, Training, and Evaluation (`sprint_3.ipynb`)

Transfer learning on MobileNetV2 with hyperparameter experimentation.

- Loads pretrained MobileNetV2 (ImageNet weights)
- Replaces the 1,000-class classifier head with a new `Linear(1280 → 29)` layer
- Freezes the feature extractor (2,223,872 frozen params); trains only the classifier head (37,149 trainable params)
- Runs baseline training (5 epochs) and two optimization experiments
- Evaluates the final model on the held-out test set

---

## Model Architecture

| Component | Detail |
|-----------|--------|
| Base model | MobileNetV2 (pretrained, ImageNet weights) |
| Classifier head | `Linear(1280, 29)` |
| Feature extractor | Frozen during training |
| Trainable parameters | 37,149 |
| Frozen parameters | 2,223,872 |
| Loss function | CrossEntropyLoss |
| Optimizer | Adam (lr = 0.001) |
| Batch size | 32 |
| Input shape | `[32, 3, 224, 224]` |

---

## Experiments

| Experiment | LR | Epochs | Val Accuracy | Notes |
|---|---|---|---|---|
| **Baseline** | 0.001 | 5 | **90.75%** | Best overall; peaked at epoch 4 |
| 3-Epochs | 0.001 | 3 | 88.50% | One extra epoch over the 2-epoch baseline improved performance |
| Experimental Baseline | 0.001 | 2 | 87.06% | Useful fast-iteration reference |
| Lower LR | 0.0001 | 2 | 73.25% | Too slow to converge under a short training budget |

The 5-epoch baseline was selected as the final model. The lower learning rate (0.0001) significantly hurt convergence when only the classifier head was trainable, suggesting the frozen-feature setup benefits from a standard learning rate.

---

## Results

**Test accuracy: 90.69%**

The model generalizes well — test accuracy closely tracks best validation accuracy (91.03%), indicating no significant overfitting.

**Strong classes** (F1 ≥ 0.96): `C`, `F`, `L`, `O`, `P`, `Q`

**Weaker classes** (most confusion): `R`/`U`, `M`/`N`, `S`/`X`, `E`/`I` — all visually similar hand shapes where fine-grained finger positioning is the key distinguishing feature.

---

## Setup

**1. Clone the repository**

```bash
git clone <your-repo-url>
cd <repo-name>
```

**2. Install dependencies**

```bash
pip install torch torchvision pandas pillow scikit-learn matplotlib seaborn nbformat
```

**3. Download the dataset**

Download the [ASL Alphabet dataset from Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) and place it at:

```
data/asl_alphabet_train/asl_alphabet_train/   ← 29 class folders, 3,000 images each
data/asl_alphabet_test/                        ← Kaggle test images
```

**4. Run the notebooks in order**

```
eda.ipynb                          # EDA (optional for reproduction)
sprint_2.ipynb                          # Builds asl_metadata.db
dataset_dataloader_verification.ipynb   # Creates DataLoaders (run via %run in sprint_3)
sprint_3.ipynb                          # Training and evaluation
```

> Sprint 3 runs `dataset_dataloader_verification.ipynb` automatically via `%run`. You do not need to open it separately.

**5. Load the trained model (optional)**

To skip retraining and run only evaluation, download `baseline_mobilenetv2.pth` and place it in `models/`. The evaluation cells in `sprint_3.ipynb` will detect it automatically via `checkpoint_available`.

**6. Run the Sprint 4 web demo**

```bash
python3 sprint4_app/server.py
```

Then open:

```text
http://127.0.0.1:8000
```

The local app serves the upload interface and backend inference endpoint from one process.

```text
POST /predict
Form field: image=<jpg-or-png-file>
Response: predicted_class, confidence, top_predictions
```

---

## Notes for Retraining

The full training loop is gated behind `RUN_FULL_TRAINING = False` in `sprint_3.ipynb` to prevent accidental retraining. Set it to `True` only when intentionally running the baseline training. Experiments are similarly gated (`RUN_EXPERIMENTAL_BASELINE`, `RUN_LOWER_LR`, `RUN_THREE_EPOCHS`). Training on CPU takes several hours for 5 epochs over 60,900 images; a GPU is recommended.

---

## Next Steps

- Fine-tune the last MobileNetV2 inverted residual block (unfreeze partial feature extractor) to improve accuracy on confused pairs
- Add targeted data augmentation (hand rotation, zoom, lighting variation) to reduce sensitivity to subtle pose differences
- Investigate per-class augmentation for the weakest-performing signs (E, I, M, S)
- Improve the Sprint 4 interface with camera capture and clearer examples of strong upload framing
