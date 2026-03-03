# Multimodal Breast Cancer Tumor Grading

Predict Nottingham histologic grade (Grade 1 / 2 / 3) from the [Duke Breast Cancer MRI](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903) dataset using multimodal learning: DCE-MRI imaging + clinical features.

Two complementary approaches are implemented:

1. **Ovis2 VLM few-shot** -- a vision-language model (AIDC-AI/Ovis2-4B) that reads DCE composite images and clinical metadata through natural language prompts.
2. **Swin-Tiny + Clinical MLP** -- a frozen Swin-Tiny image encoder fused with an MLP over encoded clinical features, trained with 5-fold cross-validated grid search.

## Results Summary

| Method | Crop | Patients | Macro F1 | Balanced Acc | Accuracy |
|--------|------|----------|----------|--------------|----------|
| Ovis2-4B few-shot | Proportional (25% pad) | 60 | 0.383 | 0.388 | 0.433 |
| Ovis2-4B few-shot | None (full-size) | 60 | 0.290 | 0.301 | 0.300 |
| Ovis2-4B few-shot | Fixed 256x256 | 100 | 0.411 | 0.414 | 0.450 |
| **Swin-Tiny + MLP** | **Fixed 256x256** | **100** | **0.667** | **0.673** | **0.700** |

Detailed per-experiment results (JSON, plots, logs) are in `results/`.

## Project Structure

```
.
├── README.md
├── pyproject.toml
├── .gitignore
├── multimodal_tumor_classification/   # Python package
│   ├── __init__.py
│   ├── __main__.py                    # CLI entry point
│   ├── config.py                      # Paths, labels, crop modes, clinical features
│   ├── dicom_utils.py                 # DICOM loading, series identification
│   ├── clinical.py                    # Clinical xlsx parsing, text builder, feature encoding
│   ├── imaging.py                     # DCE composites, cropping, slice sampling
│   ├── dataset.py                     # Dataset building (orchestrates DICOM + clinical)
│   ├── prompts.py                     # Few-shot example selection, prompt formatting
│   ├── ovis2_pipeline.py              # Ovis2 VLM model loading + inference
│   ├── swin_pipeline.py               # Swin-Tiny + MLP training pipeline
│   └── evaluation.py                  # Metrics, plotting, experiment summaries
├── data/
│   ├── Duke-Breast-Cancer-MRI/        # DICOM folders per patient (not included)
│   ├── Annotation_Boxes.xlsx          # Tumor bounding box annotations
│   └── Clinical_and_Other_Features_Full.xlsx
├── output/                            # Runtime artifacts (composites, caches, feature files)
│   ├── ovis2_proportional_crop/       # Composites + results (proportional crop)
│   ├── ovis2_nocrop/                  # Composites + results (no crop)
│   ├── ovis2_fixed256_crop/           # Composites + results (256x256 crop)
│   └── swin_baseline/                 # Swin feature cache + plots
└── results/                           # Final experiment results
    ├── ovis2_proportional_crop/
    ├── ovis2_nocrop/
    ├── ovis2_256crop/
    └── swin_baseline/
```

## Data

This project uses the **Duke Breast Cancer MRI** dataset from The Cancer Imaging Archive (TCIA).

1. Download DICOM images from [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903) and place them under `data/Duke-Breast-Cancer-MRI/` (one subfolder per patient, e.g. `Breast_MRI_024/`).
2. The annotation and clinical spreadsheets (`Annotation_Boxes.xlsx`, `Clinical_and_Other_Features_Full.xlsx`) should also be placed in `data/`.

## Installation

### Prerequisites

- Python 3.10+
- MPS (Apple Silicon) or CUDA GPU recommended for the Ovis2 VLM pipeline. The Swin-Tiny baseline runs on CPU.

### Setup

```bash
# Create and activate a conda environment
conda create -n mml python=3.10 -y
conda activate mml

# Install PyTorch (adjust for your platform -- shown here for macOS MPS)
pip install torch==2.4.0 torchvision==0.19.0

# Install the package in editable mode
pip install -e .
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | Swin-Tiny encoder, tensor operations |
| `transformers` | Ovis2-4B VLM loading |
| `pydicom` | DICOM image reading |
| `scikit-learn` | Metrics, cross-validation, preprocessing |
| `pandas`, `openpyxl` | Clinical spreadsheet parsing |
| `matplotlib` | Plot generation |
| `pillow` | Image I/O |
| `numpy` | Array operations |

All dependencies are declared in `pyproject.toml` and installed automatically with `pip install -e .`.

## Usage

### Ovis2 VLM few-shot classification

```bash
# Proportional bbox crop (25% padding) -- default
python -m multimodal_tumor_classification ovis2 --crop proportional

# No crop (full-size DCE composites)
python -m multimodal_tumor_classification ovis2 --crop none

# Fixed 256x256 crop centered on tumor
python -m multimodal_tumor_classification ovis2 --crop fixed256

# Limit to N patients (useful for testing)
python -m multimodal_tumor_classification ovis2 --crop proportional --num-patients 5
```

The Ovis2 pipeline runs three phases:
1. **DICOM processing** -- loads pre-contrast and 1st post-contrast series, builds RGB composites (R=pre, G=post1, B=subtraction), applies the selected crop, caches PNGs.
2. **Few-shot inference** -- loads Ovis2-4B, builds prompts with clinical metadata and 1 example per grade, classifies 3 slices per patient, majority-votes.
3. **Evaluation** -- prints classification report, macro F1, balanced accuracy, confusion matrix. Saves results JSON to `output/`.

### Swin-Tiny + Clinical MLP baseline

```bash
# Run with defaults (200 epochs, full grid search)
python -m multimodal_tumor_classification swin

# Quick test run
python -m multimodal_tumor_classification swin --epochs 5
```

This runs:
1. **Feature extraction** -- extracts 768-d Swin-Tiny embeddings per patient (average-pooled over 3 slices). Cached to disk.
2. **Clinical encoding** -- binary, one-hot, and standard-scaled numerical features (31-d total).
3. **Grid search** -- 144 hyperparameter combinations x 5-fold stratified CV.
4. **Final evaluation** -- trains best model on 80/20 stratified split. Generates confusion matrix, ROC curves, F1 bar chart, and loss curves.

### CLI installed entry point

After `pip install -e .`, you can also use:

```bash
tumor-classify ovis2 --crop proportional
tumor-classify swin
```

## Pipeline Overview

```
DICOM volumes (pre-contrast + 1st post-contrast)
    |
    v
DCE RGB Composite (R=pre, G=post1, B=subtraction)
    |
    |---> [Ovis2 VLM] Few-shot prompt + clinical text -> Grade prediction
    |
    \---> [Swin-Tiny] Frozen encoder -> 768-d embedding --\
                                                           |---> Fusion MLP -> Grade prediction
         Clinical features -> Encoded 31-d vector -------/
```
