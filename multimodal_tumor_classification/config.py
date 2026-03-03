"""Shared configuration: paths, label maps, crop modes, clinical feature definitions."""

import os
import torch

# =============================================================================
# PATHS — resolved relative to the project root (one level above this package)
# =============================================================================
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_PACKAGE_DIR)

DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "Duke-Breast-Cancer-MRI")
ANNOTATIONS_FILE = os.path.join(PROJECT_ROOT, "data", "Annotation_Boxes.xlsx")
CLINICAL_FILE = os.path.join(PROJECT_ROOT, "data", "Clinical_and_Other_Features_Full.xlsx")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# =============================================================================
# LABEL DEFINITIONS
# =============================================================================
LABEL_COLUMN = "Nottingham_Grade_v2"
LABEL_MAP = {1: 0, 2: 1, 3: 2}
LABEL_NAMES = ["Grade 1", "Grade 2", "Grade 3"]
NUM_CLASSES = 3

# =============================================================================
# PIPELINE DEFAULTS
# =============================================================================
SLICES_PER_PATIENT = 3
RANDOM_SEED = 42

# =============================================================================
# DEVICE
# =============================================================================
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# =============================================================================
# CROP MODES
# =============================================================================
CROP_MODES = {
    "proportional": {"padding_ratio": 0.25},
    "none": {},
    "fixed256": {"size": 256},
}

# =============================================================================
# CLINICAL FEATURE DEFINITIONS
# =============================================================================
_YESNO = {0: "No", 1: "Yes"}
_MENOPAUSE_MAP = {0: "Pre-menopausal", 1: "Post-menopausal", 2: "N/A"}
_ER_MAP = {0: "Negative", 1: "Positive"}
_PR_MAP = {0: "Negative", 1: "Positive"}
_HER2_MAP = {0: "Negative", 1: "Positive", 2: "Borderline"}
_SURGERY_TYPE_MAP = {0: "BCS", 1: "Mastectomy"}

# (column_name, display_label, decode_map_or_None)
CLINICAL_FEATURES_TEXT = [
    ("Menopause (at diagnosis)", "Menopause", _MENOPAUSE_MAP),
    ("ER", "ER", _ER_MAP),
    ("PR", "PR", _PR_MAP),
    ("Surgery", "Surgery", _YESNO),
    ("Adjuvant Radiation Therapy", "Adjuvant Radiation Therapy", _YESNO),
    ("Adjuvant Endocrine Therapy Medications", "Adjuvant Endocrine Therapy", _YESNO),
    ("Pec/Chest Involvement", "Pec/Chest Involvement", _YESNO),
    ("HER2", "HER2", _HER2_MAP),
    ("Multicentric/Multifocal", "Multicentric/Multifocal", _YESNO),
    ("Lymphadenopathy or Suspicious Nodes", "Lymphadenopathy", _YESNO),
    ("Definitive Surgery Type", "Definitive Surgery Type", _SURGERY_TYPE_MAP),
    ("Neoadjuvant Chemotherapy", "Neoadjuvant Chemo", _YESNO),
    ("Adjuvant Chemotherapy", "Adjuvant Chemo", _YESNO),
    ("Neoadjuvant Anti-Her2 Neu Therapy", "Neoadjuvant Anti-HER2", _YESNO),
    ("Adjuvant Anti-Her2 Neu Therapy", "Adjuvant Anti-HER2", _YESNO),
    ("Metastatic at Presentation (Outside of Lymph Nodes)", "Metastatic at Presentation", _YESNO),
    ("Contralateral Breast Involvement", "Contralateral Breast Involvement", _YESNO),
    ("Staging(Metastasis)#(Mx -replaced by -1)[M]", "Staging M", None),
    ("Skin/Nipple Invovlement", "Skin/Nipple Involvement", _YESNO),
    ("Neoadjuvant Radiation Therapy", "Neoadjuvant Radiation Therapy", _YESNO),
    ("Recurrence event(s)", "Recurrence", _YESNO),
    ("Known Ovarian Status", "Known Ovarian Status", _YESNO),
    ("Therapeutic or Prophylactic Oophorectomy as part of Endocrine Therapy",
     "Oophorectomy for Endocrine Therapy", _YESNO),
    ("Neoadjuvant Endocrine Therapy Medications", "Neoadjuvant Endocrine Therapy", _YESNO),
    ("Staging(Nodes)#(Nx replaced by -1)[N]", "Staging N", None),
]

# For the Swin MLP encoder
BINARY_FEATURES = [
    "ER", "PR", "Surgery", "Adjuvant Radiation Therapy",
    "Adjuvant Endocrine Therapy Medications", "Pec/Chest Involvement",
    "Multicentric/Multifocal", "Lymphadenopathy or Suspicious Nodes",
    "Neoadjuvant Chemotherapy", "Adjuvant Chemotherapy",
    "Neoadjuvant Anti-Her2 Neu Therapy", "Adjuvant Anti-Her2 Neu Therapy",
    "Metastatic at Presentation (Outside of Lymph Nodes)",
    "Contralateral Breast Involvement", "Skin/Nipple Invovlement",
    "Neoadjuvant Radiation Therapy", "Recurrence event(s)",
    "Known Ovarian Status",
    "Therapeutic or Prophylactic Oophorectomy as part of Endocrine Therapy",
    "Neoadjuvant Endocrine Therapy Medications",
]

CATEGORICAL_FEATURES = {
    "Menopause (at diagnosis)": [0, 1, 2],
    "HER2": [0, 1, 2],
    "Definitive Surgery Type": [0, 1, "NP"],
}

NUMERICAL_FEATURES = [
    "Staging(Metastasis)#(Mx -replaced by -1)[M]",
    "Staging(Nodes)#(Nx replaced by -1)[N]",
]

# =============================================================================
# SWIN TRAINING DEFAULTS
# =============================================================================
SWIN_NUM_EPOCHS = 200
SWIN_NUM_CV_FOLDS = 5
SWIN_PATIENCE = 20
