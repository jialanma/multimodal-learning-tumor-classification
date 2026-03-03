"""Dataset building: combines DICOM loading, clinical data, and imaging into patient records."""

import os
import glob
import pandas as pd
from collections import Counter

from .config import (
    DATA_ROOT, ANNOTATIONS_FILE, LABEL_MAP, LABEL_COLUMN,
    SLICES_PER_PATIENT, CROP_MODES,
)
from .clinical import load_clinical_dataframe, build_clinical_text
from .dicom_utils import load_dicom_series, adjust_annotation_slices, identify_series
from .imaging import make_dce_composite, apply_crop, sample_slices_around_tumor


def process_patient(pid: str, annot: pd.Series, save_dir: str,
                    crop_mode: str = "proportional",
                    crop_kwargs: dict | None = None) -> list:
    """
    Full pipeline for one patient:
      Load DICOM series -> align annotations -> sample slices -> DCE composite -> crop -> save

    Returns list of saved PNG paths, or [] on failure.
    """
    if crop_kwargs is None:
        crop_kwargs = dict(CROP_MODES.get(crop_mode, {}))

    patient_dir = os.path.join(DATA_ROOT, pid)
    series = identify_series(patient_dir)
    if not series:
        print(f"  {pid}: cannot identify series")
        return []

    vol_pre, asc_z_pre, _ = load_dicom_series(series['pre'])
    vol_post1, _, _ = load_dicom_series(series['post1'])

    if vol_pre is None or vol_post1 is None:
        print(f"  {pid}: failed to load volumes")
        return []

    depth = min(vol_pre.shape[0], vol_post1.shape[0])
    vol_pre = vol_pre[:depth]
    vol_post1 = vol_post1[:depth]
    slice_shape = vol_pre.shape[1:]

    start_idx, end_idx = adjust_annotation_slices(annot, depth, asc_z_pre)
    indices = sample_slices_around_tumor(start_idx, end_idx, SLICES_PER_PATIENT)

    os.makedirs(save_dir, exist_ok=True)
    paths = []
    for idx in indices:
        composite = make_dce_composite(vol_pre[idx], vol_post1[idx])
        composite = apply_crop(composite, annot, slice_shape, crop_mode, **crop_kwargs)
        path = os.path.join(save_dir, f"slice_{idx:03d}_dce.png")
        composite.save(path)
        paths.append(path)

    if not asc_z_pre:
        print(f"  {pid}: filename order is descending Z (noted)")
    return paths


def build_dataset(crop_mode: str = "proportional",
                  output_dir: str | None = None,
                  num_patients: int | None = None) -> list:
    """
    Build the full patient dataset.

    Returns:
        [{pid, label, clinical_text, slice_paths}, ...]
    """
    clinical_df = load_clinical_dataframe()
    annot_df = pd.read_excel(ANNOTATIONS_FILE)

    crop_kwargs = CROP_MODES.get(crop_mode, {})
    if output_dir is None:
        from .config import DEFAULT_OUTPUT_DIR
        output_dir = os.path.join(DEFAULT_OUTPUT_DIR, f"ovis2_{crop_mode}")
    composites_dir = os.path.join(output_dir, "composites")

    available_pids = set(os.listdir(DATA_ROOT)) if os.path.isdir(DATA_ROOT) else set()

    patients = []
    processed = 0

    for _, row in clinical_df.iterrows():
        if num_patients is not None and processed >= num_patients:
            break

        pid = row["Patient ID"]
        if pid not in available_pids:
            continue

        annot_match = annot_df[annot_df["Patient ID"] == pid]
        if annot_match.empty:
            continue
        annot = annot_match.iloc[0]

        total_str = str(num_patients) if num_patients else "all"
        print(f"[{processed + 1:3d}/{total_str}] Processing {pid}...", end=" ")

        save_dir = os.path.join(composites_dir, pid)
        existing = sorted(glob.glob(os.path.join(save_dir, "*.png")))

        if len(existing) >= SLICES_PER_PATIENT:
            slice_paths = existing[:SLICES_PER_PATIENT]
            print("cached")
        else:
            slice_paths = process_patient(pid, annot, save_dir, crop_mode, crop_kwargs)
            if not slice_paths:
                continue
            print(f"{len(slice_paths)} slices")

        patients.append({
            "pid": pid,
            "label": LABEL_MAP[int(row[LABEL_COLUMN])],
            "clinical_text": build_clinical_text(row),
            "slice_paths": slice_paths,
        })
        processed += 1

    print(f"\nDataset: {len(patients)} patients")
    print(f"Grade distribution: {Counter(p['label'] for p in patients)}")
    return patients
