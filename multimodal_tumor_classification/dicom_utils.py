"""DICOM loading, series identification, and annotation alignment."""

import os
import glob
import numpy as np
import pandas as pd
import pydicom


def parse_slice_number(filepath: str) -> int:
    """Extract slice number from DICOM filename (e.g. '1-050.dcm' -> 50)."""
    basename = os.path.splitext(os.path.basename(filepath))[0]
    parts = basename.split('-')
    if len(parts) >= 2:
        try:
            return int(parts[-1])
        except ValueError:
            pass
    return -1


def load_dicom_series(series_dir: str) -> tuple:
    """
    Load all DICOMs sorted by filename (ascending).

    Returns:
        volume: np.ndarray (D, H, W) or None
        ascending_z: bool or None
        total_slices: int
    """
    dcm_files = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))
    if not dcm_files:
        dcm_files = sorted([
            f for f in glob.glob(os.path.join(series_dir, "*"))
            if os.path.isfile(f)
        ])
    if not dcm_files:
        return None, None, 0

    dcm_files.sort(key=lambda f: parse_slice_number(f))

    slices = []
    z_positions = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f)
            slices.append(ds.pixel_array.astype(np.float32))
            z_positions.append(float(ds.ImagePositionPatient[2]))
        except Exception:
            continue

    if not slices:
        return None, None, 0

    volume = np.stack(slices)
    total_slices = volume.shape[0]
    ascending_z = z_positions[0] <= z_positions[-1] if len(z_positions) >= 2 else True
    return volume, ascending_z, total_slices


def adjust_annotation_slices(annot: pd.Series, total_slices: int,
                             _ascending_z: bool = True) -> tuple:
    """
    Convert 1-indexed annotation slice numbers to 0-indexed volume indices.

    Returns:
        (start_idx, end_idx) -- 0-indexed, inclusive
    """
    start_idx = int(annot['Start Slice']) - 1
    end_idx = int(annot['End Slice']) - 1

    start_idx = max(0, min(start_idx, total_slices - 1))
    end_idx = max(0, min(end_idx, total_slices - 1))

    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    return start_idx, end_idx


def identify_series(patient_dir: str) -> dict:
    """
    Identify pre-contrast and first post-contrast DCE series.

    Returns:
        {'pre': dir, 'post1': dir} or {} on failure
    """
    study_dirs = sorted(glob.glob(os.path.join(patient_dir, "*", "*")))
    if not study_dirs:
        study_dirs = sorted(glob.glob(os.path.join(patient_dir, "*")))

    series_info = []
    for sdir in study_dirs:
        if not os.path.isdir(sdir):
            continue
        sample_files = sorted(glob.glob(os.path.join(sdir, "*")))[:1]
        if not sample_files:
            continue
        try:
            ds = pydicom.dcmread(sample_files[0], stop_before_pixels=True)
            desc = getattr(ds, 'SeriesDescription', '').strip().lower()
            num = int(getattr(ds, 'SeriesNumber', 999))
            series_info.append({
                'dir': sdir,
                'description': desc,
                'series_number': num,
            })
        except Exception:
            continue

    pre_dir = None
    post1_dir = None

    def _is_dce(desc):
        return 'dyn' in desc or 'vibrant' in desc

    for s in series_info:
        desc = s['description']
        if _is_dce(desc) and ('pre' in desc or not any(
            p in desc for p in ['ph1', 'ph2', 'ph3', 'ph4', '1st', '2nd', '3rd', '4th']
        )):
            pre_dir = s['dir']
        elif _is_dce(desc) and ('1st' in desc or 'ph1' in desc):
            post1_dir = s['dir']

    if pre_dir and post1_dir:
        return {'pre': pre_dir, 'post1': post1_dir}

    # Fallback: first two DCE series by series number
    series_info.sort(key=lambda x: x['series_number'])
    dynamic = [s for s in series_info if _is_dce(s['description'])]
    if len(dynamic) >= 2:
        return {'pre': dynamic[0]['dir'], 'post1': dynamic[1]['dir']}

    print(f"  Could not identify series. Found: "
          f"{[s['description'] for s in series_info]}")
    return {}
