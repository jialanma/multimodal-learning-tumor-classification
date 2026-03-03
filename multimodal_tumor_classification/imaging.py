"""DCE composite generation, bounding box cropping, and slice sampling."""

import numpy as np
import pandas as pd
from PIL import Image


def normalize_slice(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 255] uint8."""
    mn, mx = arr.min(), arr.max()
    if mx - mn > 0:
        return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
    return np.zeros_like(arr, dtype=np.uint8)


def make_dce_composite(pre: np.ndarray, post1: np.ndarray) -> Image.Image:
    """
    Fuse contrast phases into RGB:
      R = pre-contrast, G = first post-contrast, B = subtraction
    """
    subtracted = np.clip(post1 - pre, 0, None)
    rgb = np.stack([
        normalize_slice(pre),
        normalize_slice(post1),
        normalize_slice(subtracted),
    ], axis=-1)
    return Image.fromarray(rgb, mode='RGB')


def crop_proportional(image: Image.Image, annot: pd.Series,
                      slice_shape: tuple, padding_ratio: float = 0.25) -> Image.Image:
    """Crop to annotation bbox with proportional padding (default 25% of bbox size)."""
    bbox_h = int(annot['End Row']) - int(annot['Start Row'])
    bbox_w = int(annot['End Column']) - int(annot['Start Column'])
    padding = max(5, int(padding_ratio * max(bbox_h, bbox_w)))

    r1 = max(0, int(annot['Start Row']) - 1 - padding)
    r2 = min(slice_shape[0], int(annot['End Row']) + padding)
    c1 = max(0, int(annot['Start Column']) - 1 - padding)
    c2 = min(slice_shape[1], int(annot['End Column']) + padding)
    return image.crop((c1, r1, c2, r2))


def crop_fixed(image: Image.Image, annot: pd.Series,
               slice_shape: tuple, size: int = 256) -> Image.Image:
    """Crop to a fixed-size region centered on the annotation bbox."""
    img_h, img_w = slice_shape

    center_r = (int(annot['Start Row']) - 1 + int(annot['End Row'])) // 2
    center_c = (int(annot['Start Column']) - 1 + int(annot['End Column'])) // 2

    r1 = center_r - size // 2
    r2 = r1 + size
    c1 = center_c - size // 2
    c2 = c1 + size

    if r1 < 0:
        r1, r2 = 0, size
    if r2 > img_h:
        r1, r2 = img_h - size, img_h
    if c1 < 0:
        c1, c2 = 0, size
    if c2 > img_w:
        c1, c2 = img_w - size, img_w

    r1 = max(0, r1)
    c1 = max(0, c1)
    return image.crop((c1, r1, c2, r2))


def apply_crop(image: Image.Image, annot: pd.Series,
               slice_shape: tuple, crop_mode: str, **kwargs) -> Image.Image:
    """Dispatch to the appropriate crop function based on crop_mode."""
    if crop_mode == "proportional":
        return crop_proportional(image, annot, slice_shape,
                                 padding_ratio=kwargs.get("padding_ratio", 0.25))
    elif crop_mode == "fixed256":
        return crop_fixed(image, annot, slice_shape,
                          size=kwargs.get("size", 256))
    elif crop_mode == "none":
        return image
    else:
        raise ValueError(f"Unknown crop mode: {crop_mode}")


def sample_slices_around_tumor(start_idx: int, end_idx: int,
                               num_slices: int = 3) -> list:
    """Return slice indices clustered around the tumor center."""
    center = (start_idx + end_idx) // 2
    half = num_slices // 2

    first = center - half
    last = first + num_slices - 1

    if first < start_idx:
        first = start_idx
        last = first + num_slices - 1
    if last > end_idx:
        last = end_idx
        first = max(start_idx, last - num_slices + 1)

    return list(range(first, last + 1))
