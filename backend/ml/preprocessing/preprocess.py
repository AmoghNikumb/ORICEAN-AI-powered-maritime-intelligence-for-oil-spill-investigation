"""
Preprocessing utilities for ORICEAN SAR inference pipeline.

These functions match the training pipeline preprocessing steps exactly:
  1. dB conversion: 10 * log10(max(VV, eps)) — converts linear SAR backscatter to decibels
  2. Clip to [-25, 0] dB — removes outliers, matches training-time clipping range
  3. Normalize to [0, 255] uint8 — linear map from [-25, 0] dB to [0, 255]

These utilities are used by ml_service.py when converting raw GEO_TIFF pixel
values to PNG bytes before handing them to the OilSpillPredictor.
"""

from __future__ import annotations

import numpy as np

# Training pipeline clipping range (dB)
DB_CLIP_MIN = -25.0
DB_CLIP_MAX = 0.0


def db_convert_and_clip(
    linear_backscatter: np.ndarray,
    eps: float = 1e-10,
) -> np.ndarray:
    """Convert linear SAR backscatter to dB and clip to training range.

    Applies 10 * log10(max(arr, eps)) then clips the result to [DB_CLIP_MIN, DB_CLIP_MAX].

    Parameters
    ----------
    linear_backscatter:
        Array of linear VV backscatter values (any shape, positive floats).
    eps:
        Small epsilon to prevent log(0). Default 1e-10.

    Returns
    -------
    np.ndarray
        Array of dB values clipped to [-25.0, 0.0], same shape as input.
    """
    db = 10.0 * np.log10(np.maximum(linear_backscatter, eps))
    return np.clip(db, DB_CLIP_MIN, DB_CLIP_MAX)


def normalize_db_to_uint8(db_array: np.ndarray) -> np.ndarray:
    """Map [-25, 0] dB linearly to [0, 255] uint8.

    Applies the linear mapping: output = (arr - DB_CLIP_MIN) / 25.0 * 255,
    clipped to [0, 255] and cast to uint8.

    Parameters
    ----------
    db_array:
        Array of dB values (typically in [-25, 0] range).

    Returns
    -------
    np.ndarray
        Array of uint8 values in [0, 255], same shape as input.
    """
    shifted = db_array - DB_CLIP_MIN          # [0, 25] range
    scaled = shifted / (DB_CLIP_MAX - DB_CLIP_MIN) * 255.0
    return scaled.clip(0, 255).astype(np.uint8)
