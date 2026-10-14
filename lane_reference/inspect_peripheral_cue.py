"""Exact RGB/NV12 conversion helpers, extracted without simulator side effects."""
import numpy as np
from lane_reference.fov.preprocessing import NV12Layout, MODEL_SIZE

def rgb_to_nv12_bt601_limited(rgb: np.ndarray) -> tuple[np.ndarray, NV12Layout]:
    """Deterministic offline RGB conversion, not a claim about live camerad bytes."""
    value = np.asarray(rgb, dtype=np.float64)
    if value.dtype != np.float64 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB input, got {value.shape}")
    height, width = value.shape[:2]
    if height % 2 or width % 2:
        raise ValueError("NV12 conversion requires even dimensions")
    red, green, blue = value[..., 0], value[..., 1], value[..., 2]
    y = np.clip(np.rint(16.0 + 0.256788 * red + 0.504129 * green + 0.097906 * blue), 16, 235).astype(np.uint8)
    cb_full = 128.0 - 0.148223 * red - 0.290993 * green + 0.439216 * blue
    cr_full = 128.0 + 0.439216 * red - 0.367788 * green - 0.071427 * blue
    cb = np.clip(np.rint(cb_full.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))), 16, 240).astype(np.uint8)
    cr = np.clip(np.rint(cr_full.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))), 16, 240).astype(np.uint8)
    nv12 = np.empty((height + height // 2, width), dtype=np.uint8)
    nv12[:height] = y
    nv12[height:, 0::2] = cb
    nv12[height:, 1::2] = cr
    return np.ascontiguousarray(nv12.reshape(-1)), NV12Layout(width, height)


def unpack_six_plane_yuv420(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(packed, dtype=np.uint8)
    if value.shape != (6, MODEL_SIZE[1] // 2, MODEL_SIZE[0] // 2):
        raise ValueError(f"unexpected packed frame shape {value.shape}")
    y = np.empty((MODEL_SIZE[1], MODEL_SIZE[0]), dtype=np.uint8)
    y[0::2, 0::2], y[1::2, 0::2] = value[0], value[1]
    y[0::2, 1::2], y[1::2, 1::2] = value[2], value[3]
    return y, value[4], value[5]


def packed_to_rgb_bt601_limited(packed: np.ndarray) -> np.ndarray:
    y, cb, cr = unpack_six_plane_yuv420(packed)
    cb_full = np.repeat(np.repeat(cb, 2, axis=0), 2, axis=1).astype(np.float64) - 128.0
    cr_full = np.repeat(np.repeat(cr, 2, axis=0), 2, axis=1).astype(np.float64) - 128.0
    luminance = np.maximum(y.astype(np.float64) - 16.0, 0.0)
    red = 1.164383 * luminance + 1.596027 * cr_full
    green = 1.164383 * luminance - 0.391762 * cb_full - 0.812968 * cr_full
    blue = 1.164383 * luminance + 2.017232 * cb_full
    return np.clip(np.rint(np.stack([red, green, blue], axis=-1)), 0, 255).astype(np.uint8)
