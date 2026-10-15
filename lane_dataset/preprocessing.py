"""Stock MED/SBIG preprocessing with bundled reference implementations.

Only MED/SBIG projections are exposed. The synthetic RGB producer conversion is
the locally pinned BT.601 limited-range producer, distinct from sensor calibration.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from functools import lru_cache

import numpy as np

DEPENDENCY_ROOT = Path(os.environ.get(
    "LANE_DATASET_REFERENCE_ROOT", str(Path(__file__).resolve().parents[1] / "lane_reference")
)).resolve()


@lru_cache(maxsize=None)
def reference_module(name: str):
    if name not in {"preprocessing", "camera", "frame_archive_v2", "expert_trajectory_loss"}:
        raise ValueError("unsupported read-only reference module")
    path = DEPENDENCY_ROOT / "fov" / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"missing reference implementation: {path}")
    module_name = f"_lane_dataset_reference_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_reference = reference_module("preprocessing")
NV12Layout = _reference.NV12Layout
split_nv12 = _reference.split_nv12
pack_yuv420 = _reference.pack_yuv420
warp_and_pack_nv12 = _reference.warp_and_pack_nv12
_project_plane_nearest = _reference._project_plane_nearest
RebelliousHopeState = _reference.RebelliousHopeState
PACKED_FRAME_SHAPE = _reference.PACKED_FRAME_SHAPE
MODEL_SIZE = _reference.MODEL_SIZE

RGB_PRODUCER_SOURCE = DEPENDENCY_ROOT / "inspect_peripheral_cue.py"


@lru_cache(maxsize=1)
def _rgb_producer():
    """Load only the producer function, avoiding capture script import effects."""
    tree = ast.parse(RGB_PRODUCER_SOURCE.read_text(encoding="utf-8-sig"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "rgb_to_nv12_bt601_limited"]
    if len(nodes) != 1:
        raise RuntimeError("pinned synthetic RGB producer function is missing")
    namespace = {"np": np, "NV12Layout": NV12Layout}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(RGB_PRODUCER_SOURCE), "exec"), namespace)
    return namespace["rgb_to_nv12_bt601_limited"]


def image_to_nv12(image, color_order):
    value = np.asarray(image)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError("synthetic RGB source must be uint8[H,W,3]")
    if color_order not in ("RGB", "BGR"):
        raise ValueError("color_order must explicitly be RGB or BGR")
    value = value if color_order == "RGB" else value[..., ::-1]
    height, width = value.shape[:2]
    producer = _rgb_producer()
    if height <= 16 or height % 2 or width % 2:
        return producer(value)
    # The pinned producer has only pointwise arithmetic and 2x2 chroma
    # reductions. Even row tiles preserve every operation and rounding result,
    # while keeping its float64 temporaries in cache instead of streaming
    # several full-resolution RGB copies through memory for each camera tick.
    output = np.empty((height + height // 2, width), dtype=np.uint8)
    for start in range(0, height, 16):
        end = min(start + 16, height)
        packed, _ = producer(value[start:end])
        packed = packed.reshape((end-start)*3//2, width)
        output[start:end] = packed[:end-start]
        output[height+start//2:height+end//2] = packed[end-start:]
    return output.reshape(-1), NV12Layout(width, height)


class StockPreprocessor:
    def __init__(self, calibration_rpy=(0.0, 0.0, 0.0)):
        camera = reference_module("camera")
        self.calibration_rpy = np.asarray(calibration_rpy, dtype=np.float32)
        if self.calibration_rpy.shape != (3,) or not np.isfinite(self.calibration_rpy).all():
            raise ValueError("calibration_rpy must be finite roll,pitch,yaw")
        self.profiles = (
            camera.CameraProfile("nominal-narrow", focal_px=2648.0),
            camera.CameraProfile("nominal-wide", focal_px=567.0),
        )
        self.warps = tuple(camera.original_warp(profile, self.calibration_rpy, wide=bool(i))
                           for i, profile in enumerate(self.profiles))

    def pack_nv12(self, narrow, wide, layouts=None):
        layouts = layouts or tuple(NV12Layout(p.width, p.height) for p in self.profiles)
        return np.stack([warp_and_pack_nv12(value, layout, warp)
                         for value, layout, warp in zip((narrow, wide), layouts, self.warps)])

    def pack_rgb(self, narrow_rgb, wide_rgb):
        converted = []
        for value, profile in zip((narrow_rgb, wide_rgb), self.profiles):
            if np.shape(value) != (profile.height, profile.width, 3):
                raise ValueError("RGB source does not match nominal camera dimensions")
            converted.append(image_to_nv12(value, "RGB"))
        return self.pack_nv12(converted[0][0], converted[1][0],
                              (converted[0][1], converted[1][1]))

    def contract(self):
        paths = {name: DEPENDENCY_ROOT / "fov" / f"{name}.py"
                 for name in ("preprocessing", "camera", "frame_archive_v2")}
        paths["rgb_producer"] = RGB_PRODUCER_SOURCE
        return {
            "schema_version": 1,
            "model_contract": "stock_MED_SBIG_uint8_shared_camera_format",
            "crops": ["MED", "SBIG"],
            "physical_optics": "nominal_assumed_not_physically_calibrated",
            "source_size": [1928, 1208],
            "focal_pixels": [2648.0, 567.0],
            "model_to_camera": [w.tolist() for w in self.warps],
            "calibration_rpy": self.calibration_rpy.tolist(),
            "rgb_to_nv12": "BT.601 limited-range float64, round-half-even Y, 2x2 mean then round-half-even UV",
            "rgb_producer_function": "rgb_to_nv12_bt601_limited",
            "numpy_version": np.__version__,
            "packed_order": ["Y00", "Y10", "Y01", "Y11", "U", "V"],
            "native_image_dtype": "uint8",
            "reference_sources": {name: {"path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for name, path in paths.items()},
        }
