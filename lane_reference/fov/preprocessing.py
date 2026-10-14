"""CPU reference for Rebellious Hope camera and temporal preprocessing.

This module transliterates the deployed fused runner rather than approximating
it with an image resize. Source anchors in the IQPilot tree are:

* ``tools/compile_supercombo.py:54-115``: perspective sampling, NV12 plane
  addressing, UV matrix scaling, and six-plane packing.
* ``tools/compile_supercombo.py:119-168,193-217``: queue sizes and sampling.
* ``models/runners/tinygrad/supercombo_runtime.py:79-106``: desire edges and
  hidden-state feedback.
* ``tinygrad/mixin/elementwise.py:885-893``: round-half-to-even semantics.

The only non-runtime operation here is :func:`image_to_nv12`, an explicit
offline convenience for RGB/BGR datasets. Camera parity starts from native
NV12 bytes; RGB-to-NV12 conversion is producer-dependent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


MODEL_SIZE = (512, 256)
PACKED_FRAME_SHAPE = (6, 128, 256)
FRAME_SKIP = 4
FEATURE_WIDTH = 512
DESIRE_WIDTH = 8
HIDDEN_SLICE = slice(2066, 2578)

# Elementwise multiplier used by the deployed warp before sampling NV12 U/V.
_UV_MATRIX_SCALE = np.array(
    [[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [2.0, 2.0, 1.0]],
    dtype=np.float32,
)


@dataclass(frozen=True)
class NV12Layout:
  """Byte layout of a camera NV12 buffer, including possible row padding."""

  width: int
  height: int
  stride: int | None = None
  y_height: int | None = None
  uv_height: int | None = None

  def __post_init__(self) -> None:
    stride = self.width if self.stride is None else self.stride
    y_height = self.height if self.y_height is None else self.y_height
    uv_height = self.height // 2 if self.uv_height is None else self.uv_height
    if self.width <= 0 or self.height <= 0 or self.width % 2 or self.height % 2:
      raise ValueError("NV12 width and height must be positive and even")
    if stride < self.width or y_height < self.height or uv_height < self.height // 2:
      raise ValueError("NV12 padded dimensions cannot be smaller than visible dimensions")
    object.__setattr__(self, "stride", int(stride))
    object.__setattr__(self, "y_height", int(y_height))
    object.__setattr__(self, "uv_height", int(uv_height))

  @property
  def byte_size(self) -> int:
    assert self.stride is not None and self.y_height is not None and self.uv_height is not None
    return self.stride * (self.y_height + self.uv_height)


def _project_plane_nearest(
    source: NDArray[np.uint8],
    model_to_source: NDArray[np.float32],
    output_size: tuple[int, int],
) -> NDArray[np.uint8]:
  """Inverse-map one plane using tinygrad's round-half-to-even and clamping."""

  if source.ndim != 2:
    raise ValueError(f"source plane must be 2D, got {source.shape}")
  out_w, out_h = output_size
  if out_w <= 0 or out_h <= 0:
    raise ValueError(f"invalid output size {output_size}")
  matrix = np.asarray(model_to_source, dtype=np.float32)
  if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
    raise ValueError("model_to_source must be a finite 3x3 matrix")

  # Keep the exact tinygrad lowering semantics. Its optimizer contracts the
  # first multiply/add into MULACC (one float32 rounding), then adds the
  # translation, and lowers division to a rounded reciprocal followed by a
  # rounded multiply. Direct NumPy ``numerator / denominator`` differs at a
  # few nearest-neighbor boundaries for nonzero calibration transforms.
  x = np.broadcast_to(np.arange(out_w, dtype=np.float32), (out_h, out_w))
  y = np.broadcast_to(np.arange(out_h, dtype=np.float32)[:, None], (out_h, out_w))

  def tinygrad_dot_row(row: NDArray[np.float32]) -> NDArray[np.float32]:
    y_product = np.asarray(row[1] * y, dtype=np.float32)
    # A float64 operation is exact for this float32 product/add, then the cast
    # supplies MULACC's single float32 rounding without depending on np.longdouble.
    mulacc = np.asarray(
        np.float64(row[0]) * x.astype(np.float64) + y_product.astype(np.float64),
        dtype=np.float32,
    )
    return np.asarray(mulacc + row[2], dtype=np.float32)

  src_x_h = tinygrad_dot_row(matrix[0])
  src_y_h = tinygrad_dot_row(matrix[1])
  src_w = tinygrad_dot_row(matrix[2])
  if np.any(src_w == 0):
    raise ValueError("projective transform has w=0 in the output grid")
  reciprocal_w = np.asarray(1.0 / src_w.astype(np.float64), dtype=np.float32)
  src_x = np.rint(np.asarray(src_x_h * reciprocal_w, dtype=np.float32))
  src_y = np.rint(np.asarray(src_y_h * reciprocal_w, dtype=np.float32))
  if not np.isfinite(src_x).all() or not np.isfinite(src_y).all():
    raise ValueError("projective transform produced non-finite coordinates")

  ix = np.clip(src_x, 0, source.shape[1] - 1).astype(np.intp)
  iy = np.clip(src_y, 0, source.shape[0] - 1).astype(np.intp)
  return np.ascontiguousarray(source[iy, ix])


def split_nv12(buffer: ArrayLike, layout: NV12Layout) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
  """Return visible Y, U, and V views using the deployed camera addressing."""

  flat = np.asarray(buffer, dtype=np.uint8).reshape(-1)
  if flat.size < layout.byte_size:
    raise ValueError(f"NV12 buffer has {flat.size} bytes, expected at least {layout.byte_size}")
  assert layout.stride is not None and layout.y_height is not None
  y_storage = flat[:layout.height * layout.stride].reshape(layout.height, layout.stride)
  uv_offset = layout.stride * layout.y_height
  uv_storage = flat[uv_offset:uv_offset + (layout.height // 2) * layout.stride].reshape(
      layout.height // 2, layout.stride
  )
  y = y_storage[:, :layout.width]
  u = uv_storage[:, :layout.width:2]
  v = uv_storage[:, 1:layout.width:2]
  return y, u, v


def pack_yuv420(y: ArrayLike, u: ArrayLike, v: ArrayLike) -> NDArray[np.uint8]:
  """Pack a model-view YUV420 image as ``[Y00,Y10,Y01,Y11,U,V]``."""

  y_arr = np.asarray(y, dtype=np.uint8)
  u_arr = np.asarray(u, dtype=np.uint8)
  v_arr = np.asarray(v, dtype=np.uint8)
  if y_arr.ndim != 2 or y_arr.shape[0] % 2 or y_arr.shape[1] % 2:
    raise ValueError(f"Y must be an even 2D plane, got {y_arr.shape}")
  expected_uv = (y_arr.shape[0] // 2, y_arr.shape[1] // 2)
  if u_arr.shape != expected_uv or v_arr.shape != expected_uv:
    raise ValueError(f"U/V must be {expected_uv}, got {u_arr.shape} and {v_arr.shape}")
  return np.ascontiguousarray(np.stack((
      y_arr[0::2, 0::2],
      y_arr[1::2, 0::2],
      y_arr[0::2, 1::2],
      y_arr[1::2, 1::2],
      u_arr,
      v_arr,
  )))


def warp_and_pack_nv12(
    buffer: ArrayLike,
    layout: NV12Layout,
    model_to_camera: ArrayLike,
    model_size: tuple[int, int] = MODEL_SIZE,
) -> NDArray[np.uint8]:
  """Apply the fused runner's native NV12 warp and return one six-plane frame.

  ``model_to_camera`` maps homogeneous model-view pixel coordinates to source
  camera pixel coordinates. Out-of-frame coordinates clamp to the nearest edge;
  the deployed path has no constant border fill.
  """

  model_w, model_h = model_size
  if model_w % 2 or model_h % 2:
    raise ValueError("model dimensions must be even")
  matrix = np.asarray(model_to_camera, dtype=np.float32)
  y, u, v = split_nv12(buffer, layout)
  warped_y = _project_plane_nearest(y, matrix, (model_w, model_h))
  uv_matrix = np.asarray(matrix * _UV_MATRIX_SCALE, dtype=np.float32)
  warped_u = _project_plane_nearest(u, uv_matrix, (model_w // 2, model_h // 2))
  warped_v = _project_plane_nearest(v, uv_matrix, (model_w // 2, model_h // 2))
  return pack_yuv420(warped_y, warped_u, warped_v)


def image_to_nv12(image: ArrayLike, color_order: str) -> tuple[NDArray[np.uint8], NV12Layout]:
  """Convert an offline RGB/BGR image to dense NV12 via OpenCV I420.

  This helper is deterministic for a pinned OpenCV build, but it is not a claim
  about camerad's sensor YUV matrix or quantization. ``color_order`` is required
  so replay datasets cannot silently swap RGB and BGR.
  """

  try:
    import cv2
  except ImportError as exc:  # pragma: no cover - environment dependent
    raise RuntimeError("image_to_nv12 requires OpenCV") from exc
  arr = np.asarray(image)
  if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
    raise ValueError(f"image must be uint8 HxWx3, got {arr.dtype} {arr.shape}")
  height, width = arr.shape[:2]
  if width % 2 or height % 2:
    raise ValueError("image width and height must be even for NV12")
  order = color_order.upper()
  if order not in ("RGB", "BGR"):
    raise ValueError("color_order must be explicitly 'RGB' or 'BGR'")
  code = cv2.COLOR_RGB2YUV_I420 if order == "RGB" else cv2.COLOR_BGR2YUV_I420
  i420 = cv2.cvtColor(arr, code).reshape(-1)
  y_size = width * height
  uv_size = y_size // 4
  y = i420[:y_size].reshape(height, width)
  u = i420[y_size:y_size + uv_size].reshape(height // 2, width // 2)
  v = i420[y_size + uv_size:y_size + 2 * uv_size].reshape(height // 2, width // 2)
  nv12 = np.empty((height + height // 2, width), dtype=np.uint8)
  nv12[:height] = y
  nv12[height:, 0::2] = u
  nv12[height:, 1::2] = v
  return np.ascontiguousarray(nv12.reshape(-1)), NV12Layout(width, height)


class RebelliousHopeState:
  """Exact frame-skip-4 queue state for the six-input public ONNX.

  Call :meth:`push` for tick ``t``, run the model, then call
  :meth:`note_model_output` with that tick's output. Internal auxiliary queues
  remain float32 like the deployed host runtime; returned native ONNX inputs are
  float16 unless ``native=False`` is requested.
  """

  def __init__(self) -> None:
    self.reset()

  def reset(self) -> None:
    self.img_q = np.zeros((5, *PACKED_FRAME_SHAPE), dtype=np.uint8)
    self.big_img_q = np.zeros((5, *PACKED_FRAME_SHAPE), dtype=np.uint8)
    self.feat_q = np.zeros((96, FEATURE_WIDTH), dtype=np.float32)
    self.desire_q = np.zeros((100, DESIRE_WIDTH), dtype=np.float32)
    self.prev_desire = np.zeros(DESIRE_WIDTH, dtype=np.float32)
    self.prev_feat = np.zeros(FEATURE_WIDTH, dtype=np.float32)

  @staticmethod
  def _shift_append(queue: NDArray, value: NDArray) -> None:
    queue[:-1] = queue[1:]
    queue[-1] = value

  def push(
      self,
      warped: ArrayLike,
      desire: ArrayLike,
      traffic_convention: ArrayLike,
      action_t: ArrayLike,
      *,
      native: bool = True,
  ) -> dict[str, NDArray]:
    """Advance one inference tick and materialize all six ONNX inputs."""

    frames = np.asarray(warped)
    if frames.dtype != np.uint8 or frames.shape != (2, *PACKED_FRAME_SHAPE):
      raise ValueError(f"warped must be uint8 {(2, *PACKED_FRAME_SHAPE)}, got {frames.dtype} {frames.shape}")
    cur_desire = np.asarray(desire, dtype=np.float32).reshape(-1)
    if cur_desire.shape != (DESIRE_WIDTH,):
      raise ValueError(f"desire must have {DESIRE_WIDTH} values")
    cur_desire = cur_desire.copy()
    cur_desire[0] = 0.0
    pulse = np.where(cur_desire - self.prev_desire > np.float32(0.99), cur_desire, 0.0).astype(np.float32)
    self.prev_desire[:] = cur_desire

    traffic = np.asarray(traffic_convention, dtype=np.float32).reshape(-1)
    action = np.asarray(action_t, dtype=np.float32).reshape(-1)
    if traffic.shape != (2,) or action.shape != (2,):
      raise ValueError("traffic_convention and action_t must each contain two values")

    self._shift_append(self.img_q, frames[0])
    self._shift_append(self.big_img_q, frames[1])
    self._shift_append(self.desire_q, pulse)
    self._shift_append(self.feat_q, self.prev_feat)

    aux_dtype = np.float16 if native else np.float32
    return {
        "img": np.ascontiguousarray(self.img_q[::FRAME_SKIP].reshape(1, 12, 128, 256)),
        "big_img": np.ascontiguousarray(self.big_img_q[::FRAME_SKIP].reshape(1, 12, 128, 256)),
        "desire_pulse": np.ascontiguousarray(
            self.desire_q.reshape(25, FRAME_SKIP, DESIRE_WIDTH).max(axis=1)[None].astype(aux_dtype)
        ),
        "traffic_convention": np.ascontiguousarray(traffic.reshape(1, 2).astype(aux_dtype)),
        "action_t": np.ascontiguousarray(action.reshape(1, 2).astype(aux_dtype)),
        "features_buffer": np.ascontiguousarray(
            self.feat_q[::FRAME_SKIP].reshape(1, 24, FEATURE_WIDTH).astype(aux_dtype)
        ),
    }

  def note_model_output(self, output: ArrayLike | Mapping[str, ArrayLike]) -> None:
    """Save the current 512-value hidden state for the next tick.

    Accepts a flat 2580-value model output, a 512-value hidden vector, or a
    mapping containing ``hidden_state``.
    """

    value = output["hidden_state"] if isinstance(output, Mapping) else output
    flat = np.asarray(value, dtype=np.float32).reshape(-1)
    if flat.size == FEATURE_WIDTH:
      hidden = flat
    elif flat.size >= HIDDEN_SLICE.stop:
      hidden = flat[HIDDEN_SLICE]
    else:
      raise ValueError(f"expected 512 hidden values or a full output, got {flat.size}")
    if not np.isfinite(hidden).all():
      raise ValueError("hidden state contains non-finite values")
    self.prev_feat[:] = hidden


__all__ = [
    "DESIRE_WIDTH",
    "FEATURE_WIDTH",
    "FRAME_SKIP",
    "HIDDEN_SLICE",
    "MODEL_SIZE",
    "NV12Layout",
    "PACKED_FRAME_SHAPE",
    "RebelliousHopeState",
    "image_to_nv12",
    "pack_yuv420",
    "split_nv12",
    "warp_and_pack_nv12",
]
