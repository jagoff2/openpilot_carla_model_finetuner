"""Conservative rendered-pixel evidence for the unmodified MED/SBIG inputs.

World-to-camera matrices use optical coordinates (right, down, forward).
Evidence establishes that pixels were available to the student; it does not
establish that a particular student's recurrent features retained those pixels.
Neither the full map nor a future image is an admissible evidence source.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class EvidenceRef:
    frame_id: int
    timestamp_s: float
    camera: str


@dataclass(frozen=True)
class CameraObservation:
    frame_id: int
    timestamp_s: float
    camera: str
    rgb: np.ndarray
    depth_m: np.ndarray
    camera_from_world: np.ndarray
    intrinsics: np.ndarray
    model_to_source: np.ndarray
    depth_mode: str = "forward"
    depth_validated: bool = False
    marking_mask: np.ndarray | None = None
    # Optional view from this same RGB frame's canonical NV12 conversion.
    # Capture can reuse it for both packing and evidence instead of converting
    # each full-resolution RGB frame twice. It is never a differently resized Y.
    luma: np.ndarray | None = None
    road_mask: np.ndarray | None = None
    road_exterior_mask: np.ndarray | None = None
    chroma_u: np.ndarray | None = None
    chroma_v: np.ndarray | None = None
    surface_normals_world: np.ndarray | None = None


@dataclass(frozen=True)
class EvidenceThresholds:
    absolute_depth_tolerance_m: float = 0.20
    relative_depth_tolerance: float = 0.015
    minimum_luma_contrast: float = 12.0
    minimum_chroma_contrast: float = 6.0
    minimum_marking_luma: float = 70.0
    center_radius_px: int = 1
    flank_radius_px: int = 4
    maximum_history_s: float = 4.8
    minimum_history_visibility_s: float = 0.10
    minimum_crop_preview_loss_m: float = 2.0


@dataclass(frozen=True)
class FrameEvidence:
    reference: EvidenceRef
    point_ids: tuple[str, ...]
    arc_m: np.ndarray
    source_visible: np.ndarray
    crop_visible: np.ndarray
    source_pixels: np.ndarray
    model_pixels: np.ndarray
    source_contrast: np.ndarray
    crop_contrast: np.ndarray
    source_depth_residual_m: np.ndarray
    crop_depth_residual_m: np.ndarray
    source_boundary_kind: np.ndarray | None = None
    crop_boundary_kind: np.ndarray | None = None
    crop_edge_road_bits: np.ndarray | None = None
    crop_edge_other_bits: np.ndarray | None = None
    crop_edge_other_depth_residual_m: np.ndarray | None = None
    crop_paint_mask_bits: np.ndarray | None = None
    sample_depth_basis: str = 'projected_point'

    def compact(self) -> dict:
        """Lossless scalar evidence; image payloads belong in the frame archive."""
        value = asdict(self)
        return {key: item.tolist() if isinstance(item, np.ndarray) else item
                for key, item in value.items()}


@dataclass(frozen=True)
class ObservableSupport:
    point_ids: tuple[str, ...]
    distance_m: np.ndarray
    point_times_s: np.ndarray
    required_mask: np.ndarray
    supported: np.ndarray
    current_visible: np.ndarray
    current_source_visible: np.ndarray
    evidence_frames: tuple[tuple[EvidenceRef, ...], ...]
    visibility_duration_s: np.ndarray
    preview_distance_m: float
    source_preview_distance_m: float
    historical_preview_distance_m: float
    supported_horizon_s: float
    crop_preview_loss_m: float
    action_valid: bool
    category: str
    reasons: tuple[str, ...]

    @property
    def crop_loss_qualified(self) -> bool:
        return self.category == "history_crop_loss"

    def compact(self) -> dict:
        value = asdict(self)
        return {key: item.tolist() if isinstance(item, np.ndarray) else item
                for key, item in value.items()}


def decode_carla_depth_bgra(bgra: np.ndarray) -> np.ndarray:
    """Decode raw sensor.camera.depth, never its color-converted preview.

    See https://carla.readthedocs.io/en/0.9.16/ref_sensors/#depth-camera.
    Whether these are forward or radial distances must be checked on fixtures.
    """
    image = np.asarray(bgra)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 4:
        raise ValueError("raw depth must be uint8 HxWx4 BGRA")
    encoded = (image[..., 2].astype(np.uint32)
               + 256 * image[..., 1].astype(np.uint32)
               + 65536 * image[..., 0].astype(np.uint32))
    return (encoded.astype(np.float64) * (1000.0 / 16777215)).astype(np.float32)


def validate_depth_geometry(measured_m: np.ndarray, camera_points: np.ndarray,
                            tolerance_m: float = 0.10) -> dict:
    """Qualify a depth convention against independently known fixture geometry.

    Use off-axis samples: axial-only fixtures cannot distinguish conventions.
    The caller must retain the fixture identity and result in qualification.
    """
    points = np.asarray(camera_points, dtype=np.float64)
    measured = np.asarray(measured_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or measured.shape != (len(points),):
        raise ValueError("expected measured[N] and optical camera points[N,3]")
    if len(points) < 3 or not np.isfinite(points).all() or not np.isfinite(measured).all():
        raise ValueError("depth calibration requires at least three finite samples")
    if np.any(points[:, 2] <= 0) or np.any(measured <= 0) or tolerance_m <= 0:
        raise ValueError("depth calibration requires positive depth and tolerance")
    forward, radial = points[:, 2], np.linalg.norm(points, axis=1)
    residuals = {"forward": float(np.max(np.abs(measured - forward))),
                 "radial": float(np.max(np.abs(measured - radial)))}
    passing = [mode for mode, residual in residuals.items() if residual <= tolerance_m]
    return {"qualified": len(passing) == 1,
            "depth_mode": passing[0] if len(passing) == 1 else None,
            "maximum_residual_m": residuals,
            "reason": "unique_depth_convention" if len(passing) == 1 else
                      "ambiguous_or_inaccurate_depth_fixture"}


def project_world_points(points_world: np.ndarray, camera_from_world: np.ndarray,
                         intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float64)
    transform = np.asarray(camera_from_world, dtype=np.float64)
    camera_k = np.asarray(intrinsics, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_world must be finite [N,3]")
    if transform.shape != (4, 4) or camera_k.shape != (3, 3):
        raise ValueError("expected camera_from_world[4,4] and intrinsics[3,3]")
    if not np.isfinite(transform).all() or not np.isfinite(camera_k).all():
        raise ValueError("camera matrices must be finite")
    optical = points @ transform[:3, :3].T + transform[:3, 3]
    homogeneous = optical @ camera_k.T
    pixels = np.full((len(points), 2), np.nan)
    np.divide(homogeneous[:, :2], homogeneous[:, 2:3], out=pixels,
              where=np.abs(homogeneous[:, 2:3]) > 1e-12)
    return pixels, optical


def _pixel_tests(luma: np.ndarray, depth: np.ndarray, pixels: np.ndarray,
                 expected_depth: np.ndarray, thresholds: EvidenceThresholds,
                 marking_mask: np.ndarray | None = None, depth_predictor=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    visible = np.zeros(len(pixels), dtype=bool)
    contrasts = np.zeros(len(pixels), dtype=np.float32)
    residuals = np.full(len(pixels), np.inf, dtype=np.float32)
    radius = thresholds.flank_radius_px
    core = thresholds.center_radius_px
    if not 0 <= core < radius:
        raise ValueError("contrast flank must surround the marking center")
    outer_y, outer_x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    flanks = np.maximum(np.abs(outer_x), np.abs(outer_y)) >= radius - 1
    for index, pixel in enumerate(pixels):
        if not np.isfinite(pixel).all() or expected_depth[index] <= 0:
            continue
        x, y = np.rint(pixel).astype(int)
        if x < radius or y < radius or x >= luma.shape[1] - radius or y >= luma.shape[0] - radius:
            continue
        patch = luma[y-radius:y+radius+1, x-radius:x+radius+1].astype(np.float32)
        center = luma[y-core:y+core+1, x-core:x+core+1].astype(np.float32)
        # A thin distant marking can occupy one pixel. Require contrast against
        # most surrounding road, then check depth at the SAME bright pixel.
        peak_y, peak_x = np.unravel_index(int(np.argmax(center)), center.shape)
        sample_y, sample_x = y - core + peak_y, x - core + peak_x
        peak = float(center[peak_y, peak_x])
        contrasts[index] = peak - float(np.percentile(patch[flanks], 60))
        sample_depth = float(depth[sample_y, sample_x])
        expected=expected_depth[index] if depth_predictor is None else depth_predictor(index,sample_x,sample_y)
        residuals[index] = abs(sample_depth - expected)
        tolerance = (thresholds.absolute_depth_tolerance_m
                     + thresholds.relative_depth_tolerance * expected_depth[index])
        semantic_ok = marking_mask is None or bool(marking_mask[sample_y, sample_x])
        visible[index] = (np.isfinite(sample_depth) and sample_depth > 0
                          and np.isfinite(expected) and residuals[index] <= tolerance
                          and peak >= thresholds.minimum_marking_luma
                          and contrasts[index] >= thresholds.minimum_luma_contrast
                          and semantic_ok)
    return visible, contrasts, residuals


def probe_lane_evidence(observation: CameraObservation, points_world: np.ndarray,
                        point_ids: Sequence[str], arc_m: np.ndarray,
                        thresholds: EvidenceThresholds = EvidenceThresholds()) -> FrameEvidence:
    """Test source pixels and the EXACT native-preprocessing luma samples.

    Source visibility alone never grants model support. Clamped edge pixels
    cannot make out-of-crop projected points visible. Road marking support must
    survive the stock warp, and must agree with separately qualified depth.
    """
    from .preprocessing import NV12Layout, _project_plane_nearest, image_to_nv12, split_nv12

    if observation.camera not in ("narrow", "wide"):
        raise ValueError("camera must be narrow or wide")
    if not observation.depth_validated:
        raise ValueError("depth convention/alignment must be fixture-qualified first")
    if observation.depth_mode not in ("forward", "radial"):
        raise ValueError("depth_mode must be explicitly forward or radial")
    if not np.isfinite(observation.timestamp_s):
        raise ValueError("evidence timestamp must be finite")
    ids = tuple(str(point_id) for point_id in point_ids)
    arc = np.asarray(arc_m, dtype=np.float64)
    if len(set(ids)) != len(ids) or arc.shape != (len(ids),) or not np.isfinite(arc).all():
        raise ValueError("unique point ids and finite arc_m[N] required")
    if observation.luma is None:
        nv12, layout = image_to_nv12(observation.rgb, color_order="RGB")
        luma, source_u, source_v = split_nv12(nv12, layout)
    else:
        rgb = np.asarray(observation.rgb)
        luma = np.asarray(observation.luma)
        if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
                or luma.dtype != np.uint8 or luma.shape != rgb.shape[:2]):
            raise ValueError("precomputed luma must be uint8 and match the source RGB frame")
        layout = NV12Layout(rgb.shape[1], rgb.shape[0])
        source_u,source_v=observation.chroma_u,observation.chroma_v
    depth = np.asarray(observation.depth_m)
    if depth.shape != luma.shape:
        raise ValueError("depth and RGB must have identical aligned dimensions")
    if observation.marking_mask is not None and observation.marking_mask.shape != luma.shape:
        raise ValueError("marking mask must be aligned with RGB")
    pixels, optical = project_world_points(points_world, observation.camera_from_world,
                                           observation.intrinsics)
    if len(pixels) != len(ids):
        raise ValueError("point_ids and world points must have equal length")
    expected_depth = optical[:, 2] if observation.depth_mode == "forward" else np.linalg.norm(optical, axis=1)
    expected_depth = np.where(optical[:, 2] > 0, expected_depth, -1)
    matrix = np.asarray(observation.model_to_source, dtype=np.float32)
    model_homogeneous = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(matrix).T
    model_pixels = np.full_like(pixels, np.nan)
    np.divide(model_homogeneous[:, :2], model_homogeneous[:, 2:3], out=model_pixels,
              where=np.abs(model_homogeneous[:, 2:3]) > 1e-12)
    warped_luma = _project_plane_nearest(luma, matrix, (512, 256))
    warped_depth = _project_plane_nearest(depth, matrix, (512, 256))
    warped_marking = (None if observation.marking_mask is None else
                     _project_plane_nearest(observation.marking_mask, matrix, (512, 256)))
    source_prediction=crop_prediction=None
    if observation.surface_normals_world is not None:
        from .depth_sampling import plane_depth_predictor
        source_prediction=plane_depth_predictor(observation,optical)
        crop_prediction=plane_depth_predictor(observation,optical,crop=True)
    source_ok, source_contrast, source_residual = _pixel_tests(
        luma, depth, pixels, expected_depth, thresholds, observation.marking_mask,source_prediction)
    crop_ok, crop_contrast, crop_residual = _pixel_tests(
        warped_luma, warped_depth, model_pixels, expected_depth, thresholds, warped_marking,crop_prediction)
    source_kind=source_ok.astype(np.uint8);crop_kind=crop_ok.astype(np.uint8)
    edge_proof={}
    if observation.marking_mask is not None and observation.marking_mask.any():
        from .boundary_color import paint_color_tests
        from .preprocessing import reference_module
        if source_u is None or source_v is None:
            _,source_u,source_v=split_nv12(*image_to_nv12(observation.rgb,color_order='RGB'))
        if source_u.shape!=(layout.height//2,layout.width//2) or source_v.shape!=source_u.shape:
            raise ValueError('chroma planes must match the canonical NV12 layout')
        uv_matrix=np.asarray(matrix*reference_module('preprocessing')._UV_MATRIX_SCALE,np.float32)
        source_color=np.stack((source_u,source_v),axis=-1).repeat(2,axis=0).repeat(2,axis=1)
        crop_color=np.stack([_project_plane_nearest(plane,uv_matrix,(256,128)) for plane in (source_u,source_v)],axis=-1).repeat(2,axis=0).repeat(2,axis=1)
        source_paint=paint_color_tests(luma,source_color,depth,pixels,expected_depth,observation.marking_mask,thresholds,source_prediction)
        crop_paint=paint_color_tests(warped_luma,crop_color,warped_depth,model_pixels,expected_depth,warped_marking,thresholds,crop_prediction)
        for result,ok,contrast,residual,kind in ((source_paint,source_ok,source_contrast,source_residual,source_kind),
                                               (crop_paint,crop_ok,crop_contrast,crop_residual,crop_kind)):
            selected=result[0]&~ok
            contrast[selected]=result[1][selected];residual[selected]=result[2][selected]
            kind[selected]=3;ok[selected]=True
        edge_proof['crop_paint_mask_bits']=crop_paint[3]
    if observation.road_mask is not None or observation.road_exterior_mask is not None:
        from .boundary_edges import edge_pixel_tests
        if observation.marking_mask is None:
            raise ValueError('edge evidence requires an explicit paint mask')
        road,other=observation.road_mask,observation.road_exterior_mask
        if road is None or other is None or road.shape!=luma.shape or other.shape!=luma.shape:
            raise ValueError('road and exterior masks must both align with RGB')
        source_edge=edge_pixel_tests(luma,depth,pixels,expected_depth,road,other,thresholds,source_prediction)
        crop_edge=edge_pixel_tests(warped_luma,warped_depth,model_pixels,expected_depth,
            _project_plane_nearest(road,matrix,(512,256)),
            _project_plane_nearest(other,matrix,(512,256)),thresholds,crop_prediction)
        for result,ok,contrast,residual,kind in ((source_edge,source_ok,source_contrast,source_residual,source_kind),
                                               (crop_edge,crop_ok,crop_contrast,crop_residual,crop_kind)):
            selected=result[0]&~ok
            contrast[selected]=result[1][selected];residual[selected]=result[2][selected]
            kind[selected]=2;ok[selected]=True
        edge_proof.update(crop_edge_road_bits=crop_edge[4],crop_edge_other_bits=crop_edge[5],
                         crop_edge_other_depth_residual_m=crop_edge[3])
    # Avoid crediting bright clamp artifacts or a different marking sampled at
    # the image boundary when the geometric ray never intersects source pixels.
    in_source = ((pixels[:, 0] >= 0) & (pixels[:, 0] < layout.width)
                 & (pixels[:, 1] >= 0) & (pixels[:, 1] < layout.height))
    crop_ok &= in_source
    crop_kind[~crop_ok]=0
    return FrameEvidence(EvidenceRef(observation.frame_id, observation.timestamp_s, observation.camera),
                         ids, arc.copy(), source_ok, crop_ok, pixels, model_pixels,
                         source_contrast, crop_contrast, source_residual, crop_residual,
                         source_boundary_kind=source_kind,crop_boundary_kind=crop_kind,
                         sample_depth_basis='local_surface_plane_at_exact_sample_pixel' if source_prediction else 'projected_point',**edge_proof)


class EvidenceHistory:
    """Bounded evidence ledger containing only previously admitted observations."""

    def __init__(self, thresholds: EvidenceThresholds = EvidenceThresholds()):
        self.thresholds = thresholds
        self._frames: list[FrameEvidence] = []
        self._keys: set[tuple[int, str]] = set()

    def admit(self, evidence: FrameEvidence) -> None:
        ref = evidence.reference
        key = (ref.frame_id, ref.camera)
        if key in self._keys:
            raise ValueError("duplicate evidence frame/camera")
        if self._frames and ref.timestamp_s < self._frames[-1].reference.timestamp_s - 1e-8:
            raise ValueError("evidence must be admitted in timestamp order")
        if ref.camera not in ("narrow", "wide") or not np.isfinite(ref.timestamp_s):
            raise ValueError("invalid evidence camera/timestamp")
        n = len(evidence.point_ids)
        if len(set(evidence.point_ids)) != n:
            raise ValueError("duplicate evidence point id")
        for value in (evidence.arc_m, evidence.source_visible, evidence.crop_visible,
                      evidence.source_contrast, evidence.crop_contrast,
                      evidence.source_depth_residual_m, evidence.crop_depth_residual_m):
            if np.asarray(value).shape != (n,):
                raise ValueError("invalid evidence array shape")
        self._frames.append(evidence)
        self._keys.add(key)
        cutoff = ref.timestamp_s - self.thresholds.maximum_history_s - 1e-8
        self._frames = [frame for frame in self._frames if frame.reference.timestamp_s >= cutoff]
        self._keys = {(frame.reference.frame_id, frame.reference.camera) for frame in self._frames}

    def support(self, decision_frame_id: int, timestamp_s: float, point_ids: Sequence[str],
                required_mask: np.ndarray, point_times_s: np.ndarray,
                action_point_ids: Iterable[str], *, distance_m: np.ndarray,
                require_camera_pair: bool = True) -> ObservableSupport:
        ids = tuple(str(point_id) for point_id in point_ids)
        required = np.asarray(required_mask, dtype=bool)
        times = np.asarray(point_times_s, dtype=np.float64)
        distance = np.asarray(distance_m, dtype=np.float64)
        if len(set(ids)) != len(ids) or any(x.shape != (len(ids),) for x in (required, times, distance)):
            raise ValueError("unique point ids and equally sized masks/time/distance required")
        if not np.isfinite(times).all() or not np.isfinite(distance).all() or not np.isfinite(timestamp_s):
            raise ValueError("support times and distances must be finite")
        if np.any(times < 0):
            raise ValueError("future plan times cannot be negative")
        action_ids = set(map(str, action_point_ids))
        if not action_ids or not action_ids.issubset(ids):
            raise ValueError("action support must explicitly identify available point ids")
        lookup = {point_id: index for index, point_id in enumerate(ids)}
        now = np.zeros(len(ids), dtype=bool)
        source_now = now.copy()
        refs: list[list[EvidenceRef]] = [[] for _ in ids]
        current_cameras: set[str] = set()
        cutoff = timestamp_s - self.thresholds.maximum_history_s
        for frame in self._frames:
            ref = frame.reference
            # A caller querying an older decision cannot see a newer admitted
            # observation. Frame identity and timestamp must BOTH be causal.
            if ref.timestamp_s > timestamp_s + 1e-8 or ref.frame_id > decision_frame_id:
                continue
            if ref.timestamp_s < cutoff - 1e-8:
                continue
            is_current = ref.frame_id == decision_frame_id and abs(ref.timestamp_s - timestamp_s) < 1e-7
            if ref.frame_id == decision_frame_id and not is_current:
                raise ValueError("decision frame timestamp mismatch")
            if is_current:
                current_cameras.add(ref.camera)
            for local_index, point_id in enumerate(frame.point_ids):
                index = lookup.get(point_id)
                if index is None:
                    continue
                if frame.crop_visible[local_index]:
                    refs[index].append(ref)
                    now[index] |= is_current
                if is_current:
                    source_now[index] |= bool(frame.source_visible[local_index])
        if require_camera_pair and current_cameras != {"narrow", "wide"}:
            raise ValueError("both stock-camera evidence frames required at the decision timestamp")
        supported = np.array([bool(value) for value in refs], dtype=bool)
        durations = np.zeros(len(ids), dtype=np.float64)
        for index, point_refs in enumerate(refs):
            # Sum adjacent visible intervals, excluding gaps > one 20 Hz tick.
            # This is measured duration, not first-to-last elapsed wall time.
            visible_times = np.unique([ref.timestamp_s for ref in point_refs])
            differences = np.diff(visible_times)
            durations[index] = float(differences[(differences > 0) & (differences <= 0.050001)].sum())
        action_valid = all(supported[lookup[point_id]] for point_id in action_ids)
        required_supported = bool(np.all(supported[required]))
        def preview(mask: np.ndarray) -> float:
            return float(np.max(distance[mask & (distance > 0)], initial=0.0))
        current_preview = preview(now)
        history_preview = preview(supported)
        loss_mask = (required & supported & ~now & source_now & (distance > 0)
                     & (durations >= self.thresholds.minimum_history_visibility_s - 1e-8))
        preview_loss = max(0.0, preview(loss_mask) - current_preview)
        reasons: list[str] = []
        if not action_valid or not required_supported:
            category = "diagnostic_ambiguous"
            if not action_valid:
                reasons.append("immediate_action_geometry_has_no_admitted_pixel_evidence")
            if not required_supported:
                reasons.append("required_geometry_has_no_admitted_pixel_evidence")
        elif np.any(loss_mask) and preview_loss >= self.thresholds.minimum_crop_preview_loss_m:
            category = "history_crop_loss"
        elif np.any(required & supported & ~now):
            category = "history_supported"
        else:
            category = "current"
        # Report only a continuous supported prefix, never a far isolated point.
        order = np.argsort(times, kind="stable")
        horizon = 0.0
        for index in order:
            if not supported[index]:
                break
            horizon = max(horizon, float(times[index]))
        return ObservableSupport(ids, distance.copy(), times.copy(), required.copy(), supported, now, source_now,
                                 tuple(tuple(value) for value in refs), durations,
                                 current_preview, preview(source_now), history_preview,
                                 horizon, preview_loss, action_valid, category, tuple(reasons))
