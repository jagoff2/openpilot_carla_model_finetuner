"""Replayable visual evidence for stationary obstacles, without a simulator.

Payload contract: save each Observation's RGB, metric depth, semantic IDs,
instance IDs and *actual* packed model frame as numeric NPZ members (no pickle).
Save its FrameStamp, every modality's FrameStamp, camera matrices, depth-fixture
SHA256 and aligned ActorState alongside those arrays. ``build_static_proof``
hashes every input and retains the depth samples used, so ``verify_static_proof``
can replay the same claim from immutable capture artifacts.

Instance segmentation establishes correspondence, RGB establishes contrast,
qualified depth establishes visible surfaces, and native crop replay establishes
availability to the student. Semantic classes alone cannot identify one actor.
Actor transforms are used only to check stationarity. A simulator bounding box
may be supplied as a conservative QA envelope; its hidden faces are NEVER
claimed to have been observed. This module does not certify a safe passing path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
EVIDENCE_VERSION = "stationary_obstacle_rgb_depth_instance_crop_v1"
SEMANTIC_SCHEMA = "carla-0.9.16"
# Verified against the installed CARLA 0.9.16 CityObjectLabel enumeration.
SEMANTIC_IDS = {"parked_vehicle": (14, 15, 16), "pedestrian": (12,)}
MODALITIES = ("rgb", "depth", "semantic", "instance", "packed")


@dataclass(frozen=True)
class FrameStamp:
    frame_id: int
    timestamp_s: float
    camera: str

    def __post_init__(self):
        if (isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int)
                or self.frame_id < 0 or not np.isfinite(self.timestamp_s)
                or self.camera not in ("narrow", "wide")):
            raise ValueError("invalid frame identity")


@dataclass(frozen=True)
class Observation:
    stamp: FrameStamp
    modality_stamps: Mapping[str, FrameStamp]
    rgb: np.ndarray
    depth_m: np.ndarray
    semantic_ids: np.ndarray
    instance_ids: np.ndarray | None
    packed: np.ndarray
    intrinsics: np.ndarray
    camera_from_world: np.ndarray
    model_to_source: np.ndarray
    depth_mode: str
    depth_fixture_sha256: str
    depth_qualified: bool
    semantic_schema: str = SEMANTIC_SCHEMA
    pixel_center_offset: float = .5
    instance_bgra: np.ndarray | None = None
    scene_actor_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ActorState:
    """Same-frame simulator QA only, not student input or observed geometry."""
    stamp: FrameStamp
    actor_id: int
    position_world_m: tuple[float, float, float]
    yaw_rad: float
    velocity_world_mps: tuple[float, float, float]
    yaw_rate_rps: float
    source: str = "simulator_actor_state_qa"


@dataclass(frozen=True)
class EvidenceConfig:
    minimum_source_pixels: int = 24
    minimum_crop_pixels: int = 12
    minimum_pixel_span: int = 3
    minimum_rgb_contrast: float = 12.
    maximum_depth_m: float = 120.
    minimum_depth_fraction: float = .95
    maximum_surface_samples: int = 512
    footprint_tolerance_m: float = .10
    minimum_static_frames: int = 3
    minimum_static_history_s: float = .5
    maximum_history_gap_s: float = .25
    maximum_static_displacement_m: float = .03
    maximum_static_speed_mps: float = .05
    maximum_static_heading_change_rad: float = .01
    maximum_static_yaw_rate_rps: float = .01

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("evidence thresholds must be finite and positive")
        for key in ("minimum_source_pixels", "minimum_crop_pixels", "minimum_pixel_span",
                    "maximum_surface_samples", "minimum_static_frames"):
            if not isinstance(getattr(self, key), int):
                raise ValueError("pixel, sample and frame counts must be integers")
        if self.minimum_depth_fraction > 1 or self.minimum_static_frames < 3:
            raise ValueError("invalid depth fraction or stationary history length")


def array_sha256(value):
    value = np.asarray(value)
    if value.dtype.hasobject:
        raise ValueError("object arrays are forbidden")
    header = json.dumps([value.dtype.str, list(value.shape)], separators=(",", ":")).encode()
    return hashlib.sha256(header+b"\0"+np.ascontiguousarray(value).tobytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def decode_carla_instance_bgra(raw):
    """Decode CARLA 0.9.16 Tagger.cpp: R=class, G=ID low, B=ID high.

    IDs are only 16 bits in the sensor. The observation therefore also binds a
    complete scene actor-ID inventory and rejects any ID above 65535, rather
    than silently aliasing two simulator actors onto the same pixel identity.
    """
    raw = np.asarray(raw)
    if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[-1] != 4:
        raise ValueError("raw instance segmentation must be uint8 HxWx4 BGRA")
    return raw[..., 2].copy(), raw[..., 1].astype(np.uint16)+256*raw[..., 0].astype(np.uint16)


def _lookup(matrix, shape, output_size=(512, 256)):
    """Exact float32 native nearest-neighbour lookup; unclamped validity retained."""
    matrix = np.asarray(matrix, np.float32)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("invalid model-to-source matrix")
    w, h = output_size
    x = np.broadcast_to(np.arange(w, dtype=np.float32), (h, w))
    y = np.broadcast_to(np.arange(h, dtype=np.float32)[:, None], (h, w))
    def dot(row):
        yp = np.asarray(row[1]*y, np.float32)
        return np.asarray(np.asarray(np.float64(row[0])*x.astype(np.float64)
            +yp.astype(np.float64), np.float32)+row[2], np.float32)
    denominator = dot(matrix[2])
    if np.any(denominator == 0):
        raise ValueError("model projection has zero denominator")
    reciprocal = np.asarray(1./denominator.astype(np.float64), np.float32)
    sx = np.rint(np.asarray(dot(matrix[0])*reciprocal, np.float32))
    sy = np.rint(np.asarray(dot(matrix[1])*reciprocal, np.float32))
    if not np.isfinite(sx).all() or not np.isfinite(sy).all():
        raise ValueError("nonfinite model projection")
    inside = (sx >= 0) & (sy >= 0) & (sx < shape[1]) & (sy < shape[0])
    return (np.clip(sx, 0, shape[1]-1).astype(np.intp),
            np.clip(sy, 0, shape[0]-1).astype(np.intp), inside)


def pack_model_rgb(rgb, model_to_source):
    """Pinned BT.601 limited RGB producer plus native MED/SBIG warp and packing.

    This reproduces the existing collector's source-RGB contract; it does not
    assert equivalence to a physical device's camera colour conversion.
    """
    rgb = np.asarray(rgb)
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
            or min(rgb.shape[:2]) < 2 or any(v % 2 for v in rgb.shape[:2])):
        raise ValueError("RGB must be uint8 HxWx3 with positive even dimensions")
    h, w = rgb.shape[:2]
    r, g, b = rgb.astype(np.float64).transpose(2, 0, 1)
    y = np.clip(np.rint(16.+.256788*r+.504129*g+.097906*b), 16, 235).astype(np.uint8)
    u = 128.-.148223*r-.290993*g+.439216*b
    v = 128.+.439216*r-.367788*g-.071427*b
    u, v = (np.clip(np.rint(c.reshape(h//2, 2, w//2, 2).mean(axis=(1, 3))),
                       16, 240).astype(np.uint8) for c in (u, v))
    ix, iy, _ = _lookup(model_to_source, y.shape)
    y = y[iy, ix]
    uv_matrix = np.asarray(model_to_source, np.float32)*np.array(
        [[1., 1., .5], [1., 1., .5], [2., 2., 1.]], np.float32)
    ix, iy, _ = _lookup(uv_matrix, u.shape, (256, 128))
    return np.stack((y[::2, ::2], y[1::2, ::2], y[::2, 1::2], y[1::2, 1::2],
                     u[iy, ix], v[iy, ix]))


def _rgb_contrast(rgb, mask):
    surrounding = mask.copy()
    for _ in range(3):
        pad = np.pad(surrounding, 1)
        surrounding = np.logical_or.reduce([pad[dy:dy+mask.shape[0], dx:dx+mask.shape[1]]
            for dy in range(3) for dx in range(3)])
    ring = surrounding & ~mask
    if not mask.any() or not ring.any():
        return 0.
    return float(np.linalg.norm(np.median(rgb[mask].astype(float), axis=0)
                                -np.median(rgb[ring].astype(float), axis=0)))


def _polygon(value):
    polygon = np.asarray(value, float)
    if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
            or not np.isfinite(polygon).all()):
        raise ValueError("footprint must be a finite convex polygon")
    edge = np.roll(polygon, -1, axis=0)-polygon
    cross = edge[:, 0]*np.roll(edge, -1, axis=0)[:, 1]-edge[:, 1]*np.roll(edge, -1, axis=0)[:, 0]
    if not (np.all(cross > 1e-10) or np.all(cross < -1e-10)):
        raise ValueError("footprint must be ordered, convex and nondegenerate")
    if np.all(cross < 0):
        polygon = polygon[::-1]
    # Convexity against every edge also excludes star polygons.
    edge = np.roll(polygon, -1, axis=0)-polygon
    delta = polygon[:, None]-polygon[None, :]
    if np.any(edge[None, :, 0]*delta[..., 1]-edge[None, :, 1]*delta[..., 0] < -1e-9):
        raise ValueError("footprint self-intersection or nonconvexity")
    return polygon


def visible_surface_points(observation, source_pixels_xy):
    """Backproject sampled instance pixels, never actor bounding-box vertices."""
    pixels = np.asarray(source_pixels_xy)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or pixels.dtype.kind not in "iu":
        raise ValueError("surface pixels must be integer Nx2")
    depth = np.asarray(observation.depth_m)
    if (np.any(pixels < 0) or np.any(pixels[:, 0] >= depth.shape[1])
            or np.any(pixels[:, 1] >= depth.shape[0])):
        raise ValueError("surface pixel outside source image")
    k = np.asarray(observation.intrinsics, float)
    transform = np.asarray(observation.camera_from_world, float)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1]) or transform.shape != (4, 4)
            or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3]@transform[:3, :3].T, np.eye(3), atol=1e-6)
            or abs(abs(np.linalg.det(transform[:3, :3]))-1) > 1e-6):
        raise ValueError("invalid calibrated camera matrices")
    if observation.pixel_center_offset not in (0., .5):
        raise ValueError("unknown source pixel center convention")
    rays = np.c_[pixels+observation.pixel_center_offset, np.ones(len(pixels))]@np.linalg.inv(k).T
    values = depth[pixels[:, 1], pixels[:, 0]].astype(float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("nonfinite or nonpositive surface depth")
    if observation.depth_mode == "radial":
        rays /= np.linalg.norm(rays, axis=1)[:, None]
    elif observation.depth_mode != "forward":
        raise ValueError("depth convention must be qualified forward or radial")
    optical = rays*values[:, None]
    inverse = np.linalg.inv(transform)
    return optical@inverse[:3, :3].T+inverse[:3, 3]


def observation_proof(observation, *, actor_id, instance_id, category,
                      conservative_footprint_world_xy, config=None):
    """Prove current visible surface/crop correspondence; no static claim yet."""
    cfg = config or EvidenceConfig()
    if category not in SEMANTIC_IDS or observation.semantic_schema != SEMANTIC_SCHEMA:
        raise ValueError("unsupported obstacle category or semantic schema")
    if (any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (actor_id, instance_id))
            or actor_id != instance_id or instance_id > 65535):
        raise ValueError("actor and nonzero 16-bit instance identity required")
    scene_ids = observation.scene_actor_ids
    if (not scene_ids or any(isinstance(v, bool) or not isinstance(v, int) or not 0 < v <= 65535
                            for v in scene_ids) or len(set(scene_ids)) != len(scene_ids)
            or actor_id not in scene_ids):
        raise ValueError("complete unambiguous 16-bit scene actor inventory required")
    if (set(observation.modality_stamps) != set(MODALITIES)
            or any(observation.modality_stamps[k] != observation.stamp for k in MODALITIES)):
        raise ValueError("camera modalities must have the exact same frame, time and camera")
    fixture = observation.depth_fixture_sha256
    if (observation.depth_qualified is not True or not isinstance(fixture, str)
            or len(fixture) != 64 or any(c not in "0123456789abcdef" for c in fixture)):
        raise ValueError("qualified depth fixture SHA256 required")
    rgb, depth = np.asarray(observation.rgb), np.asarray(observation.depth_m)
    if observation.instance_ids is None:
        raise ValueError("semantic-only evidence cannot bind an individual actor")
    semantic, instance = np.asarray(observation.semantic_ids), np.asarray(observation.instance_ids)
    if (depth.shape != rgb.shape[:2] or depth.dtype.kind != "f"
            or semantic.shape != depth.shape or instance.shape != depth.shape
            or semantic.dtype.kind not in "iu" or instance.dtype.kind not in "iu"):
        raise ValueError("RGB, metric depth, semantic and instance images must be aligned")
    raw_semantic, raw_instance = decode_carla_instance_bgra(observation.instance_bgra)
    if not np.array_equal(raw_instance, instance) or not np.array_equal(raw_semantic, semantic):
        raise ValueError("decoded instance or semantic image disagrees with raw instance sensor")
    expected = pack_model_rgb(rgb, observation.model_to_source)
    packed = np.asarray(observation.packed)
    if packed.dtype != np.uint8 or packed.shape != (6, 128, 256) or not np.array_equal(packed, expected):
        raise ValueError("actual packed model input differs from RGB and native crop replay")
    mask = instance == instance_id
    if np.any(mask & ~np.isin(semantic, SEMANTIC_IDS[category])):
        raise ValueError("instance mask conflicts with the declared semantic category")
    ix, iy, inside = _lookup(observation.model_to_source, depth.shape)
    crop_mask = mask[iy, ix] & inside
    crop_rgb = rgb[iy, ix]
    reasons = []
    for name, image, selected, minimum in (("source", rgb, mask, cfg.minimum_source_pixels),
            ("crop", crop_rgb, crop_mask, cfg.minimum_crop_pixels)):
        if int(selected.sum()) < minimum:
            reasons.append(name+"_instance_unresolved")
        yy, xx = np.nonzero(selected)
        if not len(xx) or min(np.ptp(xx)+1, np.ptp(yy)+1) < cfg.minimum_pixel_span:
            reasons.append(name+"_instance_span_unresolved")
        if _rgb_contrast(image, selected) < cfg.minimum_rgb_contrast:
            reasons.append(name+"_rgb_contrast_unresolved")
    good_depth = np.isfinite(depth) & (depth > 0) & (depth <= cfg.maximum_depth_m)
    if not mask.any() or float(good_depth[mask].mean()) < cfg.minimum_depth_fraction:
        reasons.append("source_instance_depth_unresolved")
    crop_good = crop_mask & good_depth[iy, ix]
    if not crop_mask.any() or float(crop_good.sum()/max(1, crop_mask.sum())) < cfg.minimum_depth_fraction:
        reasons.append("crop_instance_depth_unresolved")
    # Only depth surfaces actually represented in the model crop are admitted.
    source_pixels = np.unique(np.c_[ix[crop_good], iy[crop_good]], axis=0)
    if len(source_pixels) > cfg.maximum_surface_samples:
        source_pixels = source_pixels[np.linspace(0, len(source_pixels)-1,
            cfg.maximum_surface_samples, dtype=int)]
    polygon = _polygon(conservative_footprint_world_xy)
    surface = visible_surface_points(observation, source_pixels)
    edge = np.roll(polygon, -1, axis=0)-polygon
    delta = surface[:, None, :2]-polygon[None, :, :]
    margin = (edge[None, :, 0]*delta[..., 1]-edge[None, :, 1]*delta[..., 0])/np.linalg.norm(edge, axis=1)
    if len(surface) and np.any(margin < -cfg.footprint_tolerance_m):
        reasons.append("visible_surface_outside_conservative_qa_footprint")
    arrays = {name: array_sha256(getattr(observation, name)) for name in
        ("rgb", "depth_m", "semantic_ids", "instance_ids", "instance_bgra", "packed", "intrinsics",
         "camera_from_world", "model_to_source")}
    return dict(stamp=asdict(observation.stamp), modality_stamps={k:asdict(v)
        for k,v in observation.modality_stamps.items()}, actor_id=actor_id,
        instance_id=instance_id, scene_actor_ids=list(scene_ids), category=category,
        supported=not reasons, reasons=sorted(set(reasons)),
        semantic_schema=observation.semantic_schema, array_sha256=arrays,
        depth_fixture_sha256=fixture, depth_mode=observation.depth_mode,
        pixel_center_offset=observation.pixel_center_offset, source_pixels=int(mask.sum()),
        crop_pixels=int(crop_mask.sum()), unique_crop_source_pixels=len(source_pixels),
        source_rgb_contrast=_rgb_contrast(rgb, mask), crop_rgb_contrast=_rgb_contrast(crop_rgb, crop_mask),
        visible_surface_source_pixels_xy=source_pixels.tolist(),
        visible_surface_world_xyz=surface.tolist(), conservative_qa_footprint_world_xy=polygon.tolist(),
        entire_footprint_observed=False)


def build_static_proof(observations: Sequence[Observation], actor_states: Sequence[ActorState], *,
                       actor_id: int, instance_id: int, category: str,
                       conservative_footprint_world_xy, decision: FrameStamp, config=None):
    """Qualify visible stationary presence, or return a yield decision.

    The final observation must be the decision frame; every history observation
    must pass source/crop replay. States bind the same actor and exact timestamps.
    Stationarity is a measured historical fact, not a prediction that a person
    will remain still. No future state may enter this proof.
    """
    cfg = config or EvidenceConfig()
    observations, actor_states = list(observations), list(actor_states)
    reasons, records = [], []
    if len(observations) < cfg.minimum_static_frames:
        reasons.append("stationary_history_insufficient")
    if not observations or observations[-1].stamp != decision:
        reasons.append("current_decision_frame_missing")
    if len(observations) != len(actor_states):
        reasons.append("aligned_actor_state_history_missing")
    times = np.array([o.stamp.timestamp_s for o in observations])
    frames = np.array([o.stamp.frame_id for o in observations])
    if (any(o.stamp.camera != decision.camera for o in observations)
            or np.any(times > decision.timestamp_s) or np.any(frames > decision.frame_id)
            or np.any(np.diff(times) <= 0) or np.any(np.diff(frames) <= 0)):
        reasons.append("future_mixed_camera_or_unordered_history")
    if len(times) < 2 or times[-1]-times[0] < cfg.minimum_static_history_s-1e-9:
        reasons.append("stationary_history_too_short")
    # CARLA's 20 Hz delta originates as float32; five ticks can be
    # 0.2500000037 s. Match the collector's microsecond timestamp tolerance.
    if np.any(np.diff(times) > cfg.maximum_history_gap_s+1e-6):
        reasons.append("stationary_history_gap")
    for observation in observations:
        try:
            record = observation_proof(observation, actor_id=actor_id, instance_id=instance_id,
                category=category, conservative_footprint_world_xy=conservative_footprint_world_xy, config=cfg)
            records.append(record)
            reasons.extend(record["reasons"])
        except (ValueError, np.linalg.LinAlgError) as error:
            reasons.append("observation_invalid:"+str(error))
    positions, headings = [], []
    for observation, state in zip(observations, actor_states):
        if (state.stamp != observation.stamp or state.actor_id != actor_id
                or state.source != "simulator_actor_state_qa"):
            reasons.append("actor_stationarity_identity_mismatch")
            continue
        position = np.asarray(state.position_world_m, float)
        velocity = np.asarray(state.velocity_world_mps, float)
        if (position.shape != (3,) or velocity.shape != (3,) or not np.isfinite(position).all()
                or not np.isfinite(velocity).all() or not np.isfinite([state.yaw_rad, state.yaw_rate_rps]).all()):
            reasons.append("actor_stationarity_state_invalid")
            continue
        positions.append(position); headings.append(state.yaw_rad)
        if (np.linalg.norm(velocity) > cfg.maximum_static_speed_mps
                or abs(state.yaw_rate_rps) > cfg.maximum_static_yaw_rate_rps):
            reasons.append("actor_moving_yield")
    if positions and np.max(np.linalg.norm(np.asarray(positions)-positions[0], axis=1)) > cfg.maximum_static_displacement_m:
        reasons.append("actor_displacement_yield")
    if headings and np.max(np.abs((np.asarray(headings)-headings[0]+np.pi) % (2*np.pi)-np.pi)) > cfg.maximum_static_heading_change_rad:
        reasons.append("actor_rotation_yield")
    result = dict(schema_version=SCHEMA_VERSION, evidence_version=EVIDENCE_VERSION,
        decision=asdict(decision), actor_id=actor_id, instance_id=instance_id, category=category,
        supported=not reasons, outcome="static_obstacle_observed" if not reasons else "yield",
        reasons=sorted(set(reasons)), config=asdict(cfg), observations=records,
        actor_states=[asdict(state) for state in actor_states],
        scope="observed stationary surface and crop availability; safe passing path not certified",
        entire_footprint_observed=False, safe_to_pass=False,
        geometry_source="visible_metric_depth_surface", footprint_source="conservative_simulator_qa_envelope",
        future_motion_assumed=False)
    # Invalid numeric metadata must fail closed, never emit nonstandard JSON.
    try:
        result["proof_sha256"] = _digest(result)
    except ValueError as error:
        raise ValueError("nonfinite proof metadata") from error
    return result


def verify_static_proof(proof, observations, actor_states, **kwargs):
    """Replay all artifacts and require exact deterministic proof equality."""
    replayed = build_static_proof(observations, actor_states, **kwargs)
    if _digest(proof) != _digest(replayed):
        raise ValueError("static obstacle proof does not match replayed capture artifacts")
    return replayed
