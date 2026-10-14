"""Reconstruct training geometry exclusively from admitted, observed boundary knots.

Geometry is a target, never a camera-model input. No recorded speed, future
steering command, hidden centerline coordinate or hidden map derivative is used.
Accepted shards remain immutable; this adapter also leaves the live capture
package's qualification hashes unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from lane_dataset.observed_corridor import METHOD

PAIRED_METHOD = "cubic_interpolation_of_rendered_paired_boundary_knots_no_hidden_map_derivatives"


@dataclass(frozen=True)
class ObservedCorridor:
    """Ordered FRD XY boundaries with interval support and current route station."""
    s: np.ndarray
    left: np.ndarray
    right: np.ndarray
    valid: np.ndarray
    ego_s: float
    length: float = 4.8
    width: float = 2.2
    bbox_center: tuple = (0.0, 0.0)
    bbox_yaw: float = 0.0
    label_index: int = -1
    manifest_sha256: str = ""

    def __post_init__(self):
        n = len(self.s)
        if (n < 2 or self.s.shape != (n,) or self.left.shape != (n, 2)
                or self.right.shape != (n, 2) or self.valid.shape != (n-1,)
                or self.valid.dtype != np.bool_):
            raise ValueError("invalid corridor arrays")
        if (not all(np.isfinite(x).all() for x in (self.s, self.left, self.right))
                or np.any(np.diff(self.s) <= 0) or not self.s[0] <= self.ego_s <= self.s[-1]
                or not np.isfinite([self.length, self.width, *self.bbox_center, self.bbox_yaw]).all()
                or min(self.length, self.width) <= 0):
            raise ValueError("invalid corridor geometry or vehicle dimensions")
        tangent = np.diff((self.left+self.right)/2, axis=0)
        across = ((self.right-self.left)[:-1]+(self.right-self.left)[1:])/2
        if np.any(tangent[:, 0]*across[:, 1]-tangent[:, 1]*across[:, 0] <= 0):
            raise ValueError("boundaries must be ordered left to right in FRD")

    def sample(self, query):
        """Linear dense interpolation; out-of-range queries are explicitly masked."""
        query = np.asarray(query, dtype=float)
        idx = np.searchsorted(self.s, query, side="right")-1
        idx = np.clip(idx, 0, len(self.s)-2)
        fraction = np.clip((query-self.s[idx])/(self.s[idx+1]-self.s[idx]), 0, 1)
        left = self.left[idx]+fraction[..., None]*(self.left[idx+1]-self.left[idx])
        right = self.right[idx]+fraction[..., None]*(self.right[idx+1]-self.right[idx])
        center = (left+right)/2
        centers = (self.left+self.right)/2
        tangent = centers[idx+1]-centers[idx]
        tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
        valid = self.valid[idx] & np.isfinite(query) & (query >= self.s[0]) & (query <= self.s[-1])
        return center, tangent, left, right, valid


def _indices(value, n):
    index = np.asarray(value)
    if (index.ndim != 1 or index.dtype.kind not in "iu" or len(index) < 4
            or np.any(np.diff(index) <= 0) or index[0] < 0 or index[-1] >= n):
        raise ValueError("invalid observed knot indices")
    return index


def corridor_from_record(geometry, record, position, rotation, vehicle, frame_times,
                         *, manifest_sha256="", spacing_m=.25):
    """Pure target builder; the caller binds the record and artifacts to a manifest."""
    if not np.isfinite(spacing_m) or not 0 < spacing_m <= .25:
        raise ValueError("target spacing must be in (0, .25] metres")
    if not record.get("accepted") or not record.get("qa_passed"):
        raise ValueError("centerline targets require an accepted physical label")
    proof = record["interpolated_corridor"]
    left_all, right_all = (np.asarray(geometry[key], float) for key in ("left_xyz", "right_xyz"))
    if left_all.shape != right_all.shape or left_all.ndim != 2 or left_all.shape[1] != 3:
        raise ValueError("invalid boundary coordinate arrays")
    n = len(left_all)
    grid = _indices(proof["source_indices"], n)
    independent = proof["support_method"] == METHOD
    if not independent and proof["support_method"] != PAIRED_METHOD:
        raise ValueError("unsupported observed corridor reconstruction")
    li = _indices(proof["source_left_indices"], n) if independent else grid
    ri = _indices(proof["source_right_indices"], n) if independent else grid
    if grid[0] < max(li[0], ri[0]) or grid[-1] > min(li[-1], ri[-1]):
        raise ValueError("boundary extrapolation is forbidden")
    source = np.r_[li, ri+n]
    visibility = record["visibility"]
    supported = np.asarray(visibility["supported"])
    if supported.shape != (2*n,) or supported.dtype != np.bool_ or not supported[source].all():
        raise ValueError("unobserved source knot")
    timestamp = float(record["timestamp_s"])
    refs = []
    for point in source:
        point_refs = visibility["evidence_frames"][point]
        if not point_refs:
            raise ValueError("source knot lacks observation evidence")
        refs.extend(point_refs)
    for ref in refs + proof["spline_evidence_frames"]:
        observed_time = float(ref["timestamp_s"])
        if (ref["camera"] not in ("narrow", "wide") or ref["frame_id"] not in frame_times
                or not np.isfinite(observed_time) or observed_time > timestamp+1e-6
                or observed_time < timestamp-5.0-1e-6
                or abs(frame_times[ref["frame_id"]]-observed_time) > 1e-6):
            raise ValueError("future, stale or unbound observation evidence")
    gap = float(proof["maximum_marking_gap_m"])
    if not 0 < gap <= 8.0:
        raise ValueError("invalid observation gap limit")
    for all_points, indices in ((left_all, li), (right_all, ri)):
        if not np.isfinite(all_points[indices]).all():
            raise ValueError("nonfinite observed geometry")
        if np.any(np.linalg.norm(np.diff(all_points[indices], axis=0), axis=1) > gap+1e-6):
            raise ValueError("observed boundary gap exceeds qualification limit")
    # Only observed XYZ values are evaluated. Neither the hidden road arc nor
    # the full-map center/curvature arrays enter the interpolation.
    left = CubicSpline(li, left_all[li], axis=0)(grid) if independent else left_all[grid]
    right = CubicSpline(ri, right_all[ri], axis=0)(grid) if independent else right_all[grid]
    chord = np.r_[0., np.cumsum(np.linalg.norm(np.diff((left+right)/2, axis=0), axis=1))]
    if not np.allclose(chord, proof["chord_arc_m"], atol=1e-5, rtol=1e-6):
        raise ValueError("observed reconstruction disagrees with admission")
    errors = np.asarray(proof["interval_geometry_disagreement_m"], float)
    allowed = np.asarray(proof["interval_geometry_valid"])
    if errors.shape != (len(grid)-1,) or allowed.shape != errors.shape or allowed.dtype != np.bool_:
        raise ValueError("invalid interval qualification")
    allowed = allowed & np.isfinite(errors) & (errors <= .05)
    # Include every knot so a sample never bridges an unsupported interval.
    # Subdivide each original interval separately. Merging two floating-point
    # grids can otherwise create adjacent stations only 1e-13 metres apart.
    query = np.concatenate([np.linspace(a, b, max(1, int(np.ceil((b-a)/spacing_m)))+1)[:-1]
        for a,b in zip(chord[:-1],chord[1:])] + [chord[-1:]])
    left = CubicSpline(chord, left, axis=0, extrapolate=False)(query)
    right = CubicSpline(chord, right, axis=0, extrapolate=False)(query)
    inverse = Rotation.from_euler("xyz", np.asarray(rotation, float)).inv()
    reflection = np.array([1., 1., -1.])
    left = inverse.apply((left-np.asarray(position))*reflection)[:, :2]
    right = inverse.apply((right-np.asarray(position))*reflection)[:, :2]
    center = (left+right)/2
    segment = np.diff(center, axis=0)
    distance = np.linalg.norm(segment, axis=1)
    if np.any(distance <= 1e-9):
        raise ValueError("degenerate observed corridor")
    s = np.r_[0., np.cumsum(distance)]
    fraction = np.clip(-(center[:-1]*segment).sum(1)/distance**2, 0, 1)
    closest = np.argmin(np.linalg.norm(center[:-1]+fraction[:, None]*segment, axis=1))
    ego_s = s[closest]+fraction[closest]*distance[closest]
    intervals = np.searchsorted(chord, (query[:-1]+query[1:])/2, side="right")-1
    bbox_rotation = np.asarray(vehicle.get("bbox_rotation_degrees", [0, 0, 0]), float)
    if np.any(np.abs(bbox_rotation[:2]) > 1e-4):
        raise ValueError("pitched/rolled bounding boxes require a 3D footprint adapter")
    return ObservedCorridor(s, left, right, allowed[intervals], float(ego_s),
        float(vehicle["length"]), float(vehicle["width"]),
        tuple(vehicle.get("bbox_center", [0., 0.])[:2]), np.deg2rad(bbox_rotation[2]),
        int(record["label_index"]), manifest_sha256)


def episode_corridors(episode):
    """Read targets from a verified accepted SequenceEpisode without modifying it."""
    if episode.manifest.get("status") != "accepted" or not episode.manifest.get("admission", {}).get("passed"):
        raise ValueError("direct geometry supervision requires independently admitted data")
    # Verify again because callers can construct SequenceEpisode(verify_hashes=False).
    episode._verify_file_inventory(True)
    for member in ("geometry.json", "telemetry.npz"):
        if member not in episode.manifest["files"]:
            raise ValueError("geometry and telemetry must be manifest-bound")
    manifest_hash = hashlib.sha256(episode.path.read_bytes()).hexdigest()
    geometry = json.loads(episode._path("geometry.json").read_text(encoding="utf-8"))
    times = dict(zip(episode.frame_ids.tolist(), episode.timestamps.tolist()))
    records = {int(r["label_index"]): r for r in episode.manifest["label_records"]}
    with np.load(episode._path("telemetry.npz"), allow_pickle=False) as telemetry:
        if (not np.array_equal(telemetry["frame_id"], episode.frame_ids)
                or not np.allclose(telemetry["timestamp"], episode.timestamps, atol=1e-6, rtol=0)):
            raise ValueError("geometry pose timestamps disagree with camera sequence")
        result = []
        for index in episode.labels["label_indices"]:
            record = records[int(index)]
            if abs(float(record["timestamp_s"])-episode.timestamps[index]) > 1e-6:
                raise ValueError("label timestamp disagrees with camera sequence")
            result.append(corridor_from_record(geometry, record, telemetry["position_world"][index],
                telemetry["rotation_world"][index], episode.manifest["vehicle_dimensions"], times,
                manifest_sha256=manifest_hash))
    return result
