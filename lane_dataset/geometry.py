"""Lane geometry in CARLA world coordinates; no simulator dependency.

World XY and yaw follow CARLA (positive yaw turns +X toward +Y); Z is up.
Comma local positions are forward/right/down. Full map references are validation
data. Only independently reconstructed ObservableReference objects enter MPC.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

# CARLA's rendered paired dashed markings have measured gaps up to 7.51 m
# in the Town04 development trace. This is an observation-model limit, not
# permission to use hidden knots: capture uses only visible boundary pairs,
# and admission independently requires <=5 cm interpolation disagreement.
MAX_OBSERVED_KNOT_GAP_M = 8.0


def wrap_angle(value):
    return (np.asarray(value) + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class EvidenceFrame:
    frame_id: int | str
    timestamp_s: float
    camera: str = "narrow"


class LaneReference:
    """Corresponding boundaries sampled at the same road arc coordinates.

    Boundary arrays must correspond by road arc, not closest-neighbor pairing.
    ``left`` and ``right`` refer to the direction of traversal. For a +X road,
    left is negative Y, matching CARLA's left-handed world convention.
    """

    def __init__(self, left_xyz, right_xyz, arc_length_m=None):
        left, right = np.asarray(left_xyz, float), np.asarray(right_xyz, float)
        if left.ndim != 2 or left.shape != right.shape or left.shape[1] != 3 or len(left) < 4:
            raise ValueError("paired boundaries require matching [N>=4,3] arrays")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("nonfinite lane boundaries")
        center = (left + right) * 0.5
        s = (np.r_[0., np.cumsum(np.linalg.norm(np.diff(center, axis=0), axis=1))]
             if arc_length_m is None else np.asarray(arc_length_m, float))
        if s.shape != (len(left),) or not np.isfinite(s).all() or np.any(np.diff(s) <= 1e-6):
            raise ValueError("arc coordinates must be finite and strictly increasing")
        if np.any(np.linalg.norm(left - right, axis=1) < 1):
            raise ValueError("degenerate lane width")
        tangent = np.gradient(center[:, :2], s, axis=0)
        cross = tangent[:, 0] * (right[:, 1] - left[:, 1]) - tangent[:, 1] * (right[:, 0] - left[:, 0])
        if np.any(cross <= 0):
            raise ValueError("boundaries swapped or traversal reversed without exchanging boundaries")
        self.left_xyz, self.right_xyz, self.center_xyz, self.s = (v.copy() for v in (left, right, center, s))
        for value in (self.left_xyz, self.right_xyz, self.center_xyz, self.s):
            value.flags.writeable = False
        self._center = CubicSpline(s, center, axis=0)
        self._left = CubicSpline(s, left, axis=0)
        self._right = CubicSpline(s, right, axis=0)

    @classmethod
    def from_boundaries(cls, left_xyz, right_xyz, arc_length_m=None):
        return cls(left_xyz, right_xyz, arc_length_m)

    def sample(self, s):
        q = np.asarray(s, float)
        if np.any(q < self.s[0] - 1e-8) or np.any(q > self.s[-1] + 1e-8):
            raise ValueError("reference query outside admitted arc support")
        xyz = self._center(q)
        first, second = self._center(q, 1), self._center(q, 2)
        yaw = np.arctan2(first[..., 1], first[..., 0])
        kappa = ((first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]) /
                 np.maximum(np.linalg.norm(first[..., :2], axis=-1) ** 3, 1e-12))
        width = np.linalg.norm(self._right(q) - self._left(q), axis=-1)
        return xyz, yaw, kappa, width

    def project(self, xy, hint_s=None):
        """Nearest centerline point, optionally restricted around known progress.

        A progress hint disambiguates overlapping paths/hairpins. No fixed-X
        inversion is used, so time-parametric trajectories can turn >90 degrees.
        """
        xy = np.asarray(xy, float)[:2]
        distances = np.sum((self.center_xyz[:, :2] - xy) ** 2, axis=1)
        if hint_s is not None:
            distances = np.where(np.abs(self.s - hint_s) <= 10., distances, np.inf)
        index = int(np.argmin(distances))
        lo, hi = self.s[max(0, index - 2)], self.s[min(len(self.s)-1, index + 2)]
        fitted = minimize_scalar(lambda q: float(np.sum((self._center(q)[:2] - xy) ** 2)),
                                 bounds=(lo, hi), method="bounded", options={"xatol": 1e-9})
        q = float(np.clip(fitted.x, self.s[0], self.s[-1]))
        center, yaw, _, _ = self.sample(q)
        right = np.array([-np.sin(yaw), np.cos(yaw)])
        offset = float(np.dot(xy - center[:2], right))
        return q, offset

    def admit(self, supported, evidence_frames: Sequence[Sequence], decision_time_s: float,
              start_index: int = 0):
        """Copy a contiguous supported run, then recompute all derivatives.

        Crucially, we never slice splines fit against hidden map points. Altering
        any unadmitted road geometry cannot influence these controller inputs.
        Visibility admission is independently established by the pixel/depth
        pipeline; this method enforces timestamp and reference integrity.
        """
        mask = np.asarray(supported, bool)
        if mask.shape != (len(self.s),) or len(evidence_frames) != len(self.s):
            raise ValueError("support/evidence length differs from boundary samples")
        stop = start_index
        while stop < len(mask) and mask[stop]:
            stop += 1
        if start_index < 0 or stop - start_index < 4:
            raise ValueError("insufficient contiguous observed lane support")
        admitted = []
        for point_refs in evidence_frames[start_index:stop]:
            refs = []
            for ref in point_refs:
                item = EvidenceFrame(**ref) if isinstance(ref, dict) else EvidenceFrame(
                    ref.frame_id, ref.timestamp_s, ref.camera)
                if not np.isfinite(item.timestamp_s) or item.timestamp_s > decision_time_s + 1e-8:
                    raise ValueError("future/nonfinite image evidence cannot support a decision")
                if item.camera not in ("narrow", "wide", "img", "big_img"):
                    raise ValueError("evidence must refer to a stock model camera crop")
                refs.append(item)
            if not refs:
                raise ValueError("supported road sample has no evidence-frame reference")
            admitted.append(tuple(refs))
        return ObservableReference(self.left_xyz[start_index:stop], self.right_xyz[start_index:stop],
                                   self.s[start_index:stop], tuple(admitted), decision_time_s)


class ObservableReference(LaneReference):
    def __init__(self, left_xyz, right_xyz, arc_length_m, evidence_frames, decision_time_s):
        super().__init__(left_xyz, right_xyz, arc_length_m)
        if len(evidence_frames)!=len(self.s) or not np.isfinite(decision_time_s):
            raise ValueError("observable reference requires evidence for every road sample")
        for refs in evidence_frames:
            if not refs or any(not np.isfinite(ref.timestamp_s) or ref.timestamp_s>decision_time_s+1e-8 for ref in refs):
                raise ValueError("observable reference contains missing or future evidence")
        self.evidence_frames = tuple(evidence_frames)
        self.decision_time_s = float(decision_time_s)

    def evidence_for_arc(self, lo: float, hi: float):
        # Include neighboring interpolation knots in each evidence attribution.
        first = max(0, int(np.searchsorted(self.s, lo)) - 1)
        last = min(len(self.s), int(np.searchsorted(self.s, hi, side="right")) + 1)
        return tuple(dict.fromkeys(ref for row in self.evidence_frames[first:last] for ref in row))


def recovery_reference(reference: ObservableReference, xy, yaw_rad: float,
                       speed_mps: float, yaw_rate_rps: float, duration_s=4., step_s=.05):
    """A quintic lateral recovery initialized at recorded position/yaw/yaw rate.

    This is a controller reference, never a replacement for recorded physical
    labels. Position, tangent and approximate curvature match the initial state;
    the endpoint matches lane center with zero lateral slope and curvature.
    """
    if speed_mps <= .1 or duration_s <= 0:
        raise ValueError("recovery requires forward motion and positive duration")
    s0, offset = reference.project(xy)
    _, lane_yaw, curvature, _ = reference.sample(s0)
    length = speed_mps * duration_s
    if s0 + length > reference.s[-1]:
        raise ValueError("recovery endpoint lacks admitted evidence")
    d1 = float(np.tan(wrap_angle(yaw_rad - lane_yaw)))
    d2 = yaw_rate_rps / speed_mps - float(curvature)
    coefficients = np.zeros(6)
    coefficients[:3] = [offset, d1 * length, .5 * d2 * length ** 2]
    coefficients[3:] = np.linalg.solve(np.array([[1,1,1],[3,4,5],[6,12,20]], float),
        -np.array([coefficients[:3].sum(), coefficients[1]+2*coefficients[2], 2*coefficients[2]]))
    t = np.arange(0, duration_s + step_s*.5, step_s)
    u = t / duration_s
    displacement = np.polynomial.polynomial.polyval(u, coefficients)
    xyz, yaw, _, _ = reference.sample(s0 + speed_mps*t)
    xyz[:, 0] -= displacement*np.sin(yaw)
    xyz[:, 1] += displacement*np.cos(yaw)
    xyz[0, :2] = np.asarray(xy)[:2]
    return t, xyz


def carla_to_device_points(points_xyz, origin_xyz, yaw_rad):
    """Level-camera point conversion. Full roll/pitch extrinsics belong in adapter."""
    delta = np.asarray(points_xyz, float) - np.asarray(origin_xyz, float)
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return delta @ np.array([[c, -s, 0], [s, c, 0], [0, 0, -1]], float)


def single_valued_forward_prefix(device_xyz, epsilon_m=1e-6):
    """Mask a lane-line prefix before local forward distance reverses or repeats."""
    x = np.asarray(device_xyz)[:, 0]
    valid = np.ones(len(x), bool)
    reversal = np.flatnonzero(np.diff(x) <= epsilon_m)
    if len(reversal):
        valid[reversal[0]+1:] = False
    valid &= x >= 0
    return valid
