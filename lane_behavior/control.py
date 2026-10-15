"""Qualified behavior targets adapted to the unchanged observed-reference MPC.

The MPC follows an offset/avoidance *reference center*, with zero scalar desired
offset. Its symmetric corridor is conservatively inscribed in observed road and
obstacle constraints. A second body-sweep check is required before any returned
command is marked accepted. Physical capture QA remains a separate requirement.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from lane_dataset.controller import BicycleMPC, MPCConfig, MPCCommand, VehicleState
from lane_dataset.geometry import ObservableReference
from .target import (BehaviorTarget, ObservedObstacle, _body, _boundary_distance,
    _geometry, _inside, _point_segment_distance, _polygon_distance, contract_fingerprint)


@dataclass(frozen=True)
class ControlAdapterConfig:
    sample_spacing_m: float = .15
    interpolation_guard_m: float = .01
    maximum_center_fit_error_m: float = .005

    def __post_init__(self):
        if (not np.isfinite(list(vars(self).values())).all()
                or min(vars(self).values()) <= 0 or self.sample_spacing_m > .25):
            raise ValueError("control adapter requires finite positive settings and spacing <= .25m")


@dataclass(frozen=True)
class ControlReference:
    ready: bool
    reasons: tuple[str, ...]
    reference: ObservableReference | None
    contract_sha256: str | None
    source_stations: np.ndarray
    diagnostics: dict


@dataclass(frozen=True)
class BehaviorControlDecision:
    accepted: bool
    reasons: tuple[str, ...]
    command: MPCCommand | None
    control_reference: ControlReference


def _scene(target, obstacles, config):
    if not isinstance(target, BehaviorTarget):
        raise TypeError("behavior controller requires a BehaviorTarget")
    if (target.contract_sha256 is None or target.contract_sha256 !=
            contract_fingerprint(target.road, obstacles, target.ego, target.config)):
        raise ValueError("behavior inputs do not match target scene contract")
    corners = target.ego.corners()
    # The unchanged MPC uses a centered rectangle. Its rectangle must enclose
    # any actual bounding-box offset/yaw rather than silently dropping them.
    if (config.vehicle_width_m+1e-8 < 2*np.abs(corners[:, 1]).max()
            or config.vehicle_length_m+1e-8 < 2*np.abs(corners[:, 0]).max()):
        raise ValueError("MPC rectangle does not enclose the recorded ego footprint")
    ref = target.road.reference
    q = np.linspace(ref.s[0], ref.s[-1], int(np.ceil((ref.s[-1]-ref.s[0])/.15))+1)
    _, _, divider, edge = _geometry(ref, q)
    ribbon = np.concatenate((divider, edge[::-1]))
    qualified = tuple((o, o.polygon(), target.config.obstacle_clearance(o.kind)) for o in obstacles)
    return ribbon, qualified


def _point_boundary_clearance(points, polygon):
    # Exact nearest-segment search with a midpoint-distance lower bound. This
    # avoids allocating a full path-points by road-segments distance matrix.
    start, end = polygon, np.roll(polygon, -1, axis=0)
    tree = cKDTree((start+end)*.5)
    radius = float(np.linalg.norm(end-start, axis=1).max()/2)
    remaining = np.arange(len(points))
    answer = np.empty(len(points))
    count = min(8, len(polygon))
    while len(remaining):
        distance, index = tree.query(points[remaining], k=count)
        if count == 1:
            distance, index = distance[:, None], index[:, None]
        exact = _point_segment_distance(points[remaining, None], start[index], end[index]).min(axis=1)
        complete = (distance[:, -1]-radius >= exact-1e-10) | (count == len(polygon))
        answer[remaining[complete]] = exact[complete]
        remaining = remaining[~complete]
        count = min(len(polygon), 2*count)
    return answer


def _half_width(points, ribbon, obstacles, target, mpc_config):
    """Inscribed cross-section radius; MPC's own body margin is accounted once."""
    road_margin = max(0., target.config.edge_clearance_m-mpc_config.clearance_m)
    radius = _point_boundary_clearance(points, ribbon)-road_margin
    radius = np.where(_inside(points, ribbon), radius, -1.)
    for _, polygon, margin in obstacles:
        obstacle_margin = max(0., margin-mpc_config.clearance_m)
        distance = _point_boundary_clearance(points, polygon)-obstacle_margin
        distance = np.where(_inside(points, polygon), -1., distance)
        radius = np.minimum(radius, distance)
    return radius


def build_control_reference(target: BehaviorTarget, obstacles=(), mpc_config=None, *, config=None):
    """Return a conservative ObservableReference or explicit yield reasons.

    Derived boundaries are control envelopes, never observations of paint or
    road edges. Their center matches the behavior target, while their width is
    reduced by nearby obstacles and the physical right-half boundary. Derived
    station is target chord length, because MPC progress is measured in meters.
    """
    mpc_cfg, cfg = mpc_config or MPCConfig(), config or ControlAdapterConfig()
    obstacles = tuple(obstacles)
    ribbon, qualified = _scene(target, obstacles, mpc_cfg)
    def result(reasons=(), reference=None, source=None, diagnostics=None):
        return ControlReference(not reasons, tuple(reasons), reference, target.contract_sha256,
            np.asarray([] if source is None else source), diagnostics or {})
    if not target.moving_supervision:
        return result(("behavior_target_requires_yield", *target.reasons))
    count = max(4, int(np.ceil((target.s[-1]-target.s[0])/cfg.sample_spacing_m))+1)
    source = np.linspace(target.s[0], target.s[-1], count)
    joins = np.asarray([q for e in target.excursions for q in
        (e.start_s, e.plateau_start_s, e.plateau_end_s, e.end_s)])
    source = np.unique(np.r_[source, joins])
    xy, tangent, _, _, valid = target.sample(source)
    if not valid.all():
        return result(("control_reference_lacks_observed_support",))
    half_width = _half_width(xy, ribbon, qualified, target, mpc_cfg)-cfg.interpolation_guard_m
    required_half_width = mpc_cfg.vehicle_width_m/2+mpc_cfg.clearance_m
    if np.any(half_width <= required_half_width):
        return result(("symmetric_control_envelope_too_narrow",), source=source,
            diagnostics={"minimum_half_width_m": float(half_width.min()), "required_half_width_m": required_half_width})
    ref = target.road.reference
    l, r = ref._left(source), ref._right(source)
    cross = r[:, :2]-l[:, :2]
    fraction = ((xy-l[:, :2])*cross).sum(1)/np.maximum((cross*cross).sum(1), 1e-12)
    z = l[:, 2]+fraction*(r[:, 2]-l[:, 2])
    center = np.c_[xy, z]
    normal = np.c_[-tangent[:, 1], tangent[:, 0], np.zeros(len(source))]
    left, right = center-normal*half_width[:, None], center+normal*half_width[:, None]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if np.any(np.diff(arc) <= 1e-6):
        return result(("control_reference_folds_or_repeats",), source=source)
    derived = ObservableReference(left, right, arc, [target.evidence_frames]*len(arc), ref.decision_time_s)
    # Independent dense refit check: symmetric boundary splines must preserve
    # their target center and stay inside the original physical envelopes.
    dense = np.linspace(arc[0], arc[-1], 4*(len(arc)-1)+1)
    fitted, _, _, width = derived.sample(dense)
    mapped_source = np.interp(dense, arc, source)
    expected = target.sample(mapped_source)[0]
    error = np.linalg.norm(fitted[:, :2]-expected, axis=1)
    legal_width = _half_width(fitted[:, :2], ribbon, qualified, target, mpc_cfg)
    if error.max() > cfg.maximum_center_fit_error_m:
        return result(("control_center_refit_disagrees_with_behavior_target",), source=source)
    if np.any(width/2 > legal_width+1e-8) or np.any(width/2 <= required_half_width):
        return result(("control_boundary_refit_leaves_physical_envelope",), source=source,
            diagnostics={"maximum_boundary_excess_m": float((width/2-legal_width).max()),
                "minimum_body_slack_m": float((width/2-required_half_width).min())})
    derived.behavior_contract_sha256 = target.contract_sha256
    derived.behavior_source_stations = source.copy()
    derived.support_method = "observed_right_half_obstacle_inscribed_control_envelope"
    return result(reference=derived, source=source, diagnostics=dict(
        minimum_half_width_m=float((width/2).min()), maximum_center_fit_error_m=float(error.max()),
        scalar_desired_offset_m=0., obstacle_ids=[o.identity for o in obstacles],
        envelope_kind="derived_control_constraint_not_physical_boundary_observation"))


def validate_predicted_body(states, target, obstacles, mpc_config):
    """Recheck sampled prediction chords and rotating body against actual scene."""
    ribbon, qualified = _scene(target, tuple(obstacles), mpc_config)
    if not target.moving_supervision:
        return False, ("behavior_target_requires_yield",)
    states = np.asarray(states, float)
    if states.ndim != 2 or states.shape[1] < 3 or len(states) < 2 or not np.isfinite(states).all():
        return False, ("invalid_mpc_predicted_states",)
    tangent = np.c_[np.cos(states[:, 2]), np.sin(states[:, 2])]
    corners = target.ego.corners()
    bodies = _body(states[:, :2], tangent, corners)
    radius = float(np.linalg.norm(corners, axis=1).max())
    for i in range(len(bodies)-1):
        cloud = np.concatenate((bodies[i], bodies[i+1]))
        swept = cloud[ConvexHull(cloud).vertices]
        angle = abs(np.arctan2(np.sin(states[i+1, 2]-states[i, 2]), np.cos(states[i+1, 2]-states[i, 2])))
        guard = radius*angle*angle/8
        if (not _inside(swept, ribbon).all() or
                _boundary_distance(swept, ribbon) < target.config.edge_clearance_m+guard-1e-8):
            return False, ("mpc_prediction_leaves_physical_right_half",)
        for obstacle, polygon, margin in qualified:
            if _polygon_distance(swept, polygon) < margin+guard-1e-8:
                return False, ("mpc_prediction_obstacle_clearance:"+obstacle.identity,)
    return True, ()


class BehaviorExpert:
    """Stateful MPC wrapper. Callers must yield/brake on any rejected decision.

    A command is exposed only with an explicit accepted flag. No transform or
    velocity overrides are used, and the existing MPC calibration requirement
    is retained. A target is rebuilt from causal observations each decision.
    """
    def __init__(self, mpc: BicycleMPC, *, config=None):
        if not isinstance(mpc, BicycleMPC):
            raise TypeError("behavior expert requires BicycleMPC")
        self.mpc = mpc
        self.config = config or ControlAdapterConfig()

    def reset(self):
        self.mpc.reset()

    def control(self, state: VehicleState, target: BehaviorTarget, obstacles, desired_speed_mps):
        obstacles = tuple(obstacles)
        prepared = build_control_reference(target, obstacles, self.mpc.config, config=self.config)
        if not prepared.ready:
            return BehaviorControlDecision(False, prepared.reasons, None, prepared)
        before = deepcopy(self.mpc.__dict__)
        try:
            command = self.mpc.control(state, prepared.reference, desired_speed_mps, desired_offset_m=0.)
        except ValueError as exc:
            if "insufficient_observed_stopping_corridor" not in str(exc):
                raise
            self.mpc.__dict__.clear()
            self.mpc.__dict__.update(before)
            return BehaviorControlDecision(False, ("insufficient_observed_stopping_corridor",), None, prepared)
        if not command.accepted:
            return BehaviorControlDecision(False, command.reasons, command, prepared)
        valid, reasons = validate_predicted_body(command.predicted_states, target, obstacles, self.mpc.config)
        if not valid:
            # MPC normally records an accepted command as pending execution.
            # Rejected independent geometry checks must not enter that history.
            self.mpc.__dict__.clear()
            self.mpc.__dict__.update(before)
            command = replace(command, accepted=False, reasons=tuple((*command.reasons, *reasons)))
        return BehaviorControlDecision(valid, reasons, command, prepared)
