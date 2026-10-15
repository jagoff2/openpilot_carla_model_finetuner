"""Independent dense physical QA for a static unmarked-road demonstration.

The target is the right-half nominal path with its declared obstacle excursions.
Actual telemetry, not the target's predicted poses, supplies all body sweeps and
tracking measurements. No initial-offset or avoidance exemption is applied.
Passing this module alone never grants dataset admission. External RGB/depth
proofs must be replayed and their SHA256 bindings supplied by the caller.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.spatial import ConvexHull

from .target import (BehaviorTarget, _boundary_distance, _inside, _polygon_distance,
    build_behavior_target, contract_fingerprint)


@dataclass(frozen=True)
class QualityConfig:
    mean_signed_offset_limit_m: float = .05
    p95_absolute_offset_limit_m: float = .10
    maximum_absolute_offset_limit_m: float = .20
    lateral_acceleration_limit_mps2: float = 3.5
    lateral_jerk_limit_mps3: float = 5.
    minimum_boundary_clearance_m: float = .15
    retained_nominal_recovery_m: float = 5.
    recovery_maximum_offset_m: float = .10
    recovery_heading_limit_rad: float = math.radians(1.)
    maximum_sweep_spacing_m: float = .05
    maximum_sweep_heading_step_rad: float = math.radians(.5)
    maximum_boundary_chord_spacing_m: float = .25
    maximum_pose_velocity_error_m: float = .03
    maximum_reverse_progress_m: float = .05

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("physical QA thresholds must be finite and positive")
        if (self.mean_signed_offset_limit_m > .05 or self.p95_absolute_offset_limit_m > .10
                or self.maximum_absolute_offset_limit_m > .20
                or self.lateral_acceleration_limit_mps2 > 3.5 or self.lateral_jerk_limit_mps3 > 5.
                or self.minimum_boundary_clearance_m < .15 or self.retained_nominal_recovery_m < 5.):
            raise ValueError("unmarked behavior QA cannot relax the existing physical limits")


def _project(points, stations, xy):
    """Project measured points onto an explicit sampled target, without extrapolation."""
    delta = np.diff(xy, axis=0)
    length2 = (delta*delta).sum(-1)
    if np.any(length2 <= 1e-14):
        raise ValueError("target has degenerate sampled segments")
    fraction = np.clip(((points[:, None]-xy[None, :-1])*delta[None]).sum(-1)/length2[None], 0., 1.)
    closest = xy[None, :-1]+fraction[..., None]*delta[None]
    chosen = np.argmin(((points[:, None]-closest)**2).sum(-1), axis=1)
    row = np.arange(len(points))
    error = points-closest[row, chosen]
    tangent = delta[chosen]/np.sqrt(length2[chosen, None])
    lateral = error[:, 1]*tangent[:, 0]-error[:, 0]*tangent[:, 1]
    progress = stations[chosen]+fraction[row, chosen]*np.diff(stations)[chosen]
    return progress, lateral, np.linalg.norm(error, axis=1)


def _nominal(reference, stations):
    q = np.asarray(stations)
    xy = .25*reference._left(q)[..., :2]+.75*reference._right(q)[..., :2]
    tangent = .25*reference._left(q, 1)[..., :2]+.75*reference._right(q, 1)[..., :2]
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
    return xy, tangent


def _ribbon(reference, spacing, *, excursions=(),passing_space=None):
    # Including every cubic knot makes the second derivative linear per interval.
    q = np.unique(np.r_[reference.s, np.arange(reference.s[0], reference.s[-1], spacing)])
    middle = (reference._left(q)+reference._right(q))*.5
    right = reference._right(q)
    second_middle = (reference._left(q, 2)+reference._right(q, 2))*.5
    second_right = reference._right(q, 2)
    guards = []
    for derivative in (second_middle, second_right):
        second = np.linalg.norm(derivative[:, :2], axis=1)
        guards.append(float(np.max(np.maximum(second[:-1], second[1:])*np.diff(q)**2/8)))
    if passing_space is not None:
        from .passing import permitted_ribbon
        # The virtual passing limit is explicitly the sampled polygon. It is
        # not a cubic interpolant across the start/end of permission. Physical
        # outer-road curves retain their original spline-to-chord error guard.
        second=np.linalg.norm(reference._left(q,2)[:,:2],axis=1)
        guards.append(float(np.max(np.maximum(second[:-1],second[1:])*np.diff(q)**2/8)))
        return permitted_ribbon(reference,spacing,excursions,passing_space),max(guards)
    return np.concatenate((middle[:, :2], right[::-1, :2])), max(guards)


def _body(position, yaw, corners):
    c, s = math.cos(yaw), math.sin(yaw)
    return position[:2]+corners@np.array([[c, s], [-s, c]])


def swept_clearance(positions_xyz, yaw_rad, target, obstacles=(), *, config=None):
    """Whole polygon sweep under measured-pose interpolation, with rotation guards.

    Positions interpolate linearly and yaw follows its shortest unwrapped arc.
    A corner's rotation arc differs from its endpoint chord by at most
    radius*(1-cos(delta_yaw/2)); that bound is subtracted from every clearance.
    Road spline-to-chord error is also subtracted. This is planar physical QA;
    it does not certify unrecorded roll/pitch or future moving actors.
    """
    cfg = config or QualityConfig()
    positions, yaw = np.asarray(positions_xyz, float), np.unwrap(np.asarray(yaw_rad, float))
    if (positions.shape != (len(yaw), 3) or len(yaw) < 2
            or not np.isfinite(positions).all() or not np.isfinite(yaw).all()):
        raise ValueError("swept clearance requires finite aligned actual poses")
    corners = target.ego.corners()
    radius = float(np.linalg.norm(corners, axis=1).max())
    ribbon, chord_guard = _ribbon(target.road.reference, cfg.maximum_boundary_chord_spacing_m,
        excursions=target.excursions,passing_space=target.passing_space)
    required_road = max(cfg.minimum_boundary_clearance_m, target.config.edge_clearance_m)
    polygons = [(o.identity, o.polygon(), target.config.obstacle_clearance(o.kind)) for o in obstacles]
    if any(margin is None for _, _, margin in polygons):
        raise ValueError("unsupported obstacle class in physical QA")
    minimum_road, minimum_obstacle = math.inf, {name:math.inf for name, _, _ in polygons}
    max_rotation_guard, swept_pieces, leaves = 0., 0, False
    reasons = []
    for i in range(len(positions)-1):
        distance = float(np.linalg.norm(positions[i+1, :2]-positions[i, :2]))
        angle = abs(float(yaw[i+1]-yaw[i]))
        count = max(1, math.ceil(distance/cfg.maximum_sweep_spacing_m),
                    math.ceil(angle/cfg.maximum_sweep_heading_step_rad))
        rotation_guard = radius*(1-math.cos(angle/count/2))
        max_rotation_guard = max(max_rotation_guard, rotation_guard)
        before = _body(positions[i], yaw[i], corners)
        for fraction in np.arange(1, count+1)/count:
            point = positions[i]*(1-fraction)+positions[i+1]*fraction
            direction = yaw[i]*(1-fraction)+yaw[i+1]*fraction
            after = _body(point, direction, corners)
            endpoints = np.concatenate((before, after))
            polygon = endpoints[ConvexHull(endpoints).vertices]
            inside = bool(_inside(polygon, ribbon).all())
            edge = _boundary_distance(polygon, ribbon)-rotation_guard-chord_guard
            if not inside:
                leaves = True
                edge = min(edge, 0.)
            minimum_road = min(minimum_road, edge)
            for identity, obstacle, required in polygons:
                clearance = _polygon_distance(polygon, obstacle)-rotation_guard
                minimum_obstacle[identity] = min(minimum_obstacle[identity], clearance)
            before = after
            swept_pieces += 1
    region='right_half' if target.passing_space is None else 'permitted_road'
    if leaves:
        reasons.append("actual_vehicle_body_leaves_"+region)
    if minimum_road < required_road-1e-8:
        reasons.append("actual_"+region+"_body_clearance_below_limit")
    for identity, _, required in polygons:
        if minimum_obstacle[identity] < required-1e-8:
            reasons.append("actual_obstacle_clearance_below_limit:"+identity)
    return dict(passed=not reasons, reasons=reasons,
        **{'minimum_'+region+'_clearance_m':minimum_road,'required_'+region+'_clearance_m':required_road},
        minimum_obstacle_clearance_m=minimum_obstacle,
        required_obstacle_clearance_m={name:margin for name, _, margin in polygons},
        maximum_rotation_guard_m=max_rotation_guard, boundary_chord_error_bound_m=chord_guard,
        swept_polygon_pieces=swept_pieces, body_model="planar_ego_polygon_with_bbox_offset_and_yaw",
        interpolation="linear_position_shortest_yaw_arc_with_rotation_and_boundary_chord_guards")


def _visual_bindings(value, obstacles):
    def binding(item):
        return (isinstance(item, dict) and item.get("passed") is True
            and isinstance(item.get("sha256"), str) and len(item["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in item["sha256"]))
    return bool(isinstance(value, dict) and value.get("replayed") is True
        and binding(value.get("road")) and isinstance(value.get("obstacles"), dict)
        and set(value["obstacles"]) == {o.identity for o in obstacles}
        and all(binding(item) for item in value["obstacles"].values()))


def qualify_episode(*, timestamps_s, positions_xyz, yaw_rad, velocity_world,
                    lateral_acceleration_mps2, target: BehaviorTarget, obstacles=(),
                    collision_count=None, visual_evidence=None, config=None):
    """Score the full demonstrated target and every avoidance/recovery event.

    ``visual_evidence`` is an external replay receipt, not a substitute for
    replay: {replayed: True, road: {passed: True, sha256: ...}, obstacles:
    {identity: {passed: True, sha256: ...}}}. The publisher must generate that
    receipt only after replaying manifest-bound RGB/depth evidence. This module
    never sets training_ready, even when physical and visual checks both pass.
    """
    cfg = config or QualityConfig()
    if not isinstance(target, BehaviorTarget):
        raise TypeError("BehaviorTarget required")
    obstacles = tuple(obstacles)
    if not target.moving_supervision:
        return dict(passed=False, physical_passed=False, training_ready=False,
            reasons=["behavior_target_requires_yield", *target.reasons], metrics={})
    expected = contract_fingerprint(target.road, obstacles, target.ego, target.config,target.passing_space)
    if target.contract_sha256 != expected:
        raise ValueError("physical QA scene differs from the target contract")
    replay = build_behavior_target(target.road, obstacles, target.ego, config=target.config,passing_space=target.passing_space)
    if (not replay.moving_supervision or replay.excursions != target.excursions
            or not np.array_equal(replay.s, target.s) or replay.obstacle_ids != target.obstacle_ids):
        raise ValueError("physical QA target differs from deterministic reconstruction")
    t, position = np.asarray(timestamps_s, float), np.asarray(positions_xyz, float)
    yaw, velocity = np.asarray(yaw_rad, float), np.asarray(velocity_world, float)
    acceleration = np.asarray(lateral_acceleration_mps2, float)
    if (t.ndim != 1 or len(t) < 3 or position.shape != (len(t), 3) or velocity.shape != position.shape
            or yaw.shape != t.shape or acceleration.shape != t.shape
            or not all(np.isfinite(v).all() for v in (t, position, yaw, velocity, acceleration))
            or np.any(np.diff(t) <= 0)):
        raise ValueError("physical QA requires finite, aligned, increasing dense telemetry")
    dt = np.diff(t)
    reasons = []
    if not np.allclose(dt, .05, rtol=0, atol=1e-5):
        reasons.append("missing_duplicate_or_non20hz_frames")
    displacement_error = np.linalg.norm(np.diff(position, axis=0)-.5*(velocity[:-1]+velocity[1:])*dt[:, None], axis=1)
    if displacement_error.max() > cfg.maximum_pose_velocity_error_m:
        reasons.append("actual_pose_velocity_disagreement")
    station, lateral, distance = _project(position[:, :2], target.s, target.xy)
    if distance.max() > cfg.maximum_absolute_offset_limit_m:
        reasons.append("maximum_target_distance")
    mean, p95, maximum = float(lateral.mean()), float(np.quantile(abs(lateral), .95)), float(abs(lateral).max())
    for name, measured, limit in (("mean_signed_target_offset", abs(mean), cfg.mean_signed_offset_limit_m),
            ("p95_absolute_target_offset", p95, cfg.p95_absolute_offset_limit_m),
            ("maximum_absolute_target_offset", maximum, cfg.maximum_absolute_offset_limit_m)):
        if measured > limit+1e-9:
            reasons.append(name)
    if np.any(np.diff(station) < -cfg.maximum_reverse_progress_m):
        reasons.append("reverse_target_progress")
    peak_acceleration = float(abs(acceleration).max())
    peak_jerk = float(np.max(abs(np.diff(acceleration)/dt)))
    if peak_acceleration > cfg.lateral_acceleration_limit_mps2+1e-9:
        reasons.append("lateral_acceleration_above_limit")
    if peak_jerk > cfg.lateral_jerk_limit_mps3+1e-9:
        reasons.append("lateral_jerk_above_limit")
    if isinstance(collision_count, bool) or not isinstance(collision_count, int) or collision_count < 0:
        reasons.append("collision_sensor_evidence_missing_or_invalid")
    elif collision_count:
        reasons.append("collision")
    sweep = swept_clearance(position, yaw, target, obstacles, config=cfg)
    reasons.extend(sweep["reasons"])
    event_reports = []
    speed = np.linalg.norm(velocity[:, :2], axis=1)
    motion_yaw = np.where(speed > .2, np.arctan2(velocity[:, 1], velocity[:, 0]), yaw)
    for number, event in enumerate(target.excursions):
        errors = []
        if station[0] > event.start_s+1e-6 or station[-1] < event.end_s-1e-6:
            errors.append("avoidance_event_not_fully_traversed")
        for start, end in ((event.start_s, event.plateau_start_s),
                           (event.plateau_start_s, event.plateau_end_s), (event.plateau_end_s, event.end_s)):
            if not np.any((station >= start) & (station <= end)):
                errors.append("avoidance_phase_not_recorded")
        next_start = target.excursions[number+1].start_s if number+1 < len(target.excursions) else math.inf
        retained = np.flatnonzero((station >= event.end_s) & (station < next_start))
        recovery_distance, recovery_maximum, recovery_heading = 0., None, None
        if len(retained):
            reference = target.road.reference
            # Measure physical arc of the correctly offset nominal path, rather
            # than treating the full-road midpoint's station units as its arc.
            dense_q = np.unique(np.r_[event.end_s, station[retained], np.arange(event.end_s,
                station[retained].max(), cfg.maximum_boundary_chord_spacing_m)])
            nominal, _ = _nominal(reference, dense_q)
            nominal_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(nominal, axis=0), axis=1))]
            recovery_distance = float(nominal_arc[-1])
            covered = np.interp(station[retained], dense_q, nominal_arc)
            end = int(np.searchsorted(covered, cfg.retained_nominal_recovery_m, side="left"))
            retained = retained[:min(len(retained), end+1)]
            nominal, tangent = _nominal(reference, station[retained])
            error = position[retained, :2]-nominal
            offset = error[:, 1]*tangent[:, 0]-error[:, 0]*tangent[:, 1]
            heading_error = (motion_yaw[retained]-np.arctan2(tangent[:, 1], tangent[:, 0])+np.pi) % (2*np.pi)-np.pi
            recovery_maximum = float(abs(offset).max())
            recovery_heading = float(abs(heading_error).max())
            if recovery_maximum > cfg.recovery_maximum_offset_m+1e-9:
                errors.append("nominal_recovery_offset_above_limit")
            if recovery_heading > cfg.recovery_heading_limit_rad+1e-9:
                errors.append("nominal_recovery_heading_above_limit")
        if recovery_distance < cfg.retained_nominal_recovery_m-1e-6:
            errors.append("less_than_five_metres_retained_nominal_recovery")
        report = dict(event_index=number, obstacle_ids=list(event.obstacle_ids),
            start_s=float(event.start_s), end_s=float(event.end_s), passed=not errors,
            reasons=sorted(set(errors)), retained_nominal_distance_m=recovery_distance,
            maximum_nominal_offset_m=recovery_maximum, maximum_nominal_heading_error_rad=recovery_heading)
        event_reports.append(report)
        reasons.extend(f"event_{number}:{error}" for error in errors)
    physical_passed = not reasons
    visual_verified = _visual_bindings(visual_evidence, obstacles)
    if not visual_verified:
        reasons.append("replayed_visual_road_and_obstacle_proofs_required")
    return dict(passed=physical_passed and visual_verified, physical_passed=physical_passed,
        visual_evidence_verified=visual_verified, training_ready=False,
        scope="physical static-scene demonstration QA; independent corpus admission still required",
        reasons=sorted(set(reasons)), target_contract_sha256=expected,
        visual_evidence_bindings=visual_evidence if visual_verified else None,
        metrics=dict(mean_signed_target_offset_m=mean, p95_absolute_target_offset_m=p95,
            maximum_absolute_target_offset_m=maximum, maximum_target_distance_m=float(distance.max()),
            maximum_lateral_acceleration_mps2=peak_acceleration, maximum_lateral_jerk_mps3=peak_jerk,
            maximum_pose_velocity_error_m=float(displacement_error.max()),
            initial_station_m=float(station[0]), final_station_m=float(station[-1]),
            collision_count=collision_count, quality_limits=asdict(cfg), sweep=sweep,
            avoidance_events=event_reports))
