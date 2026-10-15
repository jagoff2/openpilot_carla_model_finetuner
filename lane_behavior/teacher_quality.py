"""Actual-rollout physical QA against an explicitly privileged teacher plan.

This is deliberately separate from observed-label qualification. It verifies
that the simulated vehicle physically demonstrated its planned right-side
behavior. Road/obstacle geometry here is simulator truth, never camera evidence.
No successful result from this module grants observation or corpus admission.
"""
from __future__ import annotations

from dataclasses import asdict
import math
from numbers import Integral

import numpy as np
from scipy.spatial import ConvexHull

from .evidence import array_sha256
from .planned import PlannedTarget, planned_scene_geometry
from .quality import QualityConfig, _body, _nominal, _project, _ribbon
from .target import _boundary_distance, _inside, _polygon_distance


VERSION = "privileged_teacher_actual_rollout_physical_qa_v1"


def _sweep(position, yaw, target, cfg):
    """Numerical full-polygon sweep; no observed-reference adapter is created."""
    yaw = np.unwrap(yaw)
    corners = target.ego.corners()
    radius = float(np.linalg.norm(corners, axis=1).max())
    ribbon, chord_guard = _ribbon(target.reference, cfg.maximum_boundary_chord_spacing_m,
        excursions=target.excursions,passing_space=target.passing_space)
    road_margin = max(cfg.minimum_boundary_clearance_m, target.config.edge_clearance_m)
    obstacles = [(o.identity, o.polygon(), target.config.obstacle_clearance(o.kind)) for o in target.obstacles]
    if any(margin is None for _, _, margin in obstacles):
        raise ValueError("teacher QA encountered unsupported obstacle class")
    minimum_road = math.inf
    minimum_obstacle = {name:math.inf for name, _, _ in obstacles}
    maximum_rotation_guard, pieces, leaves = 0., 0, False
    for i in range(len(position)-1):
        distance = float(np.linalg.norm(position[i+1, :2]-position[i, :2]))
        angle = abs(float(yaw[i+1]-yaw[i]))
        count = max(1, math.ceil(distance/cfg.maximum_sweep_spacing_m),
            math.ceil(angle/cfg.maximum_sweep_heading_step_rad))
        rotation_guard = radius*(1-math.cos(angle/count/2))
        maximum_rotation_guard = max(maximum_rotation_guard, rotation_guard)
        before = _body(position[i], yaw[i], corners)
        for fraction in np.arange(1, count+1)/count:
            point = position[i]*(1-fraction)+position[i+1]*fraction
            heading = yaw[i]*(1-fraction)+yaw[i+1]*fraction
            after = _body(point, heading, corners)
            endpoints = np.concatenate((before, after))
            polygon = endpoints[ConvexHull(endpoints).vertices]
            inside = bool(_inside(polygon, ribbon).all())
            edge = _boundary_distance(polygon, ribbon)-rotation_guard-chord_guard
            if not inside:
                leaves = True
                edge = min(edge, 0.)
            minimum_road = min(minimum_road, edge)
            for name, obstacle, _ in obstacles:
                clearance = _polygon_distance(polygon, obstacle)-rotation_guard
                minimum_obstacle[name] = min(minimum_obstacle[name], clearance)
            before = after
            pieces += 1
    reasons = []
    region='right_half' if target.passing_space is None else 'permitted_road'
    if leaves:
        reasons.append("actual_vehicle_body_leaves_"+region)
    if minimum_road < road_margin-1e-8:
        reasons.append("actual_"+region+"_body_clearance_below_limit")
    for name, _, margin in obstacles:
        if minimum_obstacle[name] < margin-1e-8:
            reasons.append("actual_obstacle_clearance_below_limit:"+name)
    return dict(passed=not reasons, reasons=reasons,
        **{'minimum_'+region+'_clearance_m':minimum_road,'required_'+region+'_clearance_m':road_margin},
        minimum_obstacle_clearance_m=minimum_obstacle,
        required_obstacle_clearance_m={name:margin for name, _, margin in obstacles},
        maximum_rotation_guard_m=maximum_rotation_guard, boundary_chord_error_bound_m=chord_guard,
        swept_polygon_pieces=pieces, geometry_source="fixed_privileged_simulator_scene",
        body_model="planar_ego_polygon_with_bbox_offset_and_yaw",
        interpolation="linear_position_shortest_yaw_arc_with_rotation_and_boundary_chord_guards")


def _events(target, station, position, yaw, velocity, cfg):
    speed = np.linalg.norm(velocity[:, :2], axis=1)
    motion_yaw = np.where(speed > .2, np.arctan2(velocity[:, 1], velocity[:, 0]), yaw)
    reports = []
    required_recovery = max(cfg.retained_nominal_recovery_m, target.minimum_retained_nominal_recovery_m)
    for number, event in enumerate(target.excursions):
        reasons = []
        if station[0] > event.start_s+1e-6 or station[-1] < event.end_s-1e-6:
            reasons.append("avoidance_event_not_fully_traversed")
        phase_counts = {}
        for phase, start, end in (("departure", event.start_s, event.plateau_start_s),
                ("pass", event.plateau_start_s, event.plateau_end_s),
                ("return", event.plateau_end_s, event.end_s)):
            phase_counts[phase] = int(np.count_nonzero((station >= start) & (station <= end)))
            if not phase_counts[phase]:
                reasons.append("avoidance_phase_not_recorded:"+phase)
        next_start = target.excursions[number+1].start_s if number+1 < len(target.excursions) else math.inf
        retained = np.flatnonzero((station >= event.end_s) & (station < next_start))
        recovery_distance, max_offset, max_heading, count = 0., None, None, 0
        # A physical follower can finish its return slightly after the planned
        # curve ends. Require an entire stable measured interval, with a hard
        # two-metre search bound, rather than treating its first frame as settled.
        settling_limit = 2.
        actual_start, start_index, settling, stable_distance = None, None, None, 0.
        available_distance, tried, failures = 0., 0, set()
        if len(retained):
            dense = np.unique(np.r_[event.end_s, station[retained], np.arange(event.end_s,
                station[retained].max(), cfg.maximum_boundary_chord_spacing_m)])
            nominal, _ = _nominal(target.reference, dense)
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(nominal, axis=0), axis=1))]
            available_distance = float(arc[-1])
            covered = np.interp(station[retained], dense, arc)
            nominal, tangent = _nominal(target.reference, station[retained])
            error = position[retained, :2]-nominal
            offset = error[:, 1]*tangent[:, 0]-error[:, 0]*tangent[:, 1]
            heading = (motion_yaw[retained]-np.arctan2(tangent[:, 1], tangent[:, 0])+np.pi) % (2*np.pi)-np.pi
            for first in np.flatnonzero(covered <= settling_limit+1e-9):
                reached = np.flatnonzero(covered[first:]-covered[first] >= required_recovery-1e-6)
                if not len(reached):
                    continue
                last = first+int(reached[0])
                if np.any(np.diff(retained[first:last+1]) != 1):
                    continue
                tried += 1
                window_offset = float(abs(offset[first:last+1]).max())
                window_heading = float(abs(heading[first:last+1]).max())
                if window_offset > cfg.recovery_maximum_offset_m+1e-9:
                    failures.add("nominal_recovery_offset_above_limit")
                if window_heading > cfg.recovery_heading_limit_rad+1e-9:
                    failures.add("nominal_recovery_heading_above_limit")
                if (window_offset > cfg.recovery_maximum_offset_m+1e-9
                        or window_heading > cfg.recovery_heading_limit_rad+1e-9):
                    continue
                actual_start = float(station[retained[first]])
                start_index = int(retained[first])
                settling = float(covered[first])
                stable_distance = float(covered[last]-covered[first])
                recovery_distance = float(covered[-1]-covered[first])
                max_offset, max_heading = window_offset, window_heading
                count = int(last-first+1)
                break
        if actual_start is None:
            reasons.extend(sorted(failures))
            if tried:
                reasons.append("nominal_recovery_not_stable_within_settling_bound")
        if actual_start is None and not tried:
            reasons.append("insufficient_retained_nominal_recovery")
        reports.append(dict(event_index=number, obstacle_ids=list(event.obstacle_ids),
            start_s=float(event.start_s), end_s=float(event.end_s), passed=not reasons,
            reasons=reasons, phase_frame_counts=phase_counts, retained_nominal_frames=count,
            required_nominal_recovery_m=required_recovery, retained_nominal_distance_m=recovery_distance,
            planned_return_end_s=float(event.end_s), actual_recovery_start_s=actual_start,
            actual_recovery_start_frame_index=start_index, settling_distance_m=settling,
            maximum_allowed_settling_distance_m=settling_limit,
            actual_return_extended=bool(settling is not None and settling > 1e-6),
            stable_nominal_recovery_distance_m=stable_distance,
            available_nominal_distance_after_planned_return_m=available_distance,
            recovery_windows_checked=tried,
            maximum_nominal_offset_m=max_offset, maximum_nominal_heading_error_rad=max_heading))
    return reports


def qualify_teacher_rollout(*, target: PlannedTarget, target_sha256: str, timestamps_s,
                            positions_xyz, yaw_rad, velocity_world, lateral_acceleration_mps2,
                            collision_count=None, config=None):
    """Validate measured physical behavior, with no visible-label admission claim.

    target_sha256 must come from the capture's recorded PlannedTarget contract.
    Passing requires complete avoidance phases and >=5 m nominal recovery. No
    speed target or longitudinal loss is scored. The sweep assumes the declared
    static actor envelopes stay fixed; capture must separately record/verify
    their actual stationarity. Model inputs and visual support are not checked
    here and must be independently admitted before any training publication.
    """
    if not isinstance(target, PlannedTarget):
        raise TypeError("teacher physical QA requires a distinct PlannedTarget")
    if (not isinstance(target_sha256, str) or len(target_sha256) != 64
            or target_sha256 != target.contract_sha256):
        raise ValueError("recorded teacher target SHA256 does not match")
    target.verify_contract()
    replay = planned_scene_geometry(target.reference, target.obstacles, target.ego,
        config=target.config, source_metadata=target.source_metadata,
        minimum_retained_nominal_recovery_m=target.minimum_retained_nominal_recovery_m,
        obstacle_tracking_reserve_m=target.obstacle_tracking_reserve_m,passing_space=target.passing_space)
    if replay.contract_sha256 != target_sha256:
        raise ValueError("recorded teacher target differs from deterministic reconstruction")
    base = dict(version=VERSION, source_scope="teacher_only", target_source_scope=target.source_scope,
        target_contract_sha256=target_sha256, training_ready=False, supervision_allowed=False,
        observation_admission_passed=False, simulator_stationarity_verification_required=True,
        scope="actual simulated physical demonstration against privileged planned geometry")
    if not target.feasible:
        return dict(**base, passed=False, physical_passed=False,
            reasons=["teacher_plan_requires_yield", *target.reasons], metrics={})
    cfg = config or QualityConfig()
    t, position = np.asarray(timestamps_s, float), np.asarray(positions_xyz, float)
    yaw, velocity = np.asarray(yaw_rad, float), np.asarray(velocity_world, float)
    acceleration = np.asarray(lateral_acceleration_mps2, float)
    if (t.ndim != 1 or len(t) < 3 or position.shape != (len(t), 3) or velocity.shape != position.shape
            or yaw.shape != t.shape or acceleration.shape != t.shape
            or not all(np.isfinite(v).all() for v in (t, position, yaw, velocity, acceleration))
            or np.any(np.diff(t) <= 0)):
        raise ValueError("teacher QA requires finite, aligned, increasing actual telemetry")
    dt, reasons = np.diff(t), []
    if not np.allclose(dt, .05, rtol=0, atol=1e-5):
        reasons.append("missing_duplicate_or_non20hz_frames")
    pose_error = np.linalg.norm(np.diff(position, axis=0)-.5*(velocity[:-1]+velocity[1:])*dt[:, None], axis=1)
    if pose_error.max() > cfg.maximum_pose_velocity_error_m:
        reasons.append("actual_pose_velocity_disagreement")
    station, lateral, distance = _project(position[:, :2], target.s, target.xy)
    mean, p95, maximum = float(lateral.mean()), float(np.quantile(abs(lateral), .95)), float(abs(lateral).max())
    for name, measured, limit in (("mean_signed_target_offset", abs(mean), cfg.mean_signed_offset_limit_m),
            ("p95_absolute_target_offset", p95, cfg.p95_absolute_offset_limit_m),
            ("maximum_absolute_target_offset", maximum, cfg.maximum_absolute_offset_limit_m),
            ("maximum_target_distance", float(distance.max()), cfg.maximum_absolute_offset_limit_m)):
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
    if isinstance(collision_count, bool) or not isinstance(collision_count, Integral) or collision_count < 0:
        reasons.append("collision_sensor_evidence_missing_or_invalid")
    elif collision_count:
        reasons.append("collision")
    sweep = _sweep(position, yaw, target, cfg)
    reasons.extend(sweep["reasons"])
    events = _events(target, station, position, yaw, velocity, cfg)
    reasons.extend(f"event_{report['event_index']}:{reason}" for report in events for reason in report["reasons"])
    return dict(**base, passed=not reasons, physical_passed=not reasons, reasons=sorted(set(reasons)),
        telemetry_sha256={name:array_sha256(value) for name, value in
            (("timestamps_s", t), ("positions_xyz", position), ("yaw_rad", yaw),
             ("velocity_world", velocity), ("lateral_acceleration_mps2", acceleration))},
        metrics=dict(tracking=dict(mean_signed_target_offset_m=mean, p95_absolute_target_offset_m=p95,
            maximum_absolute_target_offset_m=maximum, maximum_target_distance_m=float(distance.max()),
            initial_station_m=float(station[0]), final_station_m=float(station[-1])),
            dynamics=dict(maximum_lateral_acceleration_mps2=peak_acceleration, maximum_lateral_jerk_mps3=peak_jerk),
            integrity=dict(frame_count=len(t), duration_s=float(t[-1]-t[0]),
                maximum_pose_velocity_error_m=float(pose_error.max()),
                collision_count=int(collision_count) if isinstance(collision_count, Integral) else None),
            sweep=sweep, avoidance_events=events, quality_limits=asdict(cfg)))
