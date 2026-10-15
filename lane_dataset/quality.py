"""Strict physical episode qualification; accepts actual recorded telemetry only.

The report distinguishes missing evidence from threshold failures. Rendered
boundary agreement, swept body clearance, and collision sensor coverage must be
present, so merely following a mathematical fixture cannot qualify a corpus.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .geometry import LaneReference, wrap_angle


@dataclass(frozen=True)
class QualityConfig:
    mean_signed_offset_limit_m: float = .05
    p95_absolute_offset_limit_m: float = .10
    maximum_absolute_offset_limit_m: float = .20
    boundary_clearance_m: float = .15
    # Explicit user revision: maximum lateral acceleration is 3.5 m/s^2.
    lateral_acceleration_limit_mps2: float = 3.5
    # Explicit user revision: 5 m/s^3 is acceptable. Preserve measured values.
    lateral_jerk_limit_mps3: float = 5.
    recovery_limit_s: float = 4.
    recovery_offset_tolerance_m: float = .10
    recovery_heading_tolerance_rad: float = math.radians(1.)
    recovery_hold_tolerance_m: float = .05
    recovery_overshoot_limit_m: float = .05
    maximum_held_recovery_offset_m: float = .30
    saturation_fraction: float = .98
    sustained_saturation_s: float = .25
    maximum_sweep_spacing_m: float = .05
    maximum_sweep_heading_step_rad: float = math.radians(.5)
    maximum_boundary_chord_spacing_m: float = .25


@dataclass(frozen=True)
class QualityReport:
    accepted: bool
    reasons: tuple[str,...]
    metrics: dict

    def to_dict(self):
        return asdict(self)


def recovery_envelope(timestamps_s, offsets_m, onset_s, config=None, *, end_s=None):
    """Bound the recovery exemption using the measured signed release offset.

    A declared recovery may start off center; it may not hide an unrelated
    excursion in the recurrent prefix or a large overshoot across the lane.
    This supplements convergence, body clearance and measured dynamics.
    """
    c = config or QualityConfig()
    t, offset = np.asarray(timestamps_s, float), np.asarray(offsets_m, float)
    if (t.ndim != 1 or len(t) < 3 or offset.shape != t.shape or
            not np.isfinite(t).all() or not np.isfinite(offset).all() or
            np.any(np.diff(t) <= 0) or not math.isfinite(onset_s) or
            onset_s < 0 or onset_s >= t[-1]-t[0]):
        raise ValueError("recovery envelope requires finite timed offsets and an observed onset")
    elapsed = t-t[0]
    index = int(np.argmin(np.abs(elapsed-onset_s)))
    errors = []
    if abs(float(elapsed[index])-onset_s) > 1e-6:
        errors.append("recovery_onset_not_recorded_frame")
    end_s = min(float(elapsed[-1]), onset_s+c.recovery_limit_s) if end_s is None else end_s
    if not math.isfinite(end_s) or not onset_s <= end_s <= elapsed[-1]+1e-6:
        raise ValueError("recovery envelope end must lie after onset in the recording")
    initial = float(offset[index])
    prefix_error = float(np.max(np.abs(offset[:index+1]-initial)))
    if onset_s > 0:
        if abs(initial) < .15 or abs(initial) > c.maximum_held_recovery_offset_m+c.recovery_hold_tolerance_m:
            errors.append("held_recovery_initial_offset_outside_allowed_range")
        if prefix_error > c.recovery_hold_tolerance_m:
            errors.append("recovery_prefix_does_not_hold_initial_offset")
    direction = 1. if initial >= 0 else -1.
    end_index = max(index+1, int(np.searchsorted(elapsed, end_s+1e-6, side="right")))
    signed = offset[index:end_index]*direction
    outward = float(max(0., np.max(signed)-abs(initial)))
    overshoot = float(max(0., -np.min(signed)))
    if outward > c.recovery_hold_tolerance_m:
        errors.append("recovery_moves_farther_from_center")
    if overshoot > c.recovery_overshoot_limit_m:
        errors.append("recovery_overshoot_above_limit")
    return dict(passed=not errors, errors=errors, onset_index=index, end_s=float(end_s),
                signed_release_offset_m=initial, maximum_prefix_offset_error_m=prefix_error,
                maximum_outward_excursion_m=outward, maximum_center_overshoot_m=overshoot,
                hold_tolerance_m=c.recovery_hold_tolerance_m,
                overshoot_limit_m=c.recovery_overshoot_limit_m)


def swept_body_clearance(reference: LaneReference, positions_xyz, yaw_rad,
                         vehicle_length_m: float, vehicle_width_m: float,
                         config: QualityConfig | None = None, *, rotation_world=None,
                         vehicle_height_m=0., vehicle_bbox_center_xyz=(0.,0.,0.)):
    """Conservative sampled continuous body sweep against independent boundaries.

    Each inter-frame pose interval is subdivided to <=5 cm translation and
    <=0.5 degree yaw, and boundary clearance is reduced by that interval's
    maximum corner travel. The margin accounts for motion between samples.
    If full CARLA roll/pitch/yaw and the actor-local bounding box are supplied,
    all eight transformed box corners define the convex horizontal footprint.
    Distances and intersections use the whole polygon, including edge interiors.
    Boundary splines are approximated by chords with a certified second-
    derivative error bound, which is also subtracted from the clearance.
    This accounts for bank/grade and the box's offset from the actor origin.
    Vertical clearance is outside this lateral lane-boundary calculation.
    """
    c=config or QualityConfig(); p=np.asarray(positions_xyz,float); yaw=np.unwrap(np.asarray(yaw_rad,float))
    if p.shape != (len(yaw),3) or len(yaw) < 2 or not np.isfinite(p).all():
        raise ValueError("sweep needs matching dense [N,3] positions and yaw")
    if not all(math.isfinite(value) and value>0 for value in (
        c.maximum_sweep_spacing_m,c.maximum_sweep_heading_step_rad,
        c.maximum_boundary_chord_spacing_m)):
        raise ValueError("sweep and boundary sampling limits must be finite and positive")
    from scipy.spatial.transform import Rotation, Slerp
    from .traffic import convex_hull, polyline_polygon_distance
    origin=np.asarray(vehicle_bbox_center_xyz,float)
    if origin.shape!=(3,) or not np.isfinite(origin).all() or not np.isfinite([vehicle_length_m,vehicle_width_m,vehicle_height_m]).all() or min(vehicle_length_m,vehicle_width_m)<=0 or vehicle_height_m<0:
        raise ValueError("vehicle bounding box must be finite with positive length/width")
    corners=np.array([[sx*vehicle_length_m/2,sy*vehicle_width_m/2,sz*vehicle_height_m/2]
                      for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)])+origin
    rpy=np.column_stack([np.zeros_like(yaw),np.zeros_like(yaw),yaw]) if rotation_world is None else np.asarray(rotation_world,float)
    if rpy.shape!=(len(p),3) or not np.isfinite(rpy).all() or not np.allclose(wrap_angle(rpy[:,2]-yaw),0,atol=1e-7):
        raise ValueError("body rotations must match dense yaw and positions")
    rotations=Rotation.from_euler("xyz",rpy)
    reflect=np.diag([1.,1.,-1.])
    # Every cubic-spline knot is included. Within a knot interval its second
    # derivative is linear, so the larger endpoint norm bounds its norm over
    # that interval. M*h^2/8 bounds the curve-to-chord Hausdorff distance.
    arc=np.unique(np.r_[reference.s,np.arange(reference.s[0],reference.s[-1],
                                            c.maximum_boundary_chord_spacing_m)])
    boundaries=[]
    for spline in (reference._left,reference._right):
        second=np.linalg.norm(spline(arc,2)[:,:2],axis=1)
        error=float(np.max(np.maximum(second[:-1],second[1:])*np.diff(arc)**2/8))
        boundaries.append((spline(arc)[:,:2],error))
    maximum_chord_error=max(error for _,error in boundaries)
    minimum=float("inf"); crossing=False; hint=None; maximum_motion_margin=0.
    corner_radius=float(np.max(np.linalg.norm(corners,axis=1)))
    for i in range(len(p)-1):
        distance=float(np.linalg.norm(p[i+1]-p[i])); angle=float((rotations[i].inv()*rotations[i+1]).magnitude())
        steps=max(1,math.ceil(distance/c.maximum_sweep_spacing_m),math.ceil(angle/c.maximum_sweep_heading_step_rad))
        uncertainty=(distance+corner_radius*angle)/(2*steps)
        maximum_motion_margin=max(maximum_motion_margin,uncertainty)
        fractions=np.linspace(0,1,steps+1)
        matrices=reflect@Slerp([0.,1.],rotations[i:i+2])(fractions).as_matrix()@reflect
        for fraction,matrix in zip(fractions,matrices):
            point=p[i]*(1-fraction)+p[i+1]*fraction
            center_s,_=reference.project(point[:2],hint_s=hint); hint=center_s
            world_corners=(point+corners@matrix.T)[:,:2]
            polygon=convex_hull(world_corners)
            # Boundary distance alone would be positive for a body wholly
            # outside the lane. A connected polygon with an interior point
            # inside the corridor and no boundary intersection is wholly inside.
            s,offset=reference.project(world_corners.mean(axis=0),hint_s=center_s)
            center,heading,_,_=reference.sample(s)
            normal=np.array([-math.sin(heading),math.cos(heading)])
            left_offset=float(np.dot(reference._left(s)[:2]-center[:2],normal))
            right_offset=float(np.dot(reference._right(s)[:2]-center[:2],normal))
            center_clearance=min(offset-left_offset,right_offset-offset)
            if center_clearance<=0:
                minimum=min(minimum,center_clearance-uncertainty)
                crossing=True
            for boundary,error in boundaries:
                clearance=polyline_polygon_distance(boundary,polygon)-error-uncertainty
                minimum=min(minimum,clearance)
                crossing |= clearance<=0
    return {"minimum_boundary_clearance_m": float(minimum), "swept_boundary_crossing": bool(crossing),
            "sweep_max_spacing_m":c.maximum_sweep_spacing_m,
            "sweep_max_heading_step_rad":c.maximum_sweep_heading_step_rad,
            "boundary_chord_max_spacing_m":c.maximum_boundary_chord_spacing_m,
            "boundary_chord_error_bound_m":maximum_chord_error,
            "maximum_temporal_motion_margin_m":maximum_motion_margin,
            "body_clearance_method":"whole_convex_polygon_with_boundary_chord_and_temporal_error_bounds",
            "body_footprint_mode":"transformed_3d_bounding_box" if rotation_world is not None and vehicle_height_m>0 else "planar_fixture"}


def qualify_episode(*, timestamps_s, positions_xyz, yaw_rad, speeds_mps,
                    lateral_acceleration_mps2, steering_fractions,
                    reference: LaneReference, vehicle_length_m: float, vehicle_width_m: float,
                    collisions: int | None, rendered_boundary_agreement: bool | None,
                    completed_maneuver: bool, intended_progress_m: float,
                    recovery=False, config: QualityConfig | None = None, velocity_world=None,
                    rotation_world=None, vehicle_height_m=0.,vehicle_bbox_center_xyz=(0.,0.,0.),recovery_onset_s=0.):
    """Evaluate exact requested gates, with no smoothing of measured dynamics.

    ``lateral_acceleration_mps2`` must come from recorded world acceleration
    resolved onto the instantaneous vehicle lateral axis (not a controller
    target or future speed). Derivative jerk uses actual timestamp intervals.
    Exempt only an intentional initial recovery interval from center error gates.
    """
    c=config or QualityConfig(); t=np.asarray(timestamps_s,float); p=np.asarray(positions_xyz,float)
    yaw=np.asarray(yaw_rad,float); v=np.asarray(speeds_mps,float)
    ay=np.asarray(lateral_acceleration_mps2,float); steer=np.asarray(steering_fractions,float)
    if len(t)<3 or p.shape!=(len(t),3) or any(a.shape!=t.shape for a in (yaw,v,ay,steer)):
        raise ValueError("quality requires matching dense telemetry arrays")
    if not all(np.isfinite(a).all() for a in (t,p,yaw,v,ay,steer)) or np.any(np.diff(t)<=0):
        raise ValueError("quality telemetry must be finite and strictly chronological")
    reasons=[]
    spacing=np.diff(t)
    if not np.allclose(spacing,.05,rtol=0,atol=1e-5): reasons.append("missing_duplicate_or_non20hz_frames")
    # The CARLA actor reference can be ahead of the rear axle. On a curve its
    # body yaw is therefore different from the path tangent even when centered.
    # Recovery convergence uses the recorded velocity direction; swept-body
    # clearance below still uses actual body orientation.
    velocity=(np.gradient(p,t,axis=0) if velocity_world is None else np.asarray(velocity_world,float))
    if velocity.shape!=p.shape or not np.isfinite(velocity).all():
        raise ValueError("velocity_world must be a finite [N,3] array")
    motion_yaw=np.arctan2(velocity[:,1],velocity[:,0])
    arc=[];offset=[];heading_error=[];hint=None
    for position,angle in zip(p,motion_yaw):
        s,d=reference.project(position[:2],hint);hint=s;arc.append(s);offset.append(d)
        heading_error.append(float(wrap_angle(angle-reference.sample(s)[1])))
    arc=np.asarray(arc);offset=np.asarray(offset);heading_error=np.asarray(heading_error)
    settled=np.ones(len(t),bool); recovery_time=0.; recovery_measurement=None
    if not recovery and recovery_onset_s:
        reasons.append("recovery_onset_without_declared_recovery")
    if recovery:
        if not math.isfinite(recovery_onset_s) or recovery_onset_s<0 or recovery_onset_s>=t[-1]-t[0]:
            raise ValueError('recovery onset must lie in the retained physical recording')
        # Convergence must persist, not momentarily cross the lane center.
        body_heading=np.array([float(wrap_angle(angle-reference.sample(s)[1])) for angle,s in zip(yaw,arc)])
        recovery_heading=np.where(v>.2,heading_error,body_heading)
        within=(np.abs(offset)<=c.recovery_offset_tolerance_m)&(np.abs(recovery_heading)<=c.recovery_heading_tolerance_rad)
        persistent=np.logical_and.accumulate(within[::-1])[::-1]
        indices=np.flatnonzero(persistent & (v>.2) & (t-t[0]>=recovery_onset_s-1e-8))
        recovery_time=float(t[indices[0]]-t[0]-recovery_onset_s) if len(indices) else float("inf")
        recovery_measurement=recovery_envelope(t,offset,recovery_onset_s,c,
            end_s=min(float(t[-1]-t[0]), recovery_onset_s+min(c.recovery_limit_s,recovery_time)))
        reasons.extend(recovery_measurement['errors'])
        if recovery_time>c.recovery_limit_s: reasons.append("recovery_did_not_converge_within_four_seconds")
        settled=(t-t[0]>=recovery_onset_s+min(c.recovery_limit_s,recovery_time))
    if not np.any(settled): reasons.append("no_post_recovery_samples")
    post=offset[settled] if np.any(settled) else offset
    signed=float(np.mean(post)); p95=float(np.quantile(np.abs(post),.95)); maximum=float(np.max(np.abs(post)))
    if abs(signed)>c.mean_signed_offset_limit_m: reasons.append("mean_signed_center_offset")
    if p95>c.p95_absolute_offset_limit_m: reasons.append("p95_absolute_center_offset")
    if maximum>c.maximum_absolute_offset_limit_m: reasons.append("maximum_absolute_center_offset")
    sweep=swept_body_clearance(reference,p,yaw,vehicle_length_m,vehicle_width_m,c,
        rotation_world=rotation_world,vehicle_height_m=vehicle_height_m,vehicle_bbox_center_xyz=vehicle_bbox_center_xyz)
    if sweep["swept_boundary_crossing"]: reasons.append("swept_vehicle_body_lane_boundary_crossing")
    if sweep["minimum_boundary_clearance_m"]<c.boundary_clearance_m: reasons.append("boundary_clearance_below_fifteen_cm")
    max_ay=float(np.max(np.abs(ay))); max_jerk=float(np.max(np.abs(np.diff(ay)/spacing)))
    if max_ay>c.lateral_acceleration_limit_mps2: reasons.append("lateral_acceleration_above_limit")
    if max_jerk>c.lateral_jerk_limit_mps3: reasons.append("lateral_jerk_above_limit")
    duration=0.;max_saturation=0.
    for active,dt in zip(np.abs(steer[:-1])>=c.saturation_fraction,spacing):
        duration=duration+dt if active else 0.;max_saturation=max(max_saturation,duration)
    if max_saturation>=c.sustained_saturation_s: reasons.append("sustained_actuator_saturation")
    if collisions is None: reasons.append("collision_sensor_evidence_missing")
    elif collisions!=0: reasons.append("collision")
    if rendered_boundary_agreement is not True: reasons.append("rendered_boundary_agreement_missing_or_failed")
    progress=float(arc[-1]-arc[0])
    if not completed_maneuver: reasons.append("maneuver_incomplete")
    if progress+1e-3<intended_progress_m or np.min(np.diff(arc))<-.05:
        reasons.append("intended_progress_failed")
    metrics={"mean_signed_center_offset_m":signed,"p95_absolute_center_offset_m":p95,
        "maximum_absolute_center_offset_m":maximum,"maximum_lateral_acceleration_mps2":max_ay,
        "maximum_lateral_jerk_mps3":max_jerk,"maximum_saturation_duration_s":max_saturation,
        "recovery_time_s":recovery_time,"progress_m":progress,"duration_s":float(t[-1]-t[0]),
        "minimum_speed_mps":float(v.min()),"maximum_speed_mps":float(v.max()),
        "quality_limits":asdict(c),"recovery_envelope":recovery_measurement,**sweep}
    return QualityReport(not reasons,tuple(reasons),metrics)
