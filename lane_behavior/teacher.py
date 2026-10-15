"""Physical feedback teacher for explicitly privileged planned geometry.

This controller sends ordinary steering/throttle/brake commands. It never sets
pose or velocity, constructs visual evidence, or supplies student labels. Actual
camera observability and measured physical QA must independently admit labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull

from lane_dataset.controller import SteeringCalibration, VehicleState
from .planned import PlannedTarget, SOURCE_SCOPE
from .passing import permitted_ribbon
from .target import _body, _boundary_distance, _geometry, _inside, _polygon_distance


@dataclass(frozen=True)
class TeacherConfig:
    cruise_speed_mps: float = 3.5
    max_lateral_acceleration_mps2: float = 3.5
    max_command_lateral_jerk_mps3: float = 3.
    max_steering_rate_rps: float = .35
    max_steering_acceleration_rps2: float = 1.5
    acceleration_mps2: float = .8
    comfortable_deceleration_mps2: float = 1.5
    heading_gain: float = 1.6
    cross_track_gain: float = .9
    feedback_soft_speed_mps: float = 1.5
    preview_extra_s: float = .04
    speed_p: float = .35
    speed_i: float = .08
    speed_feedforward: float = .10
    throttle_rate_per_s: float = .6
    brake_rate_per_s: float = 2.
    stop_buffer_m: float = 1.
    stopped_speed_mps: float = .12
    maximum_cross_track_error_m: float = .6
    maximum_heading_error_rad: float = .8
    maximum_step_s: float = .25
    initial_step_s: float = .05
    contract_check_interval_s: float = .5
    require_calibration: bool = True

    def __post_init__(self):
        numeric = [v for k, v in vars(self).items() if k != 'require_calibration']
        if (not np.isfinite(numeric).all() or min(numeric) <= 0
                or self.cruise_speed_mps > 3.5 or self.max_lateral_acceleration_mps2 > 3.5
                or self.max_command_lateral_jerk_mps3 > 5.
                or self.max_steering_acceleration_rps2 > 1.5
                or self.contract_check_interval_s > 1.
                or self.initial_step_s > self.maximum_step_s):
            raise ValueError("invalid bounded low-speed teacher settings")


@dataclass(frozen=True)
class TeacherCommand:
    accepted: bool
    throttle: float
    brake: float
    steer: float
    phase: str
    reasons: tuple[str, ...]
    target_speed_mps: float
    path_arc_m: float
    road_station_m: float
    cross_track_error_m: float
    heading_error_rad: float
    steering_angle_rad: float
    source_scope: str = SOURCE_SCOPE
    supervision_allowed: bool = False
    diagnostics: dict = field(default_factory=dict)

    @property
    def vehicle_control(self):
        return dict(throttle=self.throttle, brake=self.brake, steer=self.steer)


def _readonly(value):
    value = np.asarray(value, float).copy()
    value.flags.writeable = False
    return value


def _wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


class PhysicalTeacher:
    """Follow a PlannedTarget with real telemetry and a measured steering map.

    ``step(VehicleState)`` returns controls to apply once through CARLA's normal
    VehicleControl/apply_control API. Feed subsequent measured telemetry back.
    Startup and stopping below the calibration's narrow cruise-speed range are
    explicit physical phases, not counterfeit observed expert commands.

    Immutable path/obstacle copies are prepared once. The original contract is
    revalidated at most every configured interval; its digest/provenance flags
    are checked every tick. External mutation cannot alter the cached path.
    """
    def __init__(self, target: PlannedTarget, calibration: SteeringCalibration, *, config=None):
        if not isinstance(target, PlannedTarget):
            raise TypeError("physical teacher requires distinct PlannedTarget privileged geometry")
        if not isinstance(calibration, SteeringCalibration):
            raise TypeError("measured SteeringCalibration required")
        self.config = config or TeacherConfig()
        self.calibration = calibration
        if self.config.require_calibration and not calibration.qualified:
            raise ValueError("physical teacher requires qualified measured steering calibration")
        if (not np.isfinite([calibration.wheelbase_m, calibration.steering_gain_rad,
                calibration.delay_s, calibration.time_constant_s, calibration.reference_offset_m,
                calibration.max_steering_rad, calibration.steering_fraction_min,
                calibration.steering_fraction_max]).all()
                or calibration.wheelbase_m <= 0 or calibration.steering_gain_rad <= 0
                or calibration.max_steering_rad <= 0
                or not -1. <= calibration.steering_fraction_min < calibration.steering_fraction_max <= 1.
                or min(calibration.delay_s, calibration.time_constant_s) < 0):
            raise ValueError("invalid physical steering calibration")
        if (self.config.require_calibration
                and self.config.cruise_speed_mps > calibration.speed_max_mps+.1):
            raise ValueError("configured cruise exceeds measured steering calibration speed")
        self._source = target
        self.contract_sha256 = target.contract_sha256
        self._blocked = ()
        self._last_timestamp = None
        self._last_contract_check = None
        self._last_arc = None
        self._last_body = None
        self._last_yaw = None
        self._last_angle = 0.
        self._last_steering_rate = 0.
        self._previous_throttle = 0.
        self._previous_brake = 0.
        self._speed_integral = 0.
        self._speed_target = None
        self._angle_min = max(-calibration.max_steering_rad,
            calibration.angle_from_steer(calibration.steering_fraction_min))
        self._angle_max = min(calibration.max_steering_rad,
            calibration.angle_from_steer(calibration.steering_fraction_max))
        try:
            target.verify_contract()
        except ValueError:
            self._blocked = ('planned_target_contract_invalid',)
            return
        if not target.feasible:
            self._blocked = ('planned_target_infeasible', *target.reasons)
            return
        profile = target.profile()
        self._arc = _readonly(profile['path_arc_m'])
        self._source_station = _readonly(profile['road_station_m'])
        self._xy = _readonly(np.asarray(profile['position_xyz'])[:, :2])
        self._curve = CubicSpline(self._arc, self._xy, axis=0, extrapolate=False)
        ref = target.reference
        q = np.linspace(ref.s[0], ref.s[-1], int(np.ceil((ref.s[-1]-ref.s[0])/.2))+1)
        _, _, divider, right = _geometry(ref, q)
        self._ribbon = _readonly(np.concatenate((divider, right[::-1])))
        if target.passing_space is not None:
            self._ribbon=_readonly(permitted_ribbon(ref,.2,target.excursions,target.passing_space))
        self._corners = _readonly(target.ego.corners())
        self._radius = float(np.linalg.norm(self._corners, axis=1).max())
        self._edge_margin = float(target.config.edge_clearance_m)
        self._obstacles = tuple((o.identity, _readonly(o.polygon()), float(target.config.obstacle_clearance(o.kind)))
            for o in target.obstacles)
        # Geometric infeasibility must be found before the first throttle.
        _, _, curvature = self._sample(self._arc)
        offset = calibration.reference_offset_m*curvature
        if np.any(np.abs(offset) >= .95):
            self._blocked = ('planned_curvature_exceeds_actor_reference_model',)
            return
        required = np.arctan(calibration.wheelbase_m*curvature/np.sqrt(1-offset**2))
        if required.min() < self._angle_min-1e-6 or required.max() > self._angle_max+1e-6:
            self._blocked = ('planned_curvature_exceeds_calibrated_steering_range',)

    def _sample(self, arc):
        q = np.clip(arc, self._arc[0], self._arc[-1])
        xy = self._curve(q)
        first, second = self._curve(q, 1), self._curve(q, 2)
        yaw = np.arctan2(first[..., 1], first[..., 0])
        curvature = (first[..., 0]*second[..., 1]-first[..., 1]*second[..., 0]) / np.maximum(
            np.linalg.norm(first, axis=-1)**3, 1e-12)
        return xy, yaw, curvature

    def _project(self, position, speed, dt):
        if self._last_arc is None:
            first, last = 0, len(self._arc)-1
        else:
            first = max(0, int(np.searchsorted(self._arc, self._last_arc-2.))-1)
            last = min(len(self._arc)-1, int(np.searchsorted(self._arc,
                self._last_arc+max(5., speed*dt+1.)))+1)
        a, delta = self._xy[first:last], np.diff(self._xy[first:last+1], axis=0)
        fraction = np.clip(((position-a)*delta).sum(1)/np.maximum((delta*delta).sum(1), 1e-12), 0, 1)
        nearest = int(np.argmin(np.linalg.norm(position-a-fraction[:, None]*delta, axis=1)))+first
        lo, hi = self._arc[max(0, nearest-1)], self._arc[min(len(self._arc)-1, nearest+2)]
        solved = minimize_scalar(lambda q: float(np.sum((self._curve(q)-position)**2)),
            bounds=(lo, hi), method='bounded', options={'xatol': 1e-7})
        return float(np.clip(solved.x, self._arc[0], self._arc[-1]))

    def _body_check(self, state):
        direction = np.array([[math.cos(state.yaw_rad), math.sin(state.yaw_rad)]])
        body = _body(np.array([[state.x_m, state.y_m]]), direction, self._corners)[0]
        cloud = body if self._last_body is None else np.concatenate((self._last_body, body))
        swept = cloud[ConvexHull(cloud).vertices]
        angle = 0. if self._last_yaw is None else abs(_wrap(state.yaw_rad-self._last_yaw))
        guard = self._radius*angle*angle/8
        if (not _inside(swept, self._ribbon).all()
                or _boundary_distance(swept, self._ribbon) < self._edge_margin+guard-1e-7):
            return None, ('measured_body_leaves_planned_right_half' if self._source.passing_space is None
                else 'measured_body_leaves_planned_permitted_road_space')
        for name, polygon, margin in self._obstacles:
            if _polygon_distance(swept, polygon) < margin+guard-1e-7:
                return None, 'measured_body_obstacle_clearance:'+name
        return body, None

    def _reject(self, reasons):
        steer = float(np.clip(self.calibration.steer_from_angle(self._last_angle),
            self.calibration.steering_fraction_min, self.calibration.steering_fraction_max))
        return TeacherCommand(False, 0., 1., steer, 'blocked', tuple(reasons), 0.,
            float(self._last_arc or 0.), 0., 0., 0., self._last_angle,
            diagnostics={'planned_contract_sha256': self.contract_sha256,
                'physical_qa_passed': False, 'labels_generated': False})

    def _steering_command(self, desired, speed, dt):
        """Shape wheel angle in rate and acceleration, including actor offset.

        A rate bound alone permits instantaneous rate reversal. At an actor
        reference point ahead of the rear axle, that creates lateral jerk via
        reference_offset * d(yaw_rate)/dt. Reserve command jerk for both terms
        v**2/L * delta_dot and reference_offset*v/L * delta_ddot. This remains
        a conservative low-speed command model, not measured physical QA.
        """
        cfg, cal = self.config, self.calibration
        # Include the entire next reachable angle interval in the secant bound.
        peak_angle = max(abs(desired), abs(self._last_angle)
            + abs(self._last_steering_rate)*dt + .5*cfg.max_steering_acceleration_rps2*dt**2)
        peak_angle = min(peak_angle, max(abs(self._angle_min), abs(self._angle_max)))
        secant_squared = 1.+math.tan(peak_angle)**2
        rate_coefficient = speed**2*secant_squared/cal.wheelbase_m
        acceleration_coefficient = abs(cal.reference_offset_m)*speed*secant_squared/cal.wheelbase_m
        # Leave at least half the jerk budget available to reference-point
        # steering acceleration. Zero-offset calibrations need no such share.
        share = .5 if acceleration_coefficient > 1e-12 else 1.
        rate_limit = min(cfg.max_steering_rate_rps,
            share*cfg.max_command_lateral_jerk_mps3/max(rate_coefficient, 1e-12))
        rate_budget = rate_coefficient*max(rate_limit, abs(self._last_steering_rate))
        acceleration_limit = min(cfg.max_steering_acceleration_rps2,
            max(0., cfg.max_command_lateral_jerk_mps3-rate_budget)
            / max(acceleration_coefficient, 1e-12))
        if acceleration_limit <= 1e-12:
            return None
        error = desired-self._last_angle
        # Brake the commanded rate before reaching the requested angle. The
        # discrete-step term avoids a bang-bang limit cycle around a fixed goal.
        stop_rate = math.sqrt(2.*acceleration_limit*abs(error)
            + (.5*acceleration_limit*dt)**2)-.5*acceleration_limit*dt
        wanted_rate = math.copysign(min(rate_limit, stop_rate), error)
        low = max(-rate_limit, self._last_steering_rate-acceleration_limit*dt)
        high = min(rate_limit, self._last_steering_rate+acceleration_limit*dt)
        if low > high:
            return None
        rate = float(np.clip(wanted_rate, low, high))
        angle = self._last_angle+rate*dt
        if not self._angle_min-1e-10 <= angle <= self._angle_max+1e-10:
            return None
        acceleration = (rate-self._last_steering_rate)/dt
        return angle, rate, acceleration, dict(
            command_steering_rate_rps=rate,
            command_steering_acceleration_rps2=acceleration,
            command_steering_rate_limit_rps=rate_limit,
            command_steering_acceleration_limit_rps2=acceleration_limit,
            predicted_command_lateral_jerk_bound_mps3=(
                rate_coefficient*abs(rate)+acceleration_coefficient*abs(acceleration)),
            jerk_model_reference_offset_m=cal.reference_offset_m,
            jerk_model_scope='low_speed_command_bound_not_measured_physical_qa')

    def step(self, state: VehicleState):
        if not isinstance(state, VehicleState):
            raise TypeError("teacher step requires actual VehicleState telemetry")
        values = [state.x_m, state.y_m, state.yaw_rad, state.speed_mps,
            state.yaw_rate_rps, state.timestamp_s]
        values.extend(v for v in (state.steering_rad, state.lateral_acceleration_mps2) if v is not None)
        if not np.isfinite(values).all() or state.speed_mps < 0:
            return self._reject(('invalid_vehicle_telemetry',))
        if self._blocked:
            return self._reject(self._blocked)
        cfg, cal = self.config, self.calibration
        dt = cfg.initial_step_s if self._last_timestamp is None else state.timestamp_s-self._last_timestamp
        if not 0 < dt <= cfg.maximum_step_s:
            return self._reject(('nonmonotonic_or_gapped_vehicle_telemetry',))
        if (self._source.contract_sha256 != self.contract_sha256
                or self._source.source_scope != SOURCE_SCOPE or self._source.supervision_allowed is not False):
            self._blocked = ('planned_target_contract_invalid',)
            return self._reject(self._blocked)
        if (self._last_contract_check is None
                or state.timestamp_s-self._last_contract_check >= cfg.contract_check_interval_s):
            try:
                self._source.verify_contract()
            except ValueError:
                self._blocked = ('planned_target_contract_invalid',)
                return self._reject(self._blocked)
            self._last_contract_check = state.timestamp_s
        body, failure = self._body_check(state)
        if failure:
            return self._reject((failure,))
        position = np.array([state.x_m, state.y_m])
        arc = self._project(position, state.speed_mps, dt)
        point, yaw, curvature = self._sample(arc)
        cross_track = float((position-point)@np.array([-math.sin(yaw), math.cos(yaw)]))
        beta = math.asin(float(np.clip(cal.reference_offset_m*curvature, -.95, .95)))
        heading_error = _wrap(state.yaw_rad-float(yaw)+beta)
        if abs(cross_track) > cfg.maximum_cross_track_error_m or abs(heading_error) > cfg.maximum_heading_error_rad:
            return self._reject(('teacher_tracking_error_outside_recovery_envelope',))
        remaining = max(0., float(self._arc[-1]-arc))
        preview = state.speed_mps*(cal.delay_s+cal.time_constant_s+cfg.preview_extra_s)
        _, _, ahead_curvature = self._sample(min(self._arc[-1], arc+preview))
        turn_scale = math.sqrt(max(1e-6, 1-(cal.reference_offset_m*float(ahead_curvature))**2))
        feedforward = math.atan(cal.wheelbase_m*float(ahead_curvature)/turn_scale)
        desired_angle = feedforward-cfg.heading_gain*heading_error-math.atan2(
            cfg.cross_track_gain*cross_track, max(state.speed_mps, cfg.feedback_soft_speed_mps))
        ay_angle = math.atan(cal.wheelbase_m*.85*cfg.max_lateral_acceleration_mps2/max(state.speed_mps**2, .25))
        desired_angle = float(np.clip(desired_angle, max(self._angle_min, -ay_angle), min(self._angle_max, ay_angle)))
        if self._last_timestamp is None:
            self._last_angle = float(np.clip(state.steering_rad if state.steering_rad is not None else
                math.atan(cal.wheelbase_m*state.yaw_rate_rps/max(state.speed_mps, .5)), self._angle_min, self._angle_max))
        shaped = self._steering_command(desired_angle, state.speed_mps, dt)
        if shaped is None:
            return self._reject(('steering_command_dynamics_infeasible',))
        angle, steering_rate, steering_acceleration, steering_diagnostics = shaped
        steer = float(np.clip(cal.steer_from_angle(angle), cal.steering_fraction_min, cal.steering_fraction_max))
        # Look ahead far enough to decelerate before a tighter planned bend.
        preview_end = min(self._arc[-1], arc+max(8., state.speed_mps**2/(2*cfg.comfortable_deceleration_mps2)+2.))
        _, _, future_curvature = self._sample(np.linspace(arc, preview_end, 32))
        maximum_curvature = float(np.abs(future_curvature).max())
        curve_speed = math.sqrt(.85*cfg.max_lateral_acceleration_mps2/max(maximum_curvature, 1e-8))
        stop_speed = math.sqrt(2*cfg.comfortable_deceleration_mps2*max(0., remaining-cfg.stop_buffer_m))
        goal_speed = min(cfg.cruise_speed_mps, curve_speed, stop_speed)
        if self._speed_target is None:
            self._speed_target = min(state.speed_mps, cfg.cruise_speed_mps)
        target_speed = float(np.clip(goal_speed, max(0., self._speed_target-cfg.comfortable_deceleration_mps2*dt),
            self._speed_target+cfg.acceleration_mps2*dt))
        error = target_speed-state.speed_mps
        # Stored startup demand must not keep accelerating after reaching the
        # actual cruise/curve/stop cap. Retain negative integral to compensate
        # rolling resistance feedforward, and rebuild positive demand as needed.
        if state.speed_mps >= goal_speed or goal_speed < self._speed_target:
            self._speed_integral = min(0., self._speed_integral)
        self._speed_integral = float(np.clip(self._speed_integral+error*dt, -2., 2.))
        demand = cfg.speed_feedforward+cfg.speed_p*error+cfg.speed_i*self._speed_integral
        wanted_throttle, wanted_brake = float(np.clip(demand, 0., .75)), float(np.clip(-demand, 0., .8))
        phase = 'stopping' if stop_speed < cfg.cruise_speed_mps else ('starting' if target_speed < .9*cfg.cruise_speed_mps else 'tracking')
        if goal_speed < .05:
            wanted_throttle, wanted_brake = 0., max(.25, min(.8, .3+.3*state.speed_mps))
        throttle = float(np.clip(wanted_throttle, max(0., self._previous_throttle-cfg.throttle_rate_per_s*dt),
            min(1., self._previous_throttle+cfg.throttle_rate_per_s*dt)))
        brake = float(np.clip(wanted_brake, max(0., self._previous_brake-cfg.brake_rate_per_s*dt),
            min(1., self._previous_brake+cfg.brake_rate_per_s*dt)))
        if brake > 0:
            throttle = 0.
        if remaining <= cfg.stop_buffer_m+.25 and state.speed_mps <= cfg.stopped_speed_mps:
            phase, throttle, brake, target_speed = 'complete', 0., 1., 0.
        self._last_timestamp, self._last_arc = state.timestamp_s, arc
        self._last_body, self._last_yaw = body, state.yaw_rad
        self._last_angle = angle
        self._last_steering_rate = steering_rate
        self._previous_throttle, self._previous_brake = throttle, brake
        self._speed_target = target_speed
        return TeacherCommand(True, throttle, brake, steer, phase, (), target_speed, arc,
            float(np.interp(arc, self._arc, self._source_station)), cross_track, heading_error, angle,
            diagnostics=dict(planned_contract_sha256=self.contract_sha256, remaining_path_m=remaining,
                desired_steering_angle_rad=desired_angle, planned_curvature_1pm=float(curvature),
                predicted_command_lateral_acceleration_mps2=state.speed_mps**2*math.tan(angle)/cal.wheelbase_m,
                within_measured_cruise_speed=bool(cal.speed_min_mps <= state.speed_mps <= cal.speed_max_mps),
                uses_immutable_planned_snapshot=True, contract_revalidation_interval_s=cfg.contract_check_interval_s,
                **steering_diagnostics,
                physical_qa_passed=False, labels_generated=False))
