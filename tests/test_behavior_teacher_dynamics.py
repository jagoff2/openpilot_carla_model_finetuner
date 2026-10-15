"""Independent rear-axle vehicle integration for the actor-reference teacher.

These are offline dynamics regressions, never CARLA capture qualification.
"""
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest

from lane_dataset.controller import SteeringCalibration, VehicleState
from lane_dataset.geometry import LaneReference
from lane_behavior.planned import PlannedObstacle, planned_scene_geometry
from lane_behavior.target import BehaviorConfig, EgoFootprint
from lane_behavior.teacher import PhysicalTeacher, TeacherConfig
from test_behavior_teacher import scene, configuration


def measured_calibration(delay=0.):
    # Saved 2.5 m/s CARLA identification. Explicit values make this regression
    # portable and independent of the current local calibration file.
    return SteeringCalibration(wheelbase_m=3.0046484375,
        steering_gain_rad=1.107395466644368, max_steering_rad=1.2217303432379762,
        delay_s=delay, time_constant_s=.01770167062652643,
        speed_min_mps=2.454035468997094, speed_max_mps=2.542506694798366,
        validation_yaw_rate_rmse_rps=.013873574414608713, qualified=True,
        sample_count=1889, source='saved measured calibration used in offline physics fixture',
        validation_peak_yaw_rate_error_rps=.0599305642864765,
        steering_fraction_min=-.4, steering_fraction_max=.4,
        steering_cubic_rad=-.7132689039082087, response_rate_rps=3.5759759781358307,
        reference_offset_m=1.6577672150021319)


def independent_motion(teacher, *, start_s=0., end_s=75., speed=2.5):
    """Integrate a rear axle plus rigid offset, with 5 ms actuator substeps.

    It consumes only controls and calibration, not teacher prediction/jerk
    diagnostics. Actor velocity and acceleration come from world-space motion.
    The constant rear speed isolates lateral control from longitudinal tuning.
    """
    cal = teacher.calibration
    xy, tangent, *_ = teacher._source.sample(np.array([start_s]))
    yaw = math.atan2(tangent[0, 1], tangent[0, 0])
    rear = xy[0]-cal.reference_offset_m*np.array([math.cos(yaw), math.sin(yaw)])
    motor = delta = 0.
    h, dt = .005, .05
    queue = [0.]*int(round(cal.delay_s/h))
    previous_velocity = speed*np.array([math.cos(yaw), math.sin(yaw)])
    acceleration = np.zeros(2)
    rows = []
    for index in range(1400):
        forward = np.array([math.cos(yaw), math.sin(yaw)])
        normal = np.array([-forward[1], forward[0]])
        rate = speed*math.tan(delta)/cal.wheelbase_m
        actor = rear+cal.reference_offset_m*forward
        velocity = speed*forward+cal.reference_offset_m*rate*normal
        command = teacher.step(VehicleState(*actor, yaw, float(np.linalg.norm(velocity)), rate,
            steering_rad=delta, timestamp_s=index*dt,
            lateral_acceleration_mps2=float(acceleration@normal)))
        rows.append(dict(t=index*dt, actor=actor.copy(), yaw=yaw,
            velocity=velocity.copy(), lateral_acceleration=float(acceleration@normal), command=command))
        if not command.accepted or command.road_station_m >= end_s:
            return rows
        requested = cal.angle_from_steer(command.steer)
        for _ in range(10):
            queue.append(requested)
            delayed = queue.pop(0)
            motor += float(np.clip(delayed-motor, -cal.response_rate_rps*h, cal.response_rate_rps*h))
            delta += (1-math.exp(-h/cal.time_constant_s))*(motor-delta)
            rate = speed*math.tan(delta)/cal.wheelbase_m
            middle = yaw+.5*rate*h
            rear += speed*np.array([math.cos(middle), math.sin(middle)])*h
            yaw += rate*h
            forward = np.array([math.cos(yaw), math.sin(yaw)])
            normal = np.array([-forward[1], forward[0]])
            velocity = speed*forward+cal.reference_offset_m*rate*normal
            acceleration = (velocity-previous_velocity)/h
            previous_velocity = velocity
    raise AssertionError('offline dynamics fixture did not finish')


def assert_dynamics(rows):
    commands = [r['command'] for r in rows]
    assert commands[-1].accepted, (rows[-1]['actor'], commands[-1].reasons)
    angles = np.r_[0., [c.steering_angle_rad for c in commands]]
    rates = np.r_[0., np.diff(angles)/.05]
    assert np.abs(rates).max() <= .35+1e-10
    assert np.abs(np.diff(rates)/.05).max() <= 1.5+1e-9
    acceleration = np.array([r['lateral_acceleration'] for r in rows])
    assert np.abs(acceleration).max() < 3.5
    assert np.abs(np.diff(acceleration)/.05).max() < 5.
    assert max(abs(c.cross_track_error_m) for c in commands) < .1
    assert max(c.diagnostics['predicted_command_lateral_jerk_bound_mps3'] for c in commands) <= 3.+1e-10


@pytest.mark.parametrize('delay', [0., .1])
def test_actor_reference_avoidance_and_recovery_with_measured_actuator_and_delay(delay):
    original = scene()
    target = planned_scene_geometry(original.reference, original.obstacles, original.ego,
        config=original.config, obstacle_tracking_reserve_m=.1)
    teacher = PhysicalTeacher(target, measured_calibration(delay), config=configuration())
    rows = independent_motion(teacher)
    assert_dynamics(rows)
    assert rows[-1]['command'].road_station_m >= 75.
    assert min(r['actor'][1] for r in rows) < -.3
    assert max(abs(r['actor'][1]) for r in rows if r['actor'][0] > 65.) < .04


def test_acceleration_cap_cannot_be_silently_disabled():
    for limit in (0., 1.50001, float('nan')):
        with pytest.raises(ValueError, match='bounded'):
            TeacherConfig(max_steering_acceleration_rps2=limit)


@pytest.mark.parametrize('speed,offset', [(2.5, 1.657767215), (3.5, 1.657767215), (2.5, 0.)])
def test_abrupt_requested_reversal_obeys_combined_rate_and_acceleration_budget(speed, offset):
    cal = replace(measured_calibration(), speed_max_mps=3.5, reference_offset_m=offset)
    teacher = PhysicalTeacher(scene(False), cal, config=configuration(cruise_speed_mps=speed))
    rates = []
    for desired in [.25]*12+[-.25]*18+[0.]*20:
        shaped = teacher._steering_command(desired, speed, .05)
        assert shaped is not None
        angle, rate, acceleration, diagnostics = shaped
        assert abs(acceleration) <= 1.5+1e-12
        assert diagnostics['predicted_command_lateral_jerk_bound_mps3'] <= 3.+1e-10
        teacher._last_angle, teacher._last_steering_rate = angle, rate
        rates.append(rate)
    assert np.max(abs(np.diff(rates)))/.05 <= 1.5+1e-10
    assert min(rates) < 0 < max(rates)


@pytest.mark.parametrize('road', [51, 52])
def test_actual_native_geometry_and_measured_boxes_with_reserve(road):
    directory = Path(r'G:\lane-centering-dataset-v1\reports\unmarked')
    if road == 51:
        path = directory/'captures/20261015T131042-60299648/teacher-target.json'
        if not path.exists(): pytest.skip('local measured native road51 capture required')
        source = json.loads(path.read_text())['source']
    else:
        paths = sorted(directory.glob('*road52*offline-teacher-check-tracking-reserve-0p10.json'))
        if not paths: pytest.skip('local measured native road52 geometry required')
        source = json.loads(paths[-1].read_text())['cases']['parked_car']['target']['source']
    ref = source['reference']
    target = planned_scene_geometry(LaneReference(ref['left_xyz'], ref['right_xyz'], ref['s']),
        [PlannedObstacle(**o) for o in source['obstacles']], EgoFootprint(**source['ego']),
        config=BehaviorConfig(**source['config']), obstacle_tracking_reserve_m=.1)
    teacher = PhysicalTeacher(target, measured_calibration(), config=configuration())
    rows = independent_motion(teacher, start_s=10., end_s=target.excursions[-1].end_s+12.)
    assert_dynamics(rows)
