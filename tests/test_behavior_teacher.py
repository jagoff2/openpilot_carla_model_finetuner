from dataclasses import replace
import math
import time

import numpy as np
import pytest

from lane_dataset.controller import SteeringCalibration, VehicleState
from lane_dataset.geometry import LaneReference
from lane_behavior.planned import PlannedObstacle, planned_scene_geometry, SOURCE_SCOPE
from lane_behavior.target import BehaviorConfig
from lane_behavior.teacher import PhysicalTeacher, TeacherConfig


def scene(obstacle=True, motion='static', *, obstacle_tracking_reserve_m=0.):
    x = np.arange(-10., 106.)
    ref = LaneReference(np.c_[x, x*0-6.75, x*0], np.c_[x, x*0+2.25, x*0], x)
    obstacles = () if not obstacle else (PlannedObstacle('parked',
        [[37.75, 1.2], [42.25, 1.2], [42.25, 3.], [37.75, 3.]], 'parked_vehicle', motion),)
    return planned_scene_geometry(ref, obstacles,
        obstacle_tracking_reserve_m=obstacle_tracking_reserve_m)


def calibration():
    return replace(SteeringCalibration.fixture(delay_s=.1, time_constant_s=.15), qualified=True,
        speed_min_mps=2.454, speed_max_mps=2.542, source='unit-test synthetic measured map')


def configuration(**kwargs):
    return replace(TeacherConfig(cruise_speed_mps=2.5), **kwargs)


def state(x=0., y=0., yaw=0., speed=2.5, timestamp=0., yaw_rate=0., steering=None):
    return VehicleState(x, y, yaw, speed, yaw_rate, steering_rad=steering, timestamp_s=timestamp)


@pytest.mark.parametrize('offset,sign', [(.15, -1), (-.15, 1)])
def test_feedback_sign_corrects_actual_cross_track_error(offset, sign):
    command = PhysicalTeacher(scene(False), calibration(), config=configuration()).step(state(y=offset))
    assert command.accepted, command.reasons
    assert np.sign(command.steer) == sign
    assert abs(command.steering_angle_rad) <= configuration().max_steering_rate_rps*.05+1e-12
    assert command.source_scope == SOURCE_SCOPE and not command.supervision_allowed
    assert command.vehicle_control == dict(throttle=command.throttle, brake=command.brake, steer=command.steer)


def test_starting_below_narrow_calibration_cruise_range_is_explicit_and_allowed():
    command = PhysicalTeacher(scene(False), calibration(), config=configuration()).step(state(speed=0.))
    assert command.accepted and command.phase == 'starting'
    assert command.throttle > 0 and command.brake == 0
    assert not command.diagnostics['within_measured_cruise_speed']
    assert not command.diagnostics['physical_qa_passed'] and not command.diagnostics['labels_generated']


def test_infeasible_or_mutated_scene_brakes_without_counterfeit_supervision():
    bad = PhysicalTeacher(scene(motion='moving'), calibration(), config=configuration()).step(state())
    assert not bad.accepted and bad.throttle == 0 and bad.brake == 1
    target = scene(False)
    target.reference._right.c[3, :, 1] += .1
    bad = PhysicalTeacher(target, calibration(), config=configuration()).step(state())
    assert bad.reasons == ('planned_target_contract_invalid',)
    target = scene(False)
    teacher = PhysicalTeacher(target, calibration(), config=configuration())
    assert teacher.step(state()).accepted
    target.reference._right.c[3, :, 1] += .1
    # Immutable snapshot is unchanged immediately; scheduled check then detects
    # mutation rather than following the modified caller-owned geometry.
    for i in range(1, 11):
        command = teacher.step(state(x=i*.125, timestamp=i*.05))
    assert not command.accepted and command.reasons == ('planned_target_contract_invalid',)


def test_actual_body_clearance_and_telemetry_order_are_enforced():
    teacher = PhysicalTeacher(scene(), calibration(), config=configuration())
    bad = teacher.step(state(x=40., y=0.))
    assert not bad.accepted and bad.reasons == ('measured_body_obstacle_clearance:parked',)
    teacher = PhysicalTeacher(scene(False), calibration(), config=configuration())
    assert teacher.step(state()).accepted
    assert not teacher.step(state()).accepted
    teacher = PhysicalTeacher(scene(False), calibration(), config=configuration())
    assert not teacher.step(state(y=-3.)).accepted


def test_limits_and_calibration_are_not_silently_relaxed():
    with pytest.raises(ValueError, match='bounded'):
        configuration(cruise_speed_mps=4.)
    with pytest.raises(ValueError, match='bounded'):
        configuration(max_steering_acceleration_rps2=1.6)
    with pytest.raises(ValueError, match='qualified'):
        PhysicalTeacher(scene(False), SteeringCalibration.fixture(), config=configuration())
    with pytest.raises(ValueError, match='cruise'):
        PhysicalTeacher(scene(False), calibration(), config=configuration(cruise_speed_mps=3.5))
    limited = replace(calibration(), steering_fraction_min=-.002, steering_fraction_max=.002)
    command = PhysicalTeacher(scene(), limited, config=configuration()).step(state())
    assert not command.accepted
    assert command.reasons == ('planned_curvature_exceeds_calibrated_steering_range',)
    teacher = PhysicalTeacher(scene(False), calibration(), config=configuration())
    assert teacher.step(state(steering=float('nan'))).reasons == ('invalid_vehicle_telemetry',)


def integrate(teacher, calibration, *, initial_x=0., initial_y=0., speed=0., duration=55.):
    """Independent delayed bicycle/longitudinal fixture, not a CARLA QA claim."""
    dt = .05
    x, y, yaw, effective = initial_x, initial_y, 0., 0.
    delay_count = int(round(calibration.delay_s/dt))
    queue = [0.]*(delay_count+1)
    rows = []
    for i in range(int(duration/dt)):
        rear_speed = speed*math.sqrt(max(0., 1-(calibration.reference_offset_m*math.tan(effective)/calibration.wheelbase_m)**2))
        yaw_rate = rear_speed*math.tan(effective)/calibration.wheelbase_m
        command = teacher.step(state(x, y, yaw, speed, i*dt, yaw_rate, effective))
        rows.append((x, y, yaw, speed, effective, command))
        if not command.accepted or command.phase == 'complete':
            break
        queue.append(calibration.angle_from_steer(command.steer))
        delayed = queue.pop(0)
        effective += (1-math.exp(-dt/calibration.time_constant_s))*(delayed-effective)
        acceleration = 3.*command.throttle-6.*command.brake-(.12+.04*speed if speed > .001 else 0.)
        next_speed = max(0., speed+acceleration*dt)
        mid_speed = .5*(speed+next_speed)
        beta = math.atan(calibration.reference_offset_m*math.tan(effective)/calibration.wheelbase_m)
        rate = mid_speed*math.cos(beta)*math.tan(effective)/calibration.wheelbase_m
        x += mid_speed*math.cos(yaw+beta+.5*rate*dt)*dt
        y += mid_speed*math.sin(yaw+beta+.5*rate*dt)*dt
        yaw += rate*dt
        speed = next_speed
    return rows


def test_independent_physics_fixture_avoids_parked_car_returns_and_stops():
    cal = calibration()
    teacher = PhysicalTeacher(scene(), cal, config=configuration())
    rows = integrate(teacher, cal)
    last = rows[-1][-1]
    assert last.accepted, (rows[-1][:5], last.reasons)
    assert last.phase == 'complete', (rows[-1][:5], last.diagnostics)
    xy = np.array([r[:2] for r in rows])
    assert xy[:, 1].min() < -.25
    after = xy[xy[:, 0] > 60.]
    assert len(after) > 100 and np.abs(after[:, 1]).max() < .035
    assert rows[-1][3] <= configuration().stopped_speed_mps
    assert max(r[3] for r in rows) < 2.65
    assert all(not r[-1].supervision_allowed for r in rows)
    angles = np.array([r[-1].steering_angle_rad for r in rows])
    assert np.abs(np.diff(angles)/.05).max() <= configuration().max_steering_rate_rps+1e-8


def test_actor_reference_offset_is_handled_by_feedforward_and_feedback():
    cal = replace(calibration(), reference_offset_m=1.6)
    # Slow, acceleration-limited steering consumes a small amount of planned
    # clearance. Use the same explicit reserve as native collection; the
    # measured body check still requires the unchanged .6 m actual clearance.
    teacher = PhysicalTeacher(scene(obstacle_tracking_reserve_m=.1), cal, config=configuration())
    rows = integrate(teacher, cal, duration=35.)
    assert rows[-1][-1].accepted, (rows[-1][:5], rows[-1][-1].reasons)
    assert rows[-1][0] > 60.
    assert abs(rows[-1][1]) < .04


def test_curved_road_feedforward_follows_right_half_in_actual_fixture_motion():
    radius, width = 30., 9.
    angle = np.linspace(-.4, 1.7, 400)
    center = np.c_[radius*np.sin(angle), radius*(1-np.cos(angle)), angle*0]
    right = np.c_[-np.sin(angle), np.cos(angle), angle*0]
    origin = np.array([0., width/4, 0.])
    road = LaneReference(center-width/2*right-origin, center+width/2*right-origin, radius*angle)
    target = planned_scene_geometry(road)
    assert target.feasible, target.reasons
    cal = replace(calibration(), reference_offset_m=1.6)
    teacher = PhysicalTeacher(target, cal, config=configuration())
    rows = integrate(teacher, cal, duration=20.)
    assert rows[-1][-1].accepted, (rows[-1][:5], rows[-1][-1].reasons)
    assert rows[-1][2] > 1.
    assert max(abs(r[-1].cross_track_error_m) for r in rows) < .05
    speed, delta = np.array([r[3] for r in rows]), np.array([r[4] for r in rows])
    assert np.max(np.abs(speed**2*np.tan(delta)/cal.wheelbase_m)) < 3.5


def test_stationary_endpoint_holds_brake_and_control_step_is_bounded():
    teacher = PhysicalTeacher(scene(False), calibration(), config=configuration())
    endpoint = teacher._xy[-1]
    command = teacher.step(state(x=endpoint[0]-.5, speed=0.))
    assert command.accepted and command.phase == 'complete'
    assert command.throttle == 0 and command.brake == 1
    teacher = PhysicalTeacher(scene(False), calibration(), config=configuration())
    elapsed = []
    for i in range(25):
        before = time.perf_counter()
        command = teacher.step(state(x=i*.125, timestamp=i*.05))
        elapsed.append(time.perf_counter()-before)
        assert command.accepted
    # Generous fixture budget avoids wallclock noise while detecting accidental
    # full-plan optimization every tick. This is not a CARLA throughput claim.
    assert np.median(elapsed) < .1
