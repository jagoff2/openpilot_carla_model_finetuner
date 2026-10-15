from dataclasses import replace

import numpy as np
import pytest

from lane_dataset.controller import BicycleMPC, MPCConfig, MPCCommand, SteeringCalibration, VehicleState
from lane_dataset.geometry import EvidenceFrame, ObservableReference
from lane_behavior.target import ObservedRoad, ObservedObstacle, EgoFootprint, build_behavior_target
from lane_behavior.control import (BehaviorExpert, build_control_reference, validate_predicted_body,
    _point_boundary_clearance)


def scene(obstacle=True, motion="static", width=9.):
    x = np.arange(-10., 81.)
    evidence = (EvidenceFrame(100, 5., "wide"),)
    road = ObservedRoad(ObservableReference(np.c_[x, x*0-.75*width, x*0],
        np.c_[x, x*0+.25*width, x*0], x, [evidence]*len(x), 5.))
    obstacles = () if not obstacle else (ObservedObstacle("parked",
        [[37.75, 1.35], [42.25, 1.35], [42.25, 3.15], [37.75, 3.15]],
        "parked_vehicle", evidence, motion),)
    return build_behavior_target(road, obstacles), obstacles


def mpc_config(**kwargs):
    return replace(MPCConfig(require_calibration=False, vehicle_length_m=4.5,
        vehicle_width_m=1.8, horizon_steps=8), **kwargs)


def test_reference_is_observable_centered_on_behavior_and_keeps_causal_evidence():
    target, obstacles = scene()
    prepared = build_control_reference(target, obstacles, mpc_config())
    assert prepared.ready, prepared.reasons
    reference = prepared.reference
    assert isinstance(reference, ObservableReference)
    expected = target.sample(prepared.source_stations)[0]
    np.testing.assert_allclose(reference.center_xyz[:, :2], expected, atol=1e-12)
    assert reference.center_xyz[:, 1].min() < -.1
    assert reference.center_xyz[-1, 1] == pytest.approx(0.)
    assert reference.decision_time_s == 5.
    assert set(reference.evidence_for_arc(reference.s[0], reference.s[-1])) == set(target.evidence_frames)
    assert prepared.diagnostics['scalar_desired_offset_m'] == 0.
    # Physical source edges retain the observed full-road extent, while the
    # derived MPC envelope narrows near the obstacle without rewriting them.
    np.testing.assert_allclose(target.road.reference.right_xyz[:, 1], 2.25)
    assert reference.sample(reference.s)[3].min() < reference.sample(reference.s)[3].max()


def test_clear_road_runs_real_mpc_with_actual_pose_and_zero_scalar_offset():
    target, obstacles = scene(False)
    mpc = BicycleMPC(SteeringCalibration.fixture(), mpc_config())
    decision = BehaviorExpert(mpc).control(VehicleState(0., 0., 0., 3.5, 0., timestamp_s=5.),
        target, obstacles, 3.5)
    assert decision.accepted, decision.reasons
    assert decision.command.accepted
    assert abs(decision.command.carla_steer) < 1e-5
    np.testing.assert_allclose(decision.command.predicted_states[0, :4], [0., 0., 0., 3.5])


def test_unknown_motion_yields_without_running_mpc(monkeypatch):
    target, obstacles = scene(motion="unknown")
    mpc = BicycleMPC(SteeringCalibration.fixture(), mpc_config())
    monkeypatch.setattr(mpc, 'control', lambda *a, **k: pytest.fail("yield target must not enter MPC"))
    result = BehaviorExpert(mpc).control(VehicleState(0., 0., 0., 3.5, 0., timestamp_s=5.),
        target, obstacles, 3.5)
    assert not result.accepted and result.command is None
    assert result.reasons[0] == 'behavior_target_requires_yield'


def test_expert_never_uses_scalar_offset_recovery_bypass(monkeypatch):
    target, obstacles = scene()
    mpc = BicycleMPC(SteeringCalibration.fixture(), mpc_config())
    calls = []
    actual = mpc.control
    def wrapped(*args, **kwargs):
        calls.append(kwargs)
        return actual(*args, **kwargs)
    monkeypatch.setattr(mpc, 'control', wrapped)
    result = BehaviorExpert(mpc).control(VehicleState(0., 0., 0., 3.5, 0., timestamp_s=5.),
        target, obstacles, 3.5)
    assert result.accepted, result.reasons
    assert calls == [{'desired_offset_m': 0.}]


def test_predicted_path_cannot_cut_through_parked_car_or_opposing_half():
    target, obstacles = scene()
    stations = np.linspace(30., 50., 81)
    through = np.c_[stations, stations*0, stations*0, stations*0+3.5]
    okay, reasons = validate_predicted_body(through, target, obstacles, mpc_config())
    assert not okay and reasons == ('mpc_prediction_obstacle_clearance:parked',)
    # Endpoint bodies are clear, but the chord between them crosses the actor.
    assert not validate_predicted_body(through[[0, -1]], target, obstacles, mpc_config())[0]
    across = through.copy()
    across[:, 1] = -3.
    okay, reasons = validate_predicted_body(across, target, obstacles, mpc_config())
    assert not okay and reasons == ('mpc_prediction_leaves_physical_right_half',)
    xy, tangent, *_ = target.sample(stations)
    correct = np.c_[xy, np.arctan2(tangent[:, 1], tangent[:, 0]), stations*0+3.5]
    assert validate_predicted_body(correct, target, obstacles, mpc_config())[0]


@pytest.mark.parametrize('phase', ['departure', 'return'])
def test_real_mpc_tracks_departure_and_return_with_obstacles_preserved(phase):
    target, obstacles = scene()
    event = target.excursions[0]
    station = (event.start_s if phase == 'departure' else event.plateau_end_s)+2.
    q = np.array([station-.01, station, station+.01])
    xy, tangent, *_ = target.sample(q)
    yaw = np.arctan2(tangent[:, 1], tangent[:, 0])
    curvature = (yaw[2]-yaw[0])/np.linalg.norm(xy[2]-xy[0])
    state = VehicleState(*xy[1], yaw[1], 3.5, 3.5*curvature, timestamp_s=5.)
    expert = BehaviorExpert(BicycleMPC(SteeringCalibration.fixture(), mpc_config()))
    result = expert.control(state, target, obstacles, 3.5)
    assert result.accepted, result.reasons
    assert result.command.carla_steer < 0 if phase == 'departure' else result.command.carla_steer > 0
    assert validate_predicted_body(result.command.predicted_states, target, obstacles, mpc_config())[0]


def test_postcheck_rejection_rolls_back_pending_mpc_command_history(monkeypatch):
    target, obstacles = scene()
    mpc = BicycleMPC(SteeringCalibration.fixture(), mpc_config())
    before = list(mpc._history)
    states = np.c_[np.linspace(35., 45., 21), np.zeros(21), np.zeros(21), np.full(21, 3.5)]
    def unsafe(*args, **kwargs):
        mpc._history.append((5., .1))
        mpc._last_timestamp = 5.
        return MPCCommand(True, (), 0., 0., 0., 0., 3.5, .15, states,
            target.evidence_frames, True, 'test trajectory', 0., {})
    monkeypatch.setattr(mpc, 'control', unsafe)
    result = BehaviorExpert(mpc).control(VehicleState(35., 0., 0., 3.5, 0., timestamp_s=5.),
        target, obstacles, 3.5)
    assert not result.accepted and not result.command.accepted
    assert mpc._history == before and mpc._last_timestamp is None


def test_mpc_vehicle_envelope_must_enclose_actual_offset_bounding_box():
    target, obstacles = scene(False)
    changed = build_behavior_target(target.road, obstacles, EgoFootprint(center_xy=(.8, 0.)))
    assert changed.moving_supervision
    with pytest.raises(ValueError, match='does not enclose'):
        build_control_reference(changed, obstacles, mpc_config())


def test_insufficient_symmetric_space_yields_instead_of_relaxing_body_constraints():
    target, obstacles = scene()
    result = build_control_reference(target, obstacles, mpc_config(vehicle_width_m=2.2))
    assert not result.ready
    assert result.reasons == ('symmetric_control_envelope_too_narrow',)


def test_scene_mutation_and_stale_decision_time_cannot_reach_controller():
    target, obstacles = scene(False)
    target.road.reference._right.c[3, :, 1] += 1.
    with pytest.raises(ValueError, match='scene contract'):
        build_control_reference(target, obstacles, mpc_config())
    target, obstacles = scene(False)
    expert = BehaviorExpert(BicycleMPC(SteeringCalibration.fixture(), mpc_config()))
    with pytest.raises(ValueError, match='timestamp'):
        expert.control(VehicleState(0., 0., 0., 3.5, 0., timestamp_s=5.1), target, obstacles, 3.5)


def test_exact_nearest_segment_acceleration_matches_brute_force():
    from lane_behavior.target import _point_segment_distance
    theta = np.linspace(0., 2*np.pi, 120, endpoint=False)
    polygon = np.c_[20*np.cos(theta), 4*np.sin(theta)]
    points = np.random.default_rng(4).uniform([-25., -8.], [25., 8.], (75, 2))
    expected = _point_segment_distance(points[:, None], polygon[None],
        np.roll(polygon, -1, axis=0)[None]).min(axis=1)
    np.testing.assert_allclose(_point_boundary_clearance(points, polygon), expected, atol=1e-12)
