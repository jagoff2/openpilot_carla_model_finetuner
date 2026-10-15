"""Public observed-record replay through the differentiable behavior objective."""
import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from lane_dataset.geometry import EvidenceFrame, LaneReference, ObservableReference
from lane_behavior.loss import BehaviorCorridor, behavior_loss
from lane_behavior.planned import planned_scene_geometry
from lane_behavior.records import build_record, digest, replay_record, training_corridor
from lane_behavior.target import ObservedObstacle, ObservedRoad
from lane_behavior.window import WindowTarget
from lane_training.centerline import LossConfig, recovery_target


BINDINGS = {"road-proof.json": "a"*64, "actor-proof.json": "b"*64}


def record_scene(first=30., last=80., *, ego_x=35., ego_y=1.87, angle=0., translation=(0., 0.)):
    x = np.arange(first, last+.5)
    left, right = np.c_[x, x*0-6., x*0], np.c_[x, x*0+6., x*0]
    polygon = np.array([[37.75, 3.4], [42.25, 3.4], [42.25, 5.2], [37.75, 5.2]])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    translation = np.asarray(translation)
    left[:, :2] = left[:, :2]@rotation.T+translation
    right[:, :2] = right[:, :2]@rotation.T+translation
    polygon = polygon@rotation.T+translation
    position = np.r_[np.array([ego_x, ego_y])@rotation.T+translation, 0.]
    evidence = (EvidenceFrame(100, 5., "wide"),)
    road = ObservedRoad(ObservableReference(left, right, x, [evidence]*len(x), 5.))
    obstacle = ObservedObstacle("parked", polygon, "parked_vehicle", evidence, "static")
    return build_record(road, [obstacle], label_frame_id=100, label_timestamp_s=5.,
        evidence_bindings=BINDINGS, ego_position_world=position, ego_yaw_rad=angle,
        target_mode="observed_future_window")


def corridor_from(record):
    corridor = training_corridor(record, verified_evidence_bindings=BINDINGS)
    assert corridor is not None, record["reasons"]
    assert isinstance(corridor.target, WindowTarget)
    return corridor


def rehash(record):
    record["record_sha256"] = digest({key:value for key, value in record.items() if key != "record_sha256"})


def test_public_window_record_round_trip_and_world_to_ego_covariance():
    original = record_scene()
    moved = record_scene(angle=.73, translation=(142., -63.))
    before = copy.deepcopy(moved)
    replay_record(moved, verified_evidence_bindings=BINDINGS)
    assert moved == before
    assert original["record_sha256"] != moved["record_sha256"]
    assert original["source_ego_s"] == pytest.approx(35., abs=1e-7)
    assert moved["source_ego_s"] == pytest.approx(original["source_ego_s"], abs=1e-7)
    a, b = corridor_from(original), corridor_from(moved)
    distance = np.linspace(0., 35., 141)
    for first, second in zip(a.sample(a.ego_s+distance), b.sample(b.ego_s+distance)):
        np.testing.assert_allclose(first, second, atol=1e-7)
    assert original["training_ready"] is moved["training_ready"] is False


@pytest.mark.parametrize("change", ["station", "pose", "mode", "excursion"])
def test_self_rehashed_record_cannot_change_bound_projection_or_window_solution(change):
    record = record_scene()
    if change == "station":
        record["source_ego_s"] += 1.
    elif change == "pose":
        record["ego_position_world"][0] += 1.
    elif change == "mode":
        record["target_mode"] = "complete_event"
    else:
        record["excursions"][0]["offset_m"] -= .1
    rehash(record)
    with pytest.raises(ValueError, match="deterministic reconstruction"):
        training_corridor(record, verified_evidence_bindings=BINDINGS)


def test_window_proof_bindings_are_checked_before_rebuilding_training_geometry():
    record = record_scene()
    with pytest.raises(ValueError, match="binding mismatch"):
        training_corridor(record, verified_evidence_bindings={**BINDINGS, "actor-proof.json":"c"*64})


def test_partial_event_joins_do_not_expand_metric_or_observed_query_span():
    record = record_scene(last=46.)
    corridor = corridor_from(record)
    event = corridor.target.excursions[0]
    assert event.start_s < corridor.target.s[0]
    assert event.end_s > corridor.target.s[-1]
    assert corridor.source_stations[0] == corridor.target.s[0]
    assert corridor.source_stations[-1] == corridor.target.s[-1]
    assert np.all(np.diff(corridor.source_stations) > 0)
    assert np.all(np.diff(corridor.s) > 0)
    dense = np.linspace(corridor.target.s[0], corridor.target.s[-1], 4001)
    measured_arc = np.linalg.norm(np.diff(corridor.target.sample(dense)[0], axis=0), axis=1).sum()
    assert corridor.s[-1]-corridor.s[0] == pytest.approx(measured_arc, abs=1e-7)
    query = np.array([corridor.s[0]-.01, corridor.s[0], corridor.s[-1], corridor.s[-1]+.01])
    values = corridor.sample(query)
    np.testing.assert_array_equal(values[-1], [False, True, True, False])
    assert np.isfinite(values[0]).all()
    assert not corridor.target.complete_event_qualified


def test_partial_passing_record_produces_leftward_avoidance_gradient():
    corridor = corridor_from(record_scene(last=46.))
    distance = np.linspace(0., 7., 29)
    reference = torch.tensor(np.c_[distance, distance*0][None], dtype=torch.float32)
    good = torch.tensor(corridor.sample(corridor.ego_s+distance)[0][None], dtype=torch.float32)
    baseline, good_report = behavior_loss(good, reference, [corridor])
    bad = good.clone()
    bad[0, :, 1] += torch.linspace(0., .4, len(distance))
    bad.requires_grad_(True)
    loss, report = behavior_loss(bad, reference, [corridor])
    assert good_report["valid_labels"] == report["valid_labels"] == 1
    assert good_report["obstacle_loss"] < 1e-5
    assert report["obstacle_loss"] > .01
    assert loss > baseline+.05
    loss.backward()
    assert torch.isfinite(bad.grad).all()
    # Gradient descent must move the prediction left away from the parked car.
    assert bad.grad[0, -4:, 1].sum() > 0


def test_recovery_record_gradient_returns_right_after_obstacle_without_past_pixels():
    corridor = corridor_from(record_scene(first=35., ego_x=47.))
    event = corridor.target.excursions[0]
    assert event.start_s < corridor.target.road.reference.s[0]
    assert event.plateau_end_s < corridor.source_ego_s < event.end_s
    config = LossConfig(clearance_m=.2)
    target = recovery_target(corridor, config)
    assert target is not None
    distance = np.linspace(0., 20., 81)
    reference = torch.tensor(np.c_[distance, distance*0][None], dtype=torch.float32)
    good = torch.tensor(target.sample(corridor.ego_s+distance)[0][None], dtype=torch.float32)
    correct, correct_report = behavior_loss(good, reference, [corridor], config=config)
    stuck_left = reference.clone().requires_grad_(True)
    loss, report = behavior_loss(stuck_left, reference, [corridor], config=config)
    assert correct_report["valid_labels"] == report["valid_labels"] == 1
    assert good[0, -1, 1] > 1.
    assert loss > correct+.1
    loss.backward()
    assert torch.isfinite(stuck_left.grad).all()
    # Remaining at the passing offset must generate a rightward update.
    assert stuck_left.grad[0, -20:, 1].sum() < 0


def test_curved_public_projection_and_metric_lookup_use_nominal_right_side_geometry():
    radius, width, station = 30., 12., 36.
    s = np.linspace(20., 80., 301)
    theta = s/radius
    center = np.c_[radius*np.sin(theta), radius*(1-np.cos(theta)), s*0]
    right = np.c_[-np.sin(theta), np.cos(theta), s*0]
    evidence = (EvidenceFrame(100, 5., "wide"),)
    road = ObservedRoad(ObservableReference(center-width/2*right, center+width/2*right,
        s, [evidence]*len(s), 5.))
    angle, actual_radius = station/radius, radius-width/4
    position = [actual_radius*np.sin(angle), radius-actual_radius*np.cos(angle), 0.]
    record = build_record(road, label_frame_id=100, label_timestamp_s=5.,
        evidence_bindings=BINDINGS, ego_position_world=position, ego_yaw_rad=angle,
        target_mode="observed_future_window")
    corridor = corridor_from(record)
    assert record["source_ego_s"] == pytest.approx(station, abs=1e-6)
    distance = np.linspace(0., 20., 101)
    xy, tangent, _, _, valid = corridor.sample(corridor.ego_s+distance)
    assert valid.all()
    turn = distance/actual_radius
    np.testing.assert_allclose(xy, np.c_[actual_radius*np.sin(turn), actual_radius*(1-np.cos(turn))], atol=2e-6)
    np.testing.assert_allclose(tangent, np.c_[np.cos(turn), np.sin(turn)], atol=2e-6)


def test_public_window_loss_keeps_teacher_targets_and_rebound_progress_excluded():
    corridor = corridor_from(record_scene())
    with pytest.raises(ValueError, match="ego station differs"):
        BehaviorCorridor(corridor.target, corridor.source_ego_s+.1, corridor.obstacles)
    reference = corridor.target.road.reference
    planned = planned_scene_geometry(LaneReference(reference.left_xyz, reference.right_xyz, reference.s))
    with pytest.raises(TypeError, match="observed BehaviorTarget or WindowTarget"):
        BehaviorCorridor(planned, corridor.source_ego_s)
    with pytest.raises(TypeError, match="observed"):
        build_record(planned, label_frame_id=100, label_timestamp_s=5.,
            evidence_bindings=BINDINGS, ego_position_world=[35., 1.87, 0.], ego_yaw_rad=0.,
            target_mode="observed_future_window")
