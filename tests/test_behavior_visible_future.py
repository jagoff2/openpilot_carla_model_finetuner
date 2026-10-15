"""Independent future-only road tests: observations begin 18 m ahead of ego."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from lane_dataset.geometry import EvidenceFrame, ObservableReference
from lane_behavior.future_loss import fixed_future_samples
from lane_behavior.loss import BehaviorCorridor, behavior_loss
from lane_behavior.records import build_record, training_corridor
from lane_behavior.target import ObservedObstacle, ObservedRoad
from lane_behavior.window import build_behavior_window


EVIDENCE = (EvidenceFrame(100, 5., "wide"),)
BINDINGS = {"road-proof": "a" * 64}


def straight_road(first=18., last=80.):
    x = np.linspace(first, last, max(4, int(last-first)+1))
    return ObservedRoad(ObservableReference(np.c_[x, x*0-6.75, x*0],
        np.c_[x, x*0+2.25, x*0], x, [EVIDENCE]*len(x), 5.))


def future_scene(obstacle=False, **kwargs):
    road = straight_road(**kwargs)
    obstacles = () if not obstacle else (ObservedObstacle("parked",
        [[37.75, 1.35], [42.25, 1.35], [42.25, 3.15], [37.75, 3.15]],
        "parked_vehicle", EVIDENCE, "static"),)
    target = build_behavior_window(road, obstacles)
    assert target.moving_supervision, target.reasons
    return BehaviorCorridor(target, None, obstacles)


def reference_plan(end=70., count=141):
    x = np.linspace(0., end, count)
    return torch.tensor(np.c_[x, x*0][None], dtype=torch.float32)


def good_plan(corridor, reference):
    # The unseen prefix is merely an arbitrary model output, never road evidence.
    result = reference.clone()
    query = reference[0, :, 0].numpy()
    xy, _, _, _, valid = corridor.sample(query)
    result[0, valid] = torch.tensor(xy[valid], dtype=result.dtype)
    return result


def test_future_geometry_preserves_missing_origin_and_right_half_target():
    corridor = future_scene()
    target = corridor.target
    assert target.source_ego_s is corridor.ego_s is None
    assert target.road.reference.s[0] == 18.
    assert target.s[0] > 18.
    assert target.diagnostics["origin_geometry_supported"] is False
    query = np.array([0., 10., 18., 30., 50.])
    xy, _, divider, right, valid = target.sample(query)
    np.testing.assert_array_equal(valid, [False, False, False, True, True])
    assert np.isnan(xy[~valid]).all()
    np.testing.assert_allclose(xy[valid, 1], 0., atol=1e-12)
    np.testing.assert_allclose(divider[valid, 1], -2.25)
    np.testing.assert_allclose(right[valid, 1], 2.25)
    assert not target.complete_event_qualified
    with pytest.raises(ValueError, match="no observed ego"):
        BehaviorCorridor(target, 0.)


@pytest.mark.parametrize("shift", [-2.25, -3., .6])
def test_future_clear_road_rejects_middle_opposing_half_and_edge_hugging(shift):
    corridor, reference = future_scene(), reference_plan()
    correct, good_report = behavior_loss(reference, reference, [corridor])
    wrong = reference.clone()
    wrong[0, 1:, 1] += shift
    wrong.requires_grad_(True)
    value, report = behavior_loss(wrong, reference, [corridor])
    assert value > correct+.1
    assert report["valid_labels"] == 1
    assert report["path_samples"] == good_report["path_samples"]
    value.backward()
    assert torch.sign(wrong.grad[0, 80, 1]).item() == np.sign(shift)


def test_future_avoidance_and_recovery_have_correct_lateral_gradients():
    corridor, reference = future_scene(True), reference_plan()
    good = good_plan(corridor, reference)
    assert corridor.target.excursions
    good_value, good_report = behavior_loss(good, reference, [corridor])
    straight = reference.clone().requires_grad_(True)
    bad_value, bad_report = behavior_loss(straight, reference, [corridor])
    assert good_report["obstacle_loss"] < 1e-5
    assert bad_report["obstacle_loss"] > 0
    assert bad_value > good_value+.01
    bad_value.backward()
    assert straight.grad[0, 80, 1] > 0  # Descent moves left around the car.
    recovered = good.clone()
    recovered[0, 120:, 1] = -.6
    recovered.requires_grad_(True)
    recovery_value, _ = behavior_loss(recovered, reference, [corridor])
    assert recovery_value > good_value+.01
    recovery_value.backward()
    assert recovered.grad[0, 130, 1] < 0  # Descent returns right after passing.
    assert torch.isfinite(recovered.grad).all()


@pytest.mark.parametrize("student_change", ["collapse", "shorten", "offroad", "translate"])
def test_student_cannot_change_support_or_evade_by_collapsing(student_change):
    corridor, reference = future_scene(True), reference_plan()
    good = good_plan(corridor, reference)
    baseline, expected = behavior_loss(good, reference, [corridor])
    wrong = good.clone()
    if student_change == "collapse":
        wrong.zero_()
    elif student_change == "shorten":
        wrong *= .1
    elif student_change == "offroad":
        wrong[0, 1:, 1] += 20.
    else:
        wrong += torch.tensor([8., .5])
    wrong.requires_grad_(True)
    value, actual = behavior_loss(wrong, reference, [corridor])
    for key in ("valid_labels", "path_samples", "footprint_samples", "supervised_distance_m"):
        assert actual[key] == expected[key]
    assert value > baseline+.1
    if student_change in ("collapse", "shorten"):
        assert actual["progress_loss"] > 10.
    if student_change == "translate":
        assert actual["origin_loss"] > 1.
    value.backward()
    assert torch.isfinite(wrong.grad).all()
    assert torch.count_nonzero(wrong.grad) > 0


def test_unobserved_prefix_has_no_road_target_and_frozen_reference_no_gradient():
    corridor = future_scene()
    reference = reference_plan().requires_grad_(True)
    student = reference.detach().clone()
    student[0, 1:25, 1] = 99.  # Entire excursion ends before supported geometry.
    student[0, 80, 1] = .4
    student.requires_grad_(True)
    value, report = behavior_loss(student, reference, [corridor])
    assert report["hidden_origin_road_supervision"] is False
    value.backward()
    assert reference.grad is None
    assert torch.count_nonzero(student.grad[0, 1:25]) == 0
    assert student.grad[0, 80, 1] > 0


def test_reference_not_reaching_future_support_masks_entire_row():
    corridor = future_scene()
    reference = reference_plan(end=10., count=21)
    student = (reference+10.).requires_grad_(True)
    value, report = behavior_loss(student, reference, [corridor])
    assert value.item() == 0.
    assert report["valid_labels"] == report["path_samples"] == 0
    assert report["masked_labels"] == 1
    value.backward()
    assert torch.count_nonzero(student.grad) == 0


def test_insufficient_observed_body_span_serializes_mask_and_never_fabricates_ego():
    road = straight_road(18., 20.5)
    value = build_record(road, label_frame_id=100, label_timestamp_s=5.,
        evidence_bindings=BINDINGS, ego_position_world=[0., 0., 0.], ego_yaw_rad=0.,
        target_mode="observed_future_geometry")
    assert not value["moving_supervision"]
    assert value["source_ego_s"] is None
    assert value["reasons"] == ["insufficient_observed_future_body_support"]
    assert training_corridor(value, verified_evidence_bindings=BINDINGS) is None


def test_observed_ego_window_retains_original_origin_requirement():
    road = straight_road()
    with pytest.raises(ValueError, match="actual ego station"):
        build_behavior_window(road, source_ego_s=0.)
    old = build_behavior_window(road, source_ego_s=18.5)
    assert not old.moving_supervision
    assert old.reasons == ("actual_ego_rear_body_not_supported_by_observed_road",)
    assert build_behavior_window(road).moving_supervision


def test_future_record_world_transform_restores_ego_relative_obstacle_and_road():
    base = future_scene(True)
    angle, translation = .7, np.array([123., -47., 8.])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    def world(points):
        out = np.asarray(points).copy()
        out[:, :2] = out[:, :2]@rotation.T
        return out+translation
    ref = base.target.road.reference
    road = ObservedRoad(ObservableReference(world(ref.left_xyz), world(ref.right_xyz), ref.s,
        ref.evidence_frames, ref.decision_time_s))
    obstacles = tuple(replace(o, polygon_xy=o.polygon()@rotation.T+translation[:2]) for o in base.obstacles)
    value = build_record(road, obstacles, label_frame_id=100, label_timestamp_s=5.,
        evidence_bindings=BINDINGS, ego_position_world=translation, ego_yaw_rad=angle,
        target_mode="observed_future_geometry")
    assert value["source_ego_s"] is None and value["moving_supervision"]
    assert value["training_ready"] is False
    local = training_corridor(value, verified_evidence_bindings=BINDINGS)
    assert local.ego_s is None
    q = np.linspace(25., 65., 81)
    np.testing.assert_allclose(local.sample(q)[0], base.sample(q)[0], atol=1e-8)
    np.testing.assert_allclose(local.obstacles[0].polygon(), base.obstacles[0].polygon(), atol=1e-10)


@pytest.mark.parametrize("turn", [-1, 1])
def test_future_curves_penalize_cutting_and_supply_correct_normal_gradient(turn):
    radius, width = 45., 9.
    theta = np.linspace(18./radius, 80./radius, 180)
    center = np.c_[radius*np.sin(theta), turn*radius*(1-np.cos(theta)), theta*0]
    normal = np.c_[-turn*np.sin(theta), np.cos(theta), theta*0]
    origin = np.array([0., width/4, 0.])
    road = ObservedRoad(ObservableReference(center-width/2*normal-origin,
        center+width/2*normal-origin, radius*theta, [EVIDENCE]*len(theta), 5.))
    target = build_behavior_window(road)
    assert target.moving_supervision, target.reasons
    corridor = BehaviorCorridor(target, None)
    progress = np.linspace(0., 65., 261)
    lane_radius = radius-turn*width/4
    angles = progress/lane_radius
    xy = np.c_[lane_radius*np.sin(angles), turn*lane_radius*(1-np.cos(angles))]
    reference = torch.tensor(xy[None], dtype=torch.float32)
    correct, report = behavior_loss(reference, reference, [corridor])
    cut = reference.clone()
    inward = np.c_[-np.sin(angles), turn*np.cos(angles)]
    cut[0, 1:] += torch.tensor(.8*inward[1:], dtype=cut.dtype)
    cut.requires_grad_(True)
    value, changed = behavior_loss(cut, reference, [corridor])
    assert value > correct+.1
    assert changed["path_samples"] == report["path_samples"]
    value.backward()
    assert torch.isfinite(cut.grad).all()
    assert float(cut.grad[0, 160]@torch.tensor(inward[160], dtype=cut.dtype)) > 0
    assert report["progress_loss"] < .01


def test_fixed_correspondence_never_queries_artificial_origin_geometry():
    corridor = future_scene()
    reference = reference_plan()[0].numpy()
    indices, fractions, query, weights = fixed_future_samples(reference, corridor, .25)
    supported = corridor.sample(query)[-1]
    assert np.any(supported) and np.any(~supported)
    assert query[supported].min() >= corridor.s[0] > 18.
    assert np.all(indices[supported] >= 40)
    assert np.all(weights > 0)
    assert np.all((fractions >= 0) & (fractions <= 1))


def clipped_actor(lo, hi, *, motion="static"):
    return ObservedObstacle("near-ego", [[lo, -.5], [hi, -.5], [hi, .5], [lo, .5]],
        "parked_vehicle", EVIDENCE, motion)


def test_actor_behind_future_span_stays_bound_without_asserting_ego_passed_it():
    road, actor = straight_road(), clipped_actor(12., 16.)
    target = build_behavior_window(road, [actor])
    assert target.moving_supervision, target.reasons
    assert target.diagnostics["observed_actors_behind_future_span"] == ["near-ego"]
    assert target.diagnostics["actors_automatically_omitted"] == []
    assert target.obstacle_ids == ("near-ego",)
    assert target.source_ego_s is None
    corridor = BehaviorCorridor(target, None, (actor,))
    reference = reference_plan()
    student = reference.clone()
    student[0, 1:, 0] = 14.
    value, report = behavior_loss(student, reference, [corridor])
    assert value > 0. and report["obstacle_loss"] > 0.
    with pytest.raises(ValueError, match="match target evidence"):
        BehaviorCorridor(target, None, ())
    # An observed-ego window must still obtain proper road station support.
    old = build_behavior_window(road, [actor], source_ego_s=25.)
    assert not old.moving_supervision
    assert old.reasons == ("obstacle_station_not_supported_without_endpoint_clipping:near-ego",)


@pytest.mark.parametrize("lo,hi", [(17.5, 19.5), (82., 85.)])
def test_clipped_actor_overlapping_future_body_or_ahead_never_gets_behind_exception(lo, hi):
    target = build_behavior_window(straight_road(), [clipped_actor(lo, hi)])
    assert not target.moving_supervision
    assert target.reasons == ("obstacle_station_not_supported_without_endpoint_clipping:near-ego",)


@pytest.mark.parametrize("motion", ["moving", "unknown"])
def test_behind_future_actor_with_unqualified_motion_still_masks(motion):
    target = build_behavior_window(straight_road(), [clipped_actor(12., 16., motion=motion)])
    assert not target.moving_supervision
    assert target.reasons == ("obstacle_motion_unknown_or_dynamic:near-ego",)


def test_future_segment_interior_detects_obstacle_even_when_prediction_knots_are_clear():
    corridor = future_scene(True)
    x = np.array([0., 25., 55., 70.])
    reference = torch.tensor(np.c_[x, x*0][None], dtype=torch.float32)
    student = reference.clone().requires_grad_(True)
    value, report = behavior_loss(student, reference, [corridor])
    assert report["path_samples"] > len(x)*10
    assert report["obstacle_loss"] > 0.
    assert report["lateral_loss"] > 0.
    value.backward()
    assert student.grad[0, 1, 1] > 0 and student.grad[0, 2, 1] > 0
