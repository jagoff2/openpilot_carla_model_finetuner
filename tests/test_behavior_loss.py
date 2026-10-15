from dataclasses import replace

import numpy as np
import pytest
import torch

from lane_dataset.geometry import EvidenceFrame, ObservableReference
from lane_behavior.target import ObservedRoad, ObservedObstacle, build_behavior_target
from lane_behavior.loss import BehaviorCorridor, behavior_loss, _sat_clearance


def scene(obstacle=True, motion="static"):
    # Ego is already centered in the right half. +Y is right in ego FRD.
    x = np.arange(-10., 81.)
    refs = (EvidenceFrame(100, 5., "wide"),)
    reference = ObservableReference(np.c_[x, x*0-6.75, x*0],
        np.c_[x, x*0+2.25, x*0], x, [refs]*len(x), 5.)
    obstacles = () if not obstacle else (ObservedObstacle("parked",
        [[37.75, 1.35], [42.25, 1.35], [42.25, 3.15], [37.75, 3.15]],
        "parked_vehicle", refs, motion),)
    target = build_behavior_target(ObservedRoad(reference), obstacles)
    return BehaviorCorridor(target, 0., obstacles)


def plans(corridor):
    x = np.linspace(0., 70., 141)
    good = torch.tensor(corridor.sample(x)[0][None], dtype=torch.float32)
    straight = torch.tensor(np.c_[x, x*0][None], dtype=torch.float32)
    return good, straight


def test_correct_avoidance_scores_better_than_driving_through_visible_car():
    corridor = scene()
    assert corridor.target.moving_supervision, corridor.target.reasons
    assert corridor.target.excursions
    good, straight = plans(corridor)
    good_loss, good_report = behavior_loss(good, straight, [corridor])
    bad = straight.clone().requires_grad_(True)
    bad_loss, bad_report = behavior_loss(bad, straight, [corridor])
    assert good_report["valid_labels"] == bad_report["valid_labels"] == 1
    assert good_report["obstacle_loss"] < 1e-5
    assert bad_report["obstacle_loss"] > 0
    assert bad_loss > good_loss+.01
    bad_loss.backward()
    # Positive dL/dY makes gradient descent move left around the right-side car.
    assert bad.grad[0, 80, 1] > 0
    assert torch.isfinite(bad.grad).all()


def test_road_middle_and_opposing_side_are_penalized_on_clear_road():
    corridor = scene(False)
    good, reference = plans(corridor)
    baseline, _ = behavior_loss(good, reference, [corridor])
    for shift in (-2.25, -3.0, .5):
        wrong = good.clone()
        wrong[0, 1:, 1] += shift
        value, report = behavior_loss(wrong, reference, [corridor])
        assert value > baseline+.1
        assert report["valid_labels"] == 1


def test_avoiding_then_staying_left_is_penalized_after_pass():
    corridor = scene()
    good, reference = plans(corridor)
    wrong = good.clone()
    wrong[0, 110:, 1] = -.4
    correct, _ = behavior_loss(good, reference, [corridor])
    off, _ = behavior_loss(wrong, reference, [corridor])
    assert off > correct+.01


def test_shortening_student_path_cannot_remove_obstacle_or_lateral_loss():
    corridor = scene()
    good, reference = plans(corridor)
    collapsed = torch.zeros_like(good).requires_grad_(True)
    loss, report = behavior_loss(collapsed, reference, [corridor])
    assert report["supervised_distance_m"] > 60
    assert report["progress_loss"] > 1
    loss.backward()
    assert torch.isfinite(collapsed.grad).all()


def test_moving_pedestrian_or_unknown_motion_is_masked_not_a_go_label():
    corridor = scene(motion="moving")
    prediction = torch.zeros(1, 5, 2, requires_grad=True)
    loss, report = behavior_loss(prediction, prediction.detach(), [corridor])
    assert report["valid_labels"] == 0
    assert report["masked_labels"] == 1
    loss.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_convex_clearance_detects_object_wholly_inside_car():
    body = torch.tensor([[[-2., -1.], [2., -1.], [2., 1.], [-2., 1.]]], requires_grad=True)
    small = torch.tensor([[-.1, -.1], [.1, -.1], [.1, .1], [-.1, .1]])
    value = _sat_clearance(body, small)
    assert value.item() < 0
    (-value).sum().backward()
    assert torch.isfinite(body.grad).all()


def test_missing_obstacle_in_loss_cannot_hide_target_constraints():
    c = scene()
    with pytest.raises(ValueError, match="match target evidence"):
        BehaviorCorridor(c.target, 0., ())


def test_same_obstacle_id_with_different_geometry_cannot_replace_constraint():
    corridor = scene()
    obstacle = corridor.obstacles[0]
    moved = replace(obstacle, polygon_xy=np.asarray(obstacle.polygon_xy)+[0., 10.])
    with pytest.raises(ValueError, match="scene contract"):
        BehaviorCorridor(corridor.target, 0., (moved,))


@pytest.mark.parametrize("field", ["ego", "config", "road"])
def test_loss_contract_binds_ego_clearances_and_actual_road(field):
    corridor = scene()
    target = corridor.target
    if field == "ego":
        changed = replace(target, ego=replace(target.ego, width_m=1.))
    elif field == "config":
        changed = replace(target, config=replace(target.config, parked_vehicle_clearance_m=.1))
    else:
        reference = target.road.reference
        reference._right.c[3, :, 1] += .5
        changed = target
    with pytest.raises(ValueError, match="scene contract"):
        BehaviorCorridor(changed, 0., corridor.obstacles)


def test_later_obstacle_mutation_is_detected_before_gradient_work():
    corridor = scene()
    good, reference = plans(corridor)
    corridor.obstacles[0].polygon_xy[0][1] += .2
    with pytest.raises(ValueError, match="scene contract"):
        behavior_loss(good, reference, [corridor])


def test_partial_body_support_never_propagates_masked_nan_into_loss():
    corridor = scene(False)
    # Last label reaches the end of target center support; some longitudinal
    # body samples then lie outside it and receive false fixed masks.
    x = np.linspace(0., corridor.s[-1], 80)
    path = torch.tensor(np.c_[x, x*0][None], dtype=torch.float32, requires_grad=True)
    loss, report = behavior_loss(path, path.detach(), [corridor])
    assert report['valid_labels'] == 1
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(path.grad).all()


def test_static_scene_with_stale_obstacle_or_road_evidence_is_masked():
    corridor = scene()
    reference = corridor.target.road.reference
    reference.decision_time_s = 11.
    target = build_behavior_target(ObservedRoad(reference), corridor.obstacles)
    assert not target.moving_supervision and target.reasons == ('road_evidence_stale',)
    # Fresh road, old obstacle observations.
    source = scene()
    road = source.target.road.reference
    road.decision_time_s = 11.
    road.evidence_frames = tuple((type(e)(e.frame_id+1, 11., e.camera),) for e, in road.evidence_frames)
    target = build_behavior_target(ObservedRoad(road), source.obstacles)
    assert not target.moving_supervision
    assert target.reasons == ('obstacle_evidence_stale:parked',)


def test_invalid_road_camera_cannot_support_target_even_after_reference_construction():
    source = scene(False)
    reference = source.target.road.reference
    reference.evidence_frames = tuple((EvidenceFrame(100, 5., 'debug_map'),) for _ in reference.s)
    target = build_behavior_target(ObservedRoad(reference))
    assert not target.moving_supervision
    assert target.reasons == ('road_evidence_missing_or_future',)


def test_yield_rows_still_validate_plan_shapes_and_yaw_contract():
    corridor = scene(motion='moving')
    bad = torch.zeros(1, 5, 3)
    with pytest.raises(ValueError, match='shapes'):
        behavior_loss(bad, bad, [corridor])
    path = torch.zeros(1, 5, 2)
    with pytest.raises(ValueError, match='yaw'):
        behavior_loss(path, path, [corridor], predicted_yaw=torch.zeros(1, 4))


def curved_scene(radius=14., width=9., turn=1):
    angle = np.linspace(-.7, 2., 220)
    center = np.c_[radius*np.sin(angle), turn*radius*(1-np.cos(angle)), angle*0]
    rightward = np.c_[-turn*np.sin(angle), np.cos(angle), angle*0]
    origin = np.array([0., width/4, 0.])
    evidence = (EvidenceFrame(100, 5., "wide"),)
    reference = ObservableReference(center-width/2*rightward-origin,
        center+width/2*rightward-origin, radius*angle, [evidence]*len(angle), 5.)
    target = build_behavior_target(ObservedRoad(reference))
    assert target.moving_supervision, target.reasons
    return BehaviorCorridor(target, source_ego_s=0.), radius-turn*width/4


@pytest.mark.parametrize('turn', [-1, 1])
def test_quarter_turn_queries_use_actual_right_side_metric_arc(turn):
    corridor, actual_radius = curved_scene(turn=turn)
    distance = np.linspace(0., actual_radius*np.pi/2, 200)
    xy, tangent, _, _, valid = corridor.sample(corridor.ego_s+distance)
    assert valid.all()
    theta = distance/actual_radius
    expected = np.c_[actual_radius*np.sin(theta), turn*actual_radius*(1-np.cos(theta))]
    np.testing.assert_allclose(xy, expected, atol=2e-6)
    np.testing.assert_allclose(tangent, np.c_[np.cos(theta), turn*np.sin(theta)], atol=2e-6)
    # The old whole-road station interpretation travels ~16% less distance.
    old_xy = corridor.target.sample(distance)[0]
    assert np.linalg.norm(old_xy[-1]-xy[-1]) > 2.


def test_curve_progress_loss_prefers_correct_metric_path_and_has_useful_gradient():
    corridor, radius = curved_scene()
    distance = np.linspace(0., radius*np.pi/2, 101)
    theta = distance/radius
    good = torch.tensor(np.c_[radius*np.sin(theta), radius*(1-np.cos(theta))][None], dtype=torch.float32)
    reference = torch.tensor(np.c_[distance, distance*0][None], dtype=torch.float32)
    good_loss, good_report = behavior_loss(good, reference, [corridor])
    old = torch.tensor(corridor.target.sample(distance)[0][None], dtype=torch.float32, requires_grad=True)
    wrong_loss, wrong_report = behavior_loss(old, reference, [corridor])
    assert good_report['valid_labels'] == wrong_report['valid_labels'] == 1
    assert good_report['progress_loss'] < .001
    assert wrong_report['progress_loss'] > .5
    assert wrong_loss > good_loss+.5
    wrong_loss.backward()
    assert torch.isfinite(old.grad).all() and old.grad.abs().sum() > .1
    # The final student position must advance around the bend toward the fixed
    # target. Its negative gradient must point into that missing displacement.
    desired_motion = good[0, -1]-old.detach()[0, -1]
    assert (-old.grad[0, -1]*desired_motion).sum() > 0


def test_metric_lookup_includes_avoidance_length_and_cannot_extrapolate_or_mutate():
    corridor = scene()
    source = np.linspace(corridor.target.s[0], corridor.target.s[-1], 20001)
    xy = corridor.target.sample(source)[0]
    measured_length = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
    metric_length = corridor.s[-1]-corridor.s[0]
    np.testing.assert_allclose(metric_length, measured_length, atol=1e-7)
    assert metric_length > source[-1]-source[0]+.001
    assert corridor.source_ego_s == 0.
    assert corridor.ego_s == pytest.approx(0., abs=1e-12)
    q = np.array([[corridor.s[0]-1., corridor.ego_s], [corridor.s[-1], corridor.s[-1]+1.]])
    values = corridor.sample(q)
    np.testing.assert_array_equal(values[-1], [[False, True], [True, False]])
    assert np.isfinite(values[0]).all()
    with pytest.raises(ValueError, match='read-only'):
        corridor.s[0] += 1.
    with pytest.raises(ValueError, match='read-only'):
        corridor.source_stations[0] += 1.
