from dataclasses import replace

import numpy as np
import pytest

from lane_dataset.geometry import EvidenceFrame, LaneReference, ObservableReference
from lane_behavior.target import (BehaviorConfig, EgoFootprint, ObservedObstacle,
    ObservedRoad, build_behavior_target)


EVIDENCE = EvidenceFrame(12, .6, "wide")


def road(width=12., length=80., spacing=1.):
    x = np.arange(0., length+spacing/2, spacing)
    left = np.c_[x, x*0-width/2, x*0]
    right = np.c_[x, x*0+width/2, x*0]
    return ObservedRoad(ObservableReference(left, right, x, [(EVIDENCE,)]*len(x), .6))


def box(name="parked", x=40., y=5.4, length=4.5, width=1.8,
        kind="parked_vehicle", evidence=(EVIDENCE,), motion="static"):
    polygon = np.array([[x-length/2, y-width/2], [x+length/2, y-width/2],
        [x+length/2, y+width/2], [x-length/2, y+width/2]])
    return ObservedObstacle(name, polygon, kind, evidence, motion)


def test_nominal_target_uses_right_half_center_and_observed_support_only():
    target = build_behavior_target(road())
    assert target.moving_supervision, target.reasons
    np.testing.assert_allclose(target.xy[:, 1], 3.)
    xy, tangent, left, right, valid = target.sample(np.array([-1., 10., 70., 81., np.nan]))
    np.testing.assert_array_equal(valid, [False, True, True, False, False])
    assert np.isnan(xy[~valid]).all()
    np.testing.assert_allclose(xy[valid], [[10., 3.], [70., 3.]])
    np.testing.assert_allclose(tangent[valid], [[1., 0.], [1., 0.]])
    np.testing.assert_allclose(left[valid, 1], 0.)
    np.testing.assert_allclose(right[valid, 1], 6.)
    assert target.evidence_frames == (EVIDENCE,)


def test_parked_car_pass_returns_exactly_to_right_hand_nominal_with_smooth_join():
    target = build_behavior_target(road(), [box(y=4.3)])
    assert target.moving_supervision, target.reasons
    assert len(target.excursions) == 1
    event = target.excursions[0]
    assert event.offset_m < 0  # Move toward road midpoint, without entering opposing half.
    assert target.xy[:, 1].min() > 1.1
    q = np.array([event.start_s-1, event.start_s, event.plateau_start_s,
        event.plateau_end_s, event.end_s, event.end_s+1])
    xy, tangent, *_ = target.sample(q)
    np.testing.assert_allclose(xy[[0, 1, 4, 5], 1], 3., atol=1e-12)
    np.testing.assert_allclose(tangent[[1, 2, 3, 4]], [[1., 0.]]*4, atol=1e-12)
    assert target.diagnostics['minimum_obstacle_clearance_surplus_m'] >= 0


def test_nine_metre_road_accepts_shallow_edge_intrusion_without_crossing_midpoint():
    # Right lane spans 0..4.5 m. A parked body reaches inward to 3.6 m;
    # a left shift leaves space for ego width plus both clearance margins.
    target = build_behavior_target(road(width=9.), [box(y=4.5)])
    assert target.moving_supervision, target.reasons
    assert target.excursions
    assert target.xy[:, 1].min() >= 1.1
    assert target.xy[:, 1].max() <= 2.25+1e-8


def test_parked_car_on_narrow_road_yields_instead_of_using_opposing_side():
    target = build_behavior_target(road(width=7.), [box(y=2.6)])
    assert not target.moving_supervision
    assert "no_right_half_pass_with_complete_recovery" in target.reasons
    assert not target.sample(np.array([10., 40., 60.]))[-1].any()
    assert np.isnan(target.xy).all()


def test_static_pedestrian_has_larger_margin_and_complete_recovery():
    vehicle = build_behavior_target(road(), [box(y=4.3, length=.6, width=.6)])
    pedestrian = build_behavior_target(road(), [box(y=4.3, length=.6, width=.6, kind="pedestrian")])
    assert vehicle.moving_supervision and pedestrian.moving_supervision
    assert vehicle.excursions and pedestrian.excursions
    assert pedestrian.excursions[0].offset_m < vehicle.excursions[0].offset_m < 0
    event = pedestrian.excursions[0]
    np.testing.assert_allclose(pedestrian.sample(np.array([event.end_s+1]))[0][0, 1], 3.)


@pytest.mark.parametrize("change,reason", [
    ({"motion": "unknown"}, "obstacle_motion_unknown_or_dynamic"),
    ({"motion": "moving"}, "obstacle_motion_unknown_or_dynamic"),
    ({"evidence_frames": ()}, "obstacle_evidence_missing_or_future"),
    ({"evidence_frames": (EvidenceFrame(13, .7, "wide"),)}, "obstacle_evidence_missing_or_future"),
])
def test_uncertain_or_future_obstacle_evidence_masks_moving_supervision(change, reason):
    target = build_behavior_target(road(), [replace(box(), **change)])
    assert not target.moving_supervision
    assert any(item.startswith(reason) for item in target.reasons)


def test_lane_markings_are_not_silently_treated_as_full_road_edges():
    target = build_behavior_target(replace(road(), left_role="lane_marking"))
    assert target.reasons == ("full_road_edges_not_established",)
    unobserved = LaneReference(road().reference.left_xyz, road().reference.right_xyz)
    with pytest.raises(TypeError, match="independently observed"):
        build_behavior_target(ObservedRoad(unobserved))


def test_obstacle_at_end_without_observed_return_is_masked():
    target = build_behavior_target(road(), [box(x=72., y=4.3)])
    assert not target.moving_supervision
    assert any(reason.startswith("insufficient_observed_departure_and_return") for reason in target.reasons)


def test_path_is_rigid_transform_equivariant():
    source = road()
    obstacle = box(y=4.3)
    target = build_behavior_target(source, [obstacle])
    theta = .7
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    shift = np.array([300., -80.])
    left, right = source.reference.left_xyz.copy(), source.reference.right_xyz.copy()
    left[:, :2] = left[:, :2]@rotation.T+shift
    right[:, :2] = right[:, :2]@rotation.T+shift
    reference = ObservableReference(left, right, source.reference.s, source.reference.evidence_frames, .6)
    transformed = build_behavior_target(ObservedRoad(reference),
        [replace(obstacle, polygon_xy=obstacle.polygon_xy@rotation.T+shift)])
    assert transformed.moving_supervision, transformed.reasons
    np.testing.assert_allclose(transformed.xy, target.xy@rotation.T+shift, atol=1e-8)


def test_long_body_not_only_center_point_must_clear_obstacle_on_turn():
    angle = np.linspace(0, np.pi/2, 100)
    radius, width = 18., 12.
    center = np.c_[radius*np.sin(angle), radius*(1-np.cos(angle)), angle*0]
    rightward = np.c_[-np.sin(angle), np.cos(angle), angle*0]
    ref = ObservableReference(center-rightward*width/2, center+rightward*width/2,
        radius*angle, [(EVIDENCE,)]*len(angle), .6)
    ordinary = build_behavior_target(ObservedRoad(ref))
    assert ordinary.moving_supervision, ordinary.reasons
    # Extreme body overhang sweeps across the virtual center and/or edge despite
    # a centered path. It must be rejected rather than admitted on point checks.
    long = build_behavior_target(ObservedRoad(ref), ego=EgoFootprint(length_m=25., width_m=1.8))
    assert not long.moving_supervision


def test_unsupported_obstacle_semantics_are_not_ignored():
    target = build_behavior_target(road(), [box(kind="unknown")])
    assert not target.moving_supervision
    assert target.reasons == ("unsupported_obstacle_class:parked",)


def test_dense_body_sweep_catches_small_object_between_original_observation_knots():
    target = build_behavior_target(road(spacing=5.), [box(x=35.23, y=3., length=.05, width=.05)])
    assert target.moving_supervision, target.reasons
    assert target.excursions
    assert target.diagnostics['minimum_obstacle_clearance_surplus_m'] >= 0


def test_observed_edge_gap_is_not_filled_with_supervised_geometry():
    target = build_behavior_target(road(spacing=10.))
    assert not target.moving_supervision
    assert target.reasons == ("observed_road_edge_gap_exceeds_limit",)


def test_ego_footprint_offset_and_yaw_participate_in_body_clearance():
    # The vehicle origin can be centered while an offset bounding box extends
    # into the opposing half. It must not inherit the centered-origin result.
    assert build_behavior_target(road(width=7.)).moving_supervision
    target = build_behavior_target(road(width=7.), ego=EgoFootprint(center_xy=(0., -1.)))
    assert not target.moving_supervision
    yawed = build_behavior_target(road(width=7.), ego=EgoFootprint(yaw_rad=np.pi/2))
    assert not yawed.moving_supervision


def test_no_geometry_is_derived_from_hidden_reference_fields():
    reference = road().reference
    # The module reads only this observed reference. Altering arbitrary attached
    # hidden-map metadata cannot change targets.
    first = build_behavior_target(ObservedRoad(reference))
    reference.hidden_map_center = np.full((100, 3), 1e9)
    second = build_behavior_target(ObservedRoad(reference))
    np.testing.assert_array_equal(first.xy, second.xy)


def test_bad_configuration_and_self_crossed_obstacle_polygon_are_rejected():
    with pytest.raises(ValueError):
        BehaviorConfig(sample_spacing_m=1.)
    obstacle = box()
    with pytest.raises(ValueError, match="boundary order"):
        build_behavior_target(road(), [replace(obstacle, polygon_xy=obstacle.polygon_xy[[0, 2, 1, 3]])])
