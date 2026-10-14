import math
import json

import numpy as np
import pytest

from lane_dataset.geometry import LaneReference
from lane_dataset.traffic import (BodyFootprint, LaneCorridor, TrafficConfig,
    TrafficValidator, convex_hull, polygon_distance, polyline_polygon_distance, swept_footprint)


def road(x=None, y=None):
    x = np.linspace(0, 100, 101) if x is None else np.asarray(x)
    y = np.zeros_like(x) if y is None else np.asarray(y)
    center = np.c_[x, y, np.zeros_like(x)]
    yaw = np.arctan2(np.gradient(y), np.gradient(x))
    right = np.c_[-np.sin(yaw), np.cos(yaw), np.zeros_like(x)]
    return LaneReference(center-right*1.8, center+right*1.8)


def body(identity, x, y, yaw=0., width=2., length=4.):
    rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    vertices = np.array([[a*length/2, b*width/2, c*.75] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    return BodyFootprint(identity, vertices@rotation.T+np.array([x, y, .75]), rotation)


def test_edge_intersection_containment_and_clear_distance():
    polygon = body(1, 5, 5).polygon
    assert polyline_polygon_distance([[0, 5], [10, 5]], polygon) == 0
    assert polyline_polygon_distance([[4, 5], [6, 5]], polygon) == 0
    assert polyline_polygon_distance([[0, 0], [10, 0]], polygon) == pytest.approx(4)
    assert polygon_distance(body(1, 0, 0, width=10, length=10).polygon, body(2, 0, 0).polygon) == 0
    assert polygon_distance(body(1, 0, 0).polygon, body(2, 10, 0).polygon) == pytest.approx(6)


def test_full_body_and_heading_determine_corridor_intrusion():
    corridor = LaneCorridor(road(), ego_width_m=2.)
    assert corridor.clearance(body(1, 20, 3.6).polygon) == pytest.approx(1.45)
    # Center stays outside the 1.15 m corridor; a rotated corner crosses it.
    assert corridor.clearance(body(1, 20, 2.8, yaw=math.pi/4).polygon) < 0


def test_full_hairpin_geometry_detects_other_arm_obstruction():
    theta = np.linspace(-math.pi/2, math.pi/2, 161)
    reference = road(20*np.cos(theta), 20*np.sin(theta)+20)
    corridor = LaneCorridor(reference, 2.)
    assert corridor.clearance(body(1, 0, 40).polygon) < 0
    assert corridor.chord_error_m > 0


def test_exact_spline_chord_bound_contains_dense_reference():
    x = np.linspace(0, 30, 21)
    reference = road(x, 4*np.sin(x/6))
    corridor = LaneCorridor(reference, 2., step_m=.75)
    for point in reference.sample(np.linspace(reference.s[0], reference.s[-1], 1000))[0][::13]:
        line = corridor.center_xy
        delta = line[1:]-line[:-1]
        fraction = np.clip(np.sum((point[:2]-line[:-1])*delta, axis=1)/np.sum(delta*delta, axis=1), 0, 1)
        distance = np.linalg.norm(point[:2]-line[:-1]-fraction[:, None]*delta, axis=1).min()
        assert distance <= corridor.chord_error_m+1e-10


def test_swept_envelope_catches_between_frame_crossing():
    before, after = body(2, 20, -3.6), body(2, 20, 3.6)
    corridor = LaneCorridor(road(), 2.)
    assert corridor.clearance(before.polygon) > 0 and corridor.clearance(after.polygon) > 0
    polygon, padding = swept_footprint(before, after)
    assert corridor.clearance(polygon, padding) < 0
    with pytest.raises(ValueError, match="different actors"):
        swept_footprint(before, body(3, 20, 3.6))


def test_rotating_sweep_includes_continuous_corners():
    before, after = body(2, 20, 3.6, yaw=-.3), body(2, 20.2, 3.6, yaw=.3)
    polygon, padding = swept_footprint(before, after)
    assert padding > 0
    for u in np.linspace(0, 1, 17):
        intermediate = body(2, 20+.2*u, 3.6, yaw=-.3+.6*u)
        for point in intermediate.vertices_xyz[:, :2]:
            point_line = np.array([point, point])
            assert polyline_polygon_distance(point_line, polygon) <= padding+1e-10


def test_rotational_padding_uses_actor_origin_for_offset_bounding_box():
    before, after = body(2, 20, 3.6, yaw=-.3), body(2, 20.2, 3.6, yaw=.3)
    _, centered_padding = swept_footprint(before, after)
    offset_before = BodyFootprint(2, before.vertices_xyz, before.rotation_matrix, before.origin_xyz-[2, 0, 1])
    offset_after = BodyFootprint(2, after.vertices_xyz, after.rotation_matrix, after.origin_xyz-[2, 0, 1])
    _, offset_padding = swept_footprint(offset_before, offset_after)
    assert offset_padding > centered_padding


def test_validator_latches_intrusions_collisions_and_frame_loss():
    validator = TrafficValidator(LaneCorridor(road(), 2.), [2, 3])
    ego = body(1, 2, 0)
    safe = [body(2, 20, 3.6), body(3, 40, 3.6)]
    assert validator.check(100, 5., ego, safe)["passed"]
    invalid = validator.check(102, 5.1, ego, safe, [dict(frame_id=101)])
    assert set(invalid["reasons"]) == {"collision_event", "missing_duplicate_or_non20hz_traffic_frame"}
    assert not validator.check(103, 5.15, ego, safe)["passed"]
    missing = validator.check(104, 5.2, ego, safe[:1])
    assert "missing_or_duplicate_traffic_actor" in missing["reasons"]


def test_body_pair_contacts_rejected_even_outside_ego_corridor():
    validator = TrafficValidator(LaneCorridor(road(), 2.), [2, 3])
    report = validator.check(1, .05, body(1, 2, 0), [body(2, 20, 3.6), body(3, 22, 3.6)])
    assert "traffic_body_contact_or_insufficient_clearance" in report["reasons"]
    assert "traffic_body_intrudes_intended_corridor" not in report["reasons"]


def test_optional_traffic_configuration_is_bounded_and_no_silent_reduction():
    for count in (0, 1, 5):
        with pytest.raises(ValueError):
            TrafficConfig(count=count)
    for speed in (0, -1, math.nan, math.inf, 10.1):
        with pytest.raises(ValueError):
            TrafficConfig(speed_mps=speed)
    with pytest.raises(ValueError):
        BodyFootprint(1, np.zeros((4, 3)), np.eye(3))
    with pytest.raises(ValueError):
        convex_hull([[0, 0], [1, 0]])


def test_invalid_telemetry_rejected_and_missing_actor_report_serializes():
    validator = TrafficValidator(LaneCorridor(road(), 2.), [2, 3])
    with pytest.raises(ValueError, match="frame identity"):
        validator.check(1, math.nan, body(1, 2, 0), [])
    report = validator.check(1, .05, body(1, 2, 0), [])
    assert not report["passed"]
    json.dumps(report, allow_nan=False)
