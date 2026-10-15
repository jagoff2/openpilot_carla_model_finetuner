from dataclasses import replace

import numpy as np
import pytest

from lane_behavior.quality import QualityConfig, qualify_episode, swept_clearance
from lane_behavior.target import EgoFootprint, build_behavior_target
from test_behavior_target import box, road


def telemetry(target, *, start=5., end=70., step=.2):
    station = np.arange(start, end+step*.01, step)
    xy, tangent, *_ = target.sample(station)
    assert np.isfinite(xy).all()
    t = np.arange(len(station))*.05
    position = np.c_[xy, station*0]
    velocity = np.gradient(position, t, axis=0, edge_order=2)
    yaw = np.arctan2(tangent[:, 1], tangent[:, 0])
    acceleration = np.gradient(velocity, t, axis=0, edge_order=2)
    lateral = -np.sin(yaw)*acceleration[:, 0]+np.cos(yaw)*acceleration[:, 1]
    return dict(timestamps_s=t, positions_xyz=position, yaw_rad=yaw,
        velocity_world=velocity, lateral_acceleration_mps2=lateral)


def visual(obstacles=()):
    return dict(replayed=True, road=dict(passed=True, sha256="a"*64),
        obstacles={o.identity:dict(passed=True, sha256="b"*64) for o in obstacles})


@pytest.fixture(scope="module")
def avoidance():
    obstacles = (box(y=4.3),)
    target = build_behavior_target(road(), obstacles)
    assert target.moving_supervision
    return target, obstacles


def test_complete_clear_road_demonstration_requires_separate_rendered_evidence():
    target = build_behavior_target(road(length=25.))
    data = telemetry(target, end=20.)
    measured = qualify_episode(**data, target=target, collision_count=0)
    assert measured["physical_passed"]
    assert not measured["passed"] and not measured["training_ready"]
    qualified = qualify_episode(**data, target=target, collision_count=0, visual_evidence=visual())
    assert qualified["passed"] and not qualified["training_ready"]
    assert qualified["metrics"]["maximum_absolute_target_offset_m"] < 1e-10
    assert qualified["metrics"]["sweep"]["minimum_right_half_clearance_m"] > 2.


def test_complete_avoidance_and_five_metre_nominal_recovery_pass(avoidance):
    target, obstacles = avoidance
    report = qualify_episode(**telemetry(target), target=target, obstacles=obstacles,
        collision_count=0, visual_evidence=visual(obstacles))
    assert report["passed"], report["reasons"]
    events = report["metrics"]["avoidance_events"]
    assert len(events) == 1 and events[0]["passed"]
    assert events[0]["retained_nominal_distance_m"] >= 5.
    assert events[0]["maximum_nominal_offset_m"] < 1e-9
    sweep = report["metrics"]["sweep"]
    assert sweep["minimum_obstacle_clearance_m"]["parked"] >= .6
    assert sweep["maximum_rotation_guard_m"] > 0.


def test_following_whole_road_center_fails_right_half_and_tracking():
    target = build_behavior_target(road(length=25.))
    data = telemetry(target, end=20.)
    data["positions_xyz"][:, 1] = 0.
    report = qualify_episode(**data, target=target, collision_count=0)
    assert not report["physical_passed"]
    assert "mean_signed_target_offset" in report["reasons"]
    assert "actual_vehicle_body_leaves_right_half" in report["reasons"]


@pytest.mark.parametrize("mode, reason", [
    ("mean", "mean_signed_target_offset"),
    ("p95", "p95_absolute_target_offset"),
    ("maximum", "maximum_absolute_target_offset"),
])
def test_all_target_tracking_limits_are_enforced(mode, reason):
    target = build_behavior_target(road(length=25.))
    data = telemetry(target, end=20.)
    if mode == "mean":
        data["positions_xyz"][:, 1] += .06
    elif mode == "p95":
        data["positions_xyz"][:, 1] += .15*np.sin(np.linspace(0, 2*np.pi, len(data["yaw_rad"])))
    else:
        data["positions_xyz"][20, 1] += .21
    report = qualify_episode(**data, target=target, collision_count=0)
    assert not report["physical_passed"] and reason in report["reasons"]


def test_nominal_path_through_parked_car_is_rejected_on_actual_swept_clearance(avoidance):
    target, obstacles = avoidance
    data = telemetry(target)
    data["positions_xyz"][:, 1] = 3.
    data["yaw_rad"][:] = 0.
    report = qualify_episode(**data, target=target, obstacles=obstacles, collision_count=0)
    assert "actual_obstacle_clearance_below_limit:parked" in report["reasons"]
    assert report["metrics"]["sweep"]["minimum_obstacle_clearance_m"]["parked"] == 0.


def test_rotating_body_and_bbox_offset_participate_in_clearance():
    target = build_behavior_target(road(width=7., length=25.))
    position = np.array([[10., 1.75, 0.], [10., 1.75, 0.]])
    report = swept_clearance(position, np.array([0., np.pi]), target)
    assert "actual_vehicle_body_leaves_right_half" in report["reasons"]
    assert report["maximum_rotation_guard_m"] > 0
    # Offset body geometry matters even when the actor origin stays nominal.
    offset = replace(target, ego=EgoFootprint(center_xy=(0., -1.)))
    report = swept_clearance(position, np.zeros(2), offset)
    assert "actual_vehicle_body_leaves_right_half" in report["reasons"]


def test_sweep_detects_obstacle_between_pose_endpoints():
    obstacle = box(x=40., y=3., length=.05, width=.05)
    target = build_behavior_target(road(), (obstacle,))
    assert target.moving_supervision
    position = np.array([[35., 3., 0.], [45., 3., 0.]])
    report = swept_clearance(position, np.zeros(2), target, (obstacle,))
    assert report["minimum_obstacle_clearance_m"]["parked"] == 0.
    assert "actual_obstacle_clearance_below_limit:parked" in report["reasons"]


def test_pedestrian_uses_larger_actual_body_margin():
    obstacle = box(x=18., y=4.3, length=.6, width=.6, kind="pedestrian")
    target = build_behavior_target(road(length=50.), (obstacle,))
    assert target.moving_supervision
    position = np.array([[15., 3., 0.], [21., 3., 0.]])
    report = swept_clearance(position, np.zeros(2), target, (obstacle,))
    assert report["required_obstacle_clearance_m"]["parked"] == 1.2
    assert not report["passed"]


def test_missing_approach_or_short_recovery_cannot_qualify(avoidance):
    target, obstacles = avoidance
    event = target.excursions[0]
    short = telemetry(target, end=event.end_s+4.)
    report = qualify_episode(**short, target=target, obstacles=obstacles, collision_count=0)
    assert "event_0:less_than_five_metres_retained_nominal_recovery" in report["reasons"]
    missing = telemetry(target, start=event.start_s+1.)
    report = qualify_episode(**missing, target=target, obstacles=obstacles, collision_count=0)
    assert "event_0:avoidance_event_not_fully_traversed" in report["reasons"]


def test_recovery_requires_persistent_nominal_placement(avoidance):
    target, obstacles = avoidance
    data = telemetry(target)
    selected = data["positions_xyz"][:, 0] > target.excursions[0].end_s
    data["positions_xyz"][selected, 1] += .11
    report = qualify_episode(**data, target=target, obstacles=obstacles, collision_count=0)
    assert "event_0:nominal_recovery_offset_above_limit" in report["reasons"]


@pytest.mark.parametrize("mode,reason", [
    ("acceleration", "lateral_acceleration_above_limit"),
    ("jerk", "lateral_jerk_above_limit"),
    ("collision", "collision"),
    ("missing_collision", "collision_sensor_evidence_missing_or_invalid"),
    ("timing", "missing_duplicate_or_non20hz_frames"),
    ("teleport", "actual_pose_velocity_disagreement"),
])
def test_unchanged_dynamics_and_recording_integrity_limits(mode, reason):
    target = build_behavior_target(road(length=25.))
    data = telemetry(target, end=20.)
    collisions = 0
    if mode == "acceleration":
        data["lateral_acceleration_mps2"][:] = 3.51
    elif mode == "jerk":
        data["lateral_acceleration_mps2"][20:] = .251
    elif mode == "collision":
        collisions = 1
    elif mode == "missing_collision":
        collisions = None
    elif mode == "timing":
        data["timestamps_s"] *= 2.
    else:
        data["positions_xyz"][20, 0] += 2.
    report = qualify_episode(**data, target=target, collision_count=collisions)
    assert not report["physical_passed"] and reason in report["reasons"]


def test_unknown_visual_obstacles_and_unreplayed_receipts_are_not_sufficient():
    target = build_behavior_target(road(length=25.))
    receipt = visual(); receipt["replayed"] = False
    report = qualify_episode(**telemetry(target, end=20.), target=target,
        collision_count=0, visual_evidence=receipt)
    assert report["physical_passed"] and not report["passed"]


def test_target_scene_and_excursion_tampering_is_rejected(avoidance):
    target, obstacles = avoidance
    with pytest.raises(ValueError, match="scene differs"):
        qualify_episode(**telemetry(target), target=target, obstacles=(), collision_count=0)
    altered = replace(target, excursions=())
    with pytest.raises(ValueError, match="deterministic reconstruction"):
        qualify_episode(**telemetry(target), target=altered, obstacles=obstacles, collision_count=0)


def test_yield_target_never_qualifies_and_limits_cannot_be_relaxed():
    target = build_behavior_target(road(width=7.), (box(y=2.6),))
    report = qualify_episode(timestamps_s=[], positions_xyz=[], yaw_rad=[], velocity_world=[],
        lateral_acceleration_mps2=[], target=target)
    assert not report["passed"] and "behavior_target_requires_yield" in report["reasons"]
    for changed in ({"lateral_acceleration_limit_mps2": 4.}, {"lateral_jerk_limit_mps3": 6.},
                    {"retained_nominal_recovery_m": 4.}, {"maximum_absolute_offset_limit_m": .3}):
        with pytest.raises(ValueError, match="cannot relax"):
            QualityConfig(**changed)
