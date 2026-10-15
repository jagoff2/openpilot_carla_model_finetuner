from dataclasses import replace
import json

import numpy as np
import pytest

from lane_behavior.planned import planned_scene_geometry
from lane_behavior.target import build_behavior_target
from lane_behavior.teacher_quality import qualify_teacher_rollout, _events
from lane_behavior.quality import QualityConfig, _project
from test_behavior_planned import simulator_obstacle, simulator_road
from test_behavior_quality import telemetry
from test_behavior_target import road


@pytest.fixture(scope="module")
def avoidance():
    return planned_scene_geometry(simulator_road(), [simulator_obstacle(y=4.3)],
        source_metadata={"map_sha256": "c"*64, "scene_seed": 13})


def qualify(target, data=None, **kwargs):
    return qualify_teacher_rollout(target=target, target_sha256=target.contract_sha256,
        **(telemetry(target) if data is None else data), **kwargs)


def test_actual_teacher_avoidance_and_recovery_have_physical_scope_only(avoidance):
    report = qualify(avoidance, collision_count=0)
    assert report["passed"] and report["physical_passed"], report["reasons"]
    assert report["source_scope"] == "teacher_only"
    assert not report["training_ready"] and not report["supervision_allowed"]
    assert not report["observation_admission_passed"]
    assert report["simulator_stationarity_verification_required"]
    events = report["metrics"]["avoidance_events"]
    assert len(events) == 1 and events[0]["passed"]
    assert min(events[0]["phase_frame_counts"].values()) > 0
    assert events[0]["retained_nominal_distance_m"] >= 5.
    assert report["metrics"]["sweep"]["minimum_obstacle_clearance_m"]["parked"] >= .6
    assert len(report["telemetry_sha256"]) == 5


def test_recorded_target_sha_is_mandatory_and_bound(avoidance):
    data = telemetry(avoidance)
    for digest in ("", "0"*64, None):
        with pytest.raises(ValueError, match="target SHA256"):
            qualify_teacher_rollout(**data, target=avoidance, target_sha256=digest, collision_count=0)
    with pytest.raises(ValueError, match="changed after"):
        qualify(replace(avoidance, excursions=()), collision_count=0)


def test_observed_target_is_not_coerced_into_privileged_teacher_contract():
    target = build_behavior_target(road())
    with pytest.raises(TypeError, match="PlannedTarget"):
        qualify_teacher_rollout(target=target, target_sha256=target.contract_sha256,
            timestamps_s=[], positions_xyz=[], yaw_rad=[], velocity_world=[], lateral_acceleration_mps2=[])


def test_straight_teacher_quality_needs_no_observation_receipt():
    target = planned_scene_geometry(simulator_road(length=25.))
    report = qualify(target, telemetry(target, end=20.), collision_count=0)
    assert report["physical_passed"] and not report["observation_admission_passed"]
    assert report["metrics"]["tracking"]["maximum_absolute_target_offset_m"] < 1e-10
    assert report["metrics"]["avoidance_events"] == []


@pytest.mark.parametrize("case,reason", [
    ("center", "actual_vehicle_body_leaves_right_half"),
    ("offset", "mean_signed_target_offset"),
    ("acceleration", "lateral_acceleration_above_limit"),
    ("jerk", "lateral_jerk_above_limit"),
    ("collision", "collision"),
    ("missing_collision", "collision_sensor_evidence_missing_or_invalid"),
    ("missing_frame", "missing_duplicate_or_non20hz_frames"),
    ("teleport", "actual_pose_velocity_disagreement"),
])
def test_actual_physical_failures_cannot_be_hidden_by_qualified_planned_geometry(case, reason):
    target = planned_scene_geometry(simulator_road(length=25.))
    data, collisions = telemetry(target, end=20.), 0
    if case == "center":
        data["positions_xyz"][:, 1] = 0.
    elif case == "offset":
        data["positions_xyz"][:, 1] += .06
    elif case == "acceleration":
        data["lateral_acceleration_mps2"][:] = 3.51
    elif case == "jerk":
        data["lateral_acceleration_mps2"][20:] = .251
    elif case == "collision":
        collisions = 1
    elif case == "missing_collision":
        collisions = None
    elif case == "missing_frame":
        data["timestamps_s"] *= 2.
    else:
        data["positions_xyz"][20, 0] += 1.
    report = qualify(target, data, collision_count=collisions)
    assert not report["physical_passed"] and reason in report["reasons"]


def test_measured_body_clearance_is_rechecked_independently_of_teacher_path(avoidance):
    data = telemetry(avoidance)
    data["positions_xyz"][:, 1] = 3.
    data["yaw_rad"][:] = 0.
    report = qualify(avoidance, data, collision_count=0)
    assert "actual_obstacle_clearance_below_limit:parked" in report["reasons"]
    assert report["metrics"]["sweep"]["minimum_obstacle_clearance_m"]["parked"] == 0.


def test_unrecorded_departure_or_less_than_five_metres_recovery_fails(avoidance):
    event = avoidance.excursions[0]
    short = qualify(avoidance, telemetry(avoidance, end=event.end_s+4.), collision_count=0)
    assert "event_0:insufficient_retained_nominal_recovery" in short["reasons"]
    missing = qualify(avoidance, telemetry(avoidance, start=event.plateau_start_s+1.), collision_count=0)
    assert "event_0:avoidance_event_not_fully_traversed" in missing["reasons"]
    assert "event_0:avoidance_phase_not_recorded:departure" in missing["reasons"]


def test_nominal_recovery_measures_actual_offset_not_only_station(avoidance):
    data = telemetry(avoidance)
    selected = data["positions_xyz"][:, 0] > avoidance.excursions[0].end_s
    data["positions_xyz"][selected, 1] += .11
    report = qualify(avoidance, data, collision_count=0)
    assert "event_0:nominal_recovery_offset_above_limit" in report["reasons"]


def recovery_report(target, data):
    station, _, _ = _project(data["positions_xyz"][:, :2],target.s,target.xy)
    return _events(target,station,data["positions_xyz"],data["yaw_rad"],
        data["velocity_world"],QualityConfig())[0]


def set_recovery_heading(data, selected, degrees=2.):
    data["velocity_world"][selected,1] = data["velocity_world"][selected,0]*np.tan(np.radians(degrees))


def test_short_physical_settling_requires_full_stable_five_metre_window(avoidance):
    data=telemetry(avoidance)
    end=avoidance.excursions[0].end_s
    station=data["positions_xyz"][:,0]
    set_recovery_heading(data,(station>=end)&(station<end+.3))
    report=recovery_report(avoidance,data)
    assert report["passed"],report["reasons"]
    assert .3<=report["settling_distance_m"]<.5
    assert report["planned_return_end_s"]==end
    assert report["actual_recovery_start_s"]==station[report["actual_recovery_start_frame_index"]]
    assert report["actual_return_extended"]
    assert report["stable_nominal_recovery_distance_m"]>=5.
    assert report["maximum_nominal_heading_error_rad"]<=np.radians(1.)
    assert report["maximum_allowed_settling_distance_m"]==2.
    json.dumps(report,allow_nan=False)


@pytest.mark.parametrize("case",["late_settling","one_good_frame","mid_window_failure"])
def test_recovery_search_cannot_skip_unstable_window_or_settle_beyond_bound(avoidance,case):
    data=telemetry(avoidance)
    end=avoidance.excursions[0].end_s
    station=data["positions_xyz"][:,0]
    selected=(station>=end)&(station<end+2.3)
    if case=="one_good_frame":
        selected[np.flatnonzero(selected)[0]]=False
    elif case=="mid_window_failure":
        selected=(station>=end+3.)&(station<end+3.3)
    set_recovery_heading(data,selected)
    report=recovery_report(avoidance,data)
    assert not report["passed"]
    assert "nominal_recovery_heading_above_limit" in report["reasons"]
    assert "nominal_recovery_not_stable_within_settling_bound" in report["reasons"]
    assert report["actual_recovery_start_s"] is None


def test_recovery_settling_does_not_excuse_whole_trajectory_jerk(avoidance):
    data=telemetry(avoidance)
    end=avoidance.excursions[0].end_s
    station=data["positions_xyz"][:,0]
    set_recovery_heading(data,(station>=end)&(station<end+.3))
    data["lateral_acceleration_mps2"][20:]+= .3
    report=qualify(avoidance,data,collision_count=0)
    assert report["metrics"]["avoidance_events"][0]["passed"]
    assert not report["passed"]
    assert "lateral_jerk_above_limit" in report["reasons"]


def test_delayed_stable_start_needs_five_metres_after_actual_start(avoidance):
    end=avoidance.excursions[0].end_s
    data=telemetry(avoidance,end=end+5.1)
    station=data["positions_xyz"][:,0]
    set_recovery_heading(data,(station>=end)&(station<end+.3))
    report=recovery_report(avoidance,data)
    assert not report["passed"]
    assert report["actual_recovery_start_s"] is None


def test_blocked_teacher_plan_remains_yield_with_no_physical_or_admission_claim():
    target = planned_scene_geometry(simulator_road(width=7.), [simulator_obstacle(y=2.6)])
    report = qualify_teacher_rollout(target=target, target_sha256=target.contract_sha256,
        timestamps_s=[], positions_xyz=[], yaw_rad=[], velocity_world=[], lateral_acceleration_mps2=[])
    assert not report["physical_passed"] and report["reasons"][0] == "teacher_plan_requires_yield"
    assert not report["training_ready"]
