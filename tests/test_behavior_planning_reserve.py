"""Planning tracking room is distinct from unchanged measured clearance QA."""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from lane_dataset.controller import VehicleState
from lane_dataset.geometry import LaneReference
from lane_behavior.planned import PlannedObstacle, planned_scene_geometry
from lane_behavior.target import BehaviorConfig, EgoFootprint, _polygon_distance
from lane_behavior.teacher import PhysicalTeacher
from lane_behavior.teacher_quality import qualify_teacher_rollout
from test_behavior_planned import simulator_road, simulator_obstacle
from test_behavior_teacher import calibration, configuration
from test_behavior_quality import telemetry


def reserved_target(reserve=.1):
    return planned_scene_geometry(simulator_road(), [simulator_obstacle(y=4.3)],
        obstacle_tracking_reserve_m=reserve)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def test_legacy_zero_reserve_keeps_exact_archived_source_document_and_hash():
    # Captured from the pre-change implementation before adding this option.
    legacy = reserved_target(0.)
    assert legacy.contract_sha256 == "5c73ef9c5dbb11b9ae3f735a952c35d7013e9d92624d2ea496052045444b2a57"
    assert canonical_sha(legacy.document()) == "f03c81af426bd94962a42a0de0987811d87d8a3bd18c65cca932552c14eb2048"
    assert "obstacle_tracking_reserve_m" not in legacy.document()["source"]
    default = planned_scene_geometry(simulator_road(), [simulator_obstacle(y=4.3)])
    assert default.document() == legacy.document()


def test_nonzero_reserve_expands_planned_clearance_but_preserves_actual_limits():
    target, original = reserved_target(), reserved_target(0.)
    assert target.feasible, target.reasons
    assert target.config.parked_vehicle_clearance_m == .6
    assert target.config.pedestrian_clearance_m == 1.2
    assert target.document()["source"]["obstacle_tracking_reserve_m"] == .1
    assert target.contract_sha256 != original.contract_sha256
    assert target.excursions[0].offset_m < original.excursions[0].offset_m
    assert target.diagnostics["minimum_actual_obstacle_clearance_surplus_m"] >= .1
    assert target.diagnostics["minimum_obstacle_clearance_surplus_m"] >= 0.


@pytest.mark.parametrize("reserve", [-.01, float("nan"), float("inf"), True, "0.1"])
def test_invalid_tracking_reserve_is_rejected(reserve):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        reserved_target(reserve)


def test_changing_or_removing_nonzero_reserve_breaks_bound_contract():
    target = reserved_target()
    for reserve in (0., .2):
        with pytest.raises(ValueError, match="changed after"):
            replace(target, obstacle_tracking_reserve_m=reserve).verify_contract()


def test_actual_online_body_check_can_use_tracking_reserve_above_required_point_six():
    target = reserved_target()
    event = target.excursions[0]
    station = (event.plateau_start_s+event.plateau_end_s)*.5
    xy, tangent, *_ = target.sample(np.array([station]))
    yaw = math.atan2(tangent[0, 1], tangent[0, 0])
    # Shift toward the parked actor by 8 cm. The teacher path planned >=.7 m,
    # while the real body remains safe between .6 and .7 m.
    measured = xy[0]+np.array([-math.sin(yaw), math.cos(yaw)])*.08
    state = VehicleState(float(measured[0]), float(measured[1]), yaw, 2.5, 0., timestamp_s=1.)
    teacher = PhysicalTeacher(target, calibration(), config=configuration())
    body, failure = teacher._body_check(state)
    assert failure is None
    actual_clearance = _polygon_distance(body, target.obstacles[0].polygon())
    assert .6 < actual_clearance < .7
    assert teacher.step(state).accepted


def test_physical_quality_replays_reserved_target_without_raising_actual_clearance():
    target = reserved_target()
    report = qualify_teacher_rollout(target=target, target_sha256=target.contract_sha256,
        **telemetry(target, step=.125), collision_count=0)
    assert report["passed"], report["reasons"]
    assert report["metrics"]["sweep"]["required_obstacle_clearance_m"] == {"parked": .6}


def test_admission_reconstructs_optional_reserve_and_rejects_dropped_field():
    from lane_behavior.admission import reconstruct_teacher
    target = reserved_target()
    assert reconstruct_teacher(target.document()).contract_sha256 == target.contract_sha256
    document = target.document()
    del document["source"]["obstacle_tracking_reserve_m"]
    with pytest.raises(ValueError, match="replay exactly"):
        reconstruct_teacher(document)
    assert reconstruct_teacher(reserved_target(0.).document()).obstacle_tracking_reserve_m == 0.


def test_measured_road52_geometry_and_prior_actual_boxes_provide_point_seven_planned_clearance():
    path = Path(r"G:\lane-centering-dataset-v1\reports\unmarked\unmarked-native-e7f22204bb57931c286f-Town04-road52-lane-1-collision-survey-v2-0375a57a4098-offline-teacher-check.json")
    if not path.exists(): pytest.skip("local measured road52 geometry required")
    source = json.loads(path.read_text())["cases"]["parked_car"]["target"]["source"]
    ref = source["reference"]
    target = planned_scene_geometry(LaneReference(ref["left_xyz"], ref["right_xyz"], ref["s"]),
        [PlannedObstacle(**o) for o in source["obstacles"]], EgoFootprint(**source["ego"]),
        config=BehaviorConfig(**source["config"]), obstacle_tracking_reserve_m=.1)
    assert target.feasible, target.reasons
    assert target.config.parked_vehicle_clearance_m == .6
    assert target.diagnostics["minimum_actual_obstacle_clearance_surplus_m"] >= .1
    assert .7+target.diagnostics["minimum_obstacle_clearance_surplus_m"] >= .7
    assert target.excursions[-1].end_s+5. < target.s[-1]
