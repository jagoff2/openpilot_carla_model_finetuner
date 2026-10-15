import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lane_behavior.short_native import create_short_native_candidate, sequence_budget
from lane_behavior.qualification_probe import load_probe_task
from lane_dataset.config import Config


def test_short_budget_retains_five_second_history_and_ten_second_suffix():
    budget = sequence_budget(100.24)
    first, last = budget["planned_event_envelope_m"]
    assert first == pytest.approx(35.3)
    assert last == pytest.approx(64.7)
    assert first-budget["start_s_m"] >= budget["cruise_speed_mps"]*5.
    assert budget["planned_recovery_end_m"] == pytest.approx(last+5.)
    assert budget["required_suffix_end_m"] == pytest.approx(94.7)
    assert budget["required_suffix_end_m"] < 100.24-budget["road_end_body_guard_m"]
    assert not budget["actual_timing_qualified"]


def test_on_road_pass_has_room_by_moving_obstacle_earlier_not_shortening_recovery():
    budget=sequence_budget(99.9956,start_s=9.79064538183283,obstacle_s=45.,maximum_transition_m=14.)
    first,last=budget['planned_event_envelope_m']
    assert first-budget['start_s_m']>=12.5
    assert budget['planned_recovery_end_m']-last==5.
    assert budget['required_suffix_end_m']-budget['planned_recovery_end_m']==25.
    assert budget['required_suffix_end_m']<=99.9956-3.
    assert budget['capture_duration_s']==45.
    with pytest.raises(ValueError,match='road/capture budget'):
        sequence_budget(99.9956,start_s=9.79064538183283,obstacle_s=48.70431,maximum_transition_m=14.)


@pytest.mark.parametrize("kwargs", [dict(length_m=95.), dict(length_m=100., ticks=600),
    dict(length_m=100., start_s=30.), dict(length_m=100., obstacle_s=70.),
    dict(length_m=100., speed_mps=0.), dict(length_m=100., speed_mps=3.5)])
def test_short_budget_rejects_missing_history_future_or_recovery_room(kwargs):
    with pytest.raises(ValueError): sequence_budget(**kwargs)


def test_test_only_town05_remains_excluded(tmp_path):
    with pytest.raises(ValueError, match="Town05"):
        create_short_native_candidate(tmp_path, tmp_path/"Town05.xodr")


@pytest.fixture
def native_source():
    source = Path(Config().carla_root)/"CarlaUE4/Content/Carla/Maps/OpenDrive/Town04.xodr"
    if not source.exists(): pytest.skip("installed offline Town04 source required")
    return source


@pytest.mark.parametrize("lane_id,actor_s", [(-1, 50.), (1, 49.)])
def test_native_road52_is_offline_candidate_with_real_topology_and_no_server_calls(tmp_path, native_source, lane_id, actor_s):
    carla = pytest.importorskip("carla")
    # Deliberately expose only the offline Map constructor, no Client API.
    report, path = create_short_native_candidate(tmp_path, native_source, lane_id=lane_id,
        obstacle_s=actor_s, carla_module=SimpleNamespace(Map=carla.Map))
    assert report["road_id"] == 52
    assert report["proposed_split"] == "validation"
    assert report["split"] == "development"
    assert report["active_queue_modified"] is False
    assert report["rendered_qualification"] is False
    assert report["task_count"] == 3
    assert report["mapped_outer_curb_width_m"] == pytest.approx(9.27, abs=.001)
    task = load_probe_task(path, "parked_car")
    assert task["start"]["road_id"] == 52 and task["start"]["lane_id"] == lane_id
    assert task["training_ready"] is False and task["existing_capture_compatible"] is False
    assert task["minimum_ticks"] == 900 and task["capture_start_route_m"] == 10.
    actor = task["unmarked_behavior"]["obstacles"][0]
    assert actor["route_s_m"] == actor_s and actor["maximum_road_intrusion_m"] == .8
    edges = task["route"]["unmarked_road"]
    left, right = np.array(edges["road_left_edge_xyz"]), np.array(edges["road_right_edge_xyz"])
    np.testing.assert_allclose(task["route"]["center_xyz"], .25*left+.75*right)
    assert all(t["sequence_budget"]["actual_teacher_excursions_qualified"] is False for t in report["tasks"])
    # Same inputs are idempotent, changed capture intent cannot overwrite them.
    create_short_native_candidate(tmp_path, native_source, lane_id=lane_id, obstacle_s=actor_s,
        carla_module=SimpleNamespace(Map=carla.Map))
    with pytest.raises(FileExistsError, match="immutable"):
        create_short_native_candidate(tmp_path, native_source, lane_id=lane_id, ticks=950, obstacle_s=actor_s,
            carla_module=SimpleNamespace(Map=carla.Map))


def test_painted_native_road_does_not_become_unmarked_by_changing_budget(tmp_path, native_source):
    with pytest.raises(ValueError, match="painted|one driving"):
        create_short_native_candidate(tmp_path, native_source, road_id=50)


def test_explicit_road52_validation_and_road51_training_split_are_independent(tmp_path, native_source):
    carla = pytest.importorskip("carla")
    root = tmp_path/"corpus"
    root.mkdir()
    split = root/"splits.json"
    split.write_text(json.dumps(dict(splits_frozen=True, groups=[], assignments=[])))
    from lane_dataset.config import sha256_file
    (root/"qualification.json").write_text(json.dumps(dict(split_binding=dict(path=str(split), sha256=sha256_file(split)))))
    report, _ = create_short_native_candidate(tmp_path/"plans", native_source, root=root,
        carla_module=SimpleNamespace(Map=carla.Map))
    audit = report["split_audit"]
    assert audit["passed"] and not audit["errors"]
    candidate, training = audit["proposed_allocations"]
    assert candidate["split"] == "validation" and training["split"] == "train"
    assert candidate["section"]["road_intervals"][0]["road_id"] == "52"
    assert set(candidate["section"]["junction_ids"]) == {"916", "1103"}
    assert set(training["section"]["junction_ids"]) == {"148", "1197"}
