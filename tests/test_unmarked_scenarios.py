from dataclasses import replace
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from lane_behavior.scenarios import (
    CASES, REQUIRED_CAPTURE_FEATURES, UnmarkedRoadParameters, build_unmarked_opendrive,
    create_candidate_plan, create_native_candidate_plan, obstacle_case, right_hand_corridor, route_from_opendrive,
)


@pytest.mark.parametrize("shape", ["straight", "constant_radius", "s_bend", "tightening", "hairpin"])
def test_unmarked_road_has_physical_edges_without_painted_boundaries(shape):
    p = UnmarkedRoadParameters(shape=shape)
    root = ET.fromstring(build_unmarked_opendrive(p))
    assert root.find("road").attrib["rule"] == "RHT"
    marks = root.findall(".//roadMark")
    assert len(marks) == 5
    assert all(mark.attrib["type"] == "none" and float(mark.attrib["width"]) == 0 for mark in marks)
    lanes = root.findall(".//laneSection/*/lane")
    assert {int(lane.attrib["id"]) for lane in lanes if lane.attrib["type"] == "driving"} == {-1, 1}
    sidewalks = [lane for lane in lanes if lane.attrib["type"] == "sidewalk"]
    assert len(sidewalks) == 2
    assert all(float(lane.find("height").attrib["inner"]) == .15 for lane in sidewalks)


def test_right_hand_target_is_quarter_width_from_right_edge_not_road_middle():
    x = np.arange(5., dtype=float)
    left, right = np.c_[x, np.full(5, -3.6), x*0], np.c_[x, np.full(5, 3.6), x*0]
    result = right_hand_corridor(left, right)
    np.testing.assert_allclose(np.array(result["right_lane_target_xyz"])[:, 1], 1.8)
    np.testing.assert_allclose(np.array(result["inferred_divider_xyz"])[:, 1], 0.)
    assert result["rendered_qualification"] is False
    assert "not_paint" in result["divider_evidence"]


def test_reversed_traversal_still_uses_its_own_right_half():
    x = np.arange(5., dtype=float)
    left, right = np.c_[x, np.full(5, -3.6), x*0], np.c_[x, np.full(5, 3.6), x*0]
    result = right_hand_corridor(right[::-1], left[::-1])
    np.testing.assert_allclose(np.array(result["right_lane_target_xyz"])[:, 1], -1.8)
    with pytest.raises(ValueError, match="swapped"):
        right_hand_corridor(left[::-1], right[::-1])


def test_geometry_identity_excludes_split_case_mirror_and_irrelevant_straight_radius():
    p = UnmarkedRoadParameters()
    assert p.geometry_group == replace(p, split="test", radius_m=20., turn="right").geometry_group
    assert p.geometry_group != replace(p, lane_width_m=3.7).geometry_group
    curve = replace(p, shape="constant_radius")
    assert curve.geometry_group == replace(curve, turn="right", split="train").geometry_group
    assert curve.geometry_group != replace(curve, radius_m=33.).geometry_group


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.])
def test_invalid_physical_parameters_fail_before_generation(value):
    with pytest.raises(ValueError):
        build_unmarked_opendrive(UnmarkedRoadParameters(lane_width_m=value))


@pytest.mark.parametrize("case", CASES)
def test_obstacle_plans_have_explicit_recovery_and_no_claimed_rendered_evidence(case):
    result = obstacle_case(case, 240., 3.6)
    assert not result["rendered_qualification"] and not result["training_ready"]
    assert result["recovery_target"] == "same_inferred_right_lane_center_after_obstacle_clear"
    assert result["no_safe_corridor_response"] == "yield_do_not_label_collision_path"
    assert result["crossing_divider_allowed_only_for_observed_obstruction"] is False
    if case != "clear_road":
        phases = result["planned_phase_ranges_m"]
        assert phases["recovery"][1] == phases["right_lane_follow"][0]
        assert result["obstacles"][0]["placement_requires_actual_bounding_box"]
        assert result["obstacles"][0]["teleport_after_spawn"] is False
    if case == "roadside_pedestrian":
        assert result["obstacles"][0]["motion"] == "stationary"
        assert result["obstacles"][0]["minimum_body_clearance_m"] == 1.2
        assert result["obstacles"][0]["maximum_road_intrusion_m"] == .2


@pytest.mark.parametrize("lane_id", [-1, 1])
@pytest.mark.parametrize("shape", ["straight", "constant_radius", "s_bend", "tightening", "hairpin"])
def test_actual_carla_parser_matches_virtual_corridor_and_directions(lane_id, shape):
    carla = pytest.importorskip("carla")
    p = UnmarkedRoadParameters(shape=shape)
    parsed = carla.Map(p.geometry_group, build_unmarked_opendrive(p))
    route = route_from_opendrive(parsed, p, lane_id=lane_id)
    assert route["length_m"] > 230.
    if shape == "straight":
        np.testing.assert_allclose(np.array(route["center_xyz"])[:, 1], 2.25 if lane_id == -1 else -2.25, atol=1e-5)
    assert route["boundary_semantics"]["left"] == "inferred_divider"
    assert route["source_geometry_scope"] == "offline_CARLA_map_parse_not_rendered_evidence"


def test_candidate_plan_never_joins_active_queue_or_claims_ready(tmp_path):
    carla = pytest.importorskip("carla")
    report, path = create_candidate_plan(tmp_path, UnmarkedRoadParameters(), carla_module=carla)
    saved = json.loads(path.read_text())
    assert saved == report
    assert report["status"] == "offline_candidate_only"
    assert not report["active_queue_modified"]
    assert len(report["tasks"]) == 3
    assert len({task["group_id"] for task in report["tasks"]}) == 1
    for task in report["tasks"]:
        assert not task["training_ready"] and not task["existing_capture_compatible"]
        assert task["required_capture_features"] == list(REQUIRED_CAPTURE_FEATURES)
        assert task["qualification_blockers"]
        assert task["capture_start_route_m"] == 10.
        assert task["minimum_ticks"] > 1000


def test_native_unmarked_plan_preserves_outer_curbs_and_shared_geometry_both_directions(tmp_path):
    from pathlib import Path
    from lane_dataset.config import Config
    source=Path(Config().carla_root)/"CarlaUE4/Content/Carla/Maps/OpenDrive/Town04.xodr"
    if not source.exists():
        pytest.skip("installed native Town04 OpenDRIVE required")
    carla=pytest.importorskip("carla")
    reports=[]
    for lane_id in (-1,1):
        report,path=create_native_candidate_plan(tmp_path,source,51,lane_id=lane_id,carla_module=carla)
        reports.append(report)
        assert report["source_kind"] == "native_map"
        assert report["mapped_outer_curb_width_m"] == pytest.approx(9.27,abs=1e-4)
        assert report["driving_topology_lane_width_m"] == 3.5
        for task in report["tasks"]:
            assert task["procedural"] is False and task["native_unmarked"] is True
            route=task["route"]
            assert route["mapped_outer_curb_lane_ids"] == {"left":3,"right":-3}
            assert route["start"]["lane_id"] == lane_id
            assert route["length_m"] > 210.
            physical=route["unmarked_road"]
            left,right=np.array(physical["road_left_edge_xyz"]),np.array(physical["road_right_edge_xyz"])
            np.testing.assert_allclose(route["center_xyz"],.25*left+.75*right)
            assert abs(np.linalg.norm(np.array(route["center_xyz"])-np.array(route["driving_topology_center_xyz"]),axis=1).mean()-.5675)<1e-4
            if task["unmarked_behavior"]["case"] == "parked_car":
                obstacle=task["unmarked_behavior"]["obstacles"][0]
                assert obstacle["maximum_road_intrusion_m"] == .8
                assert obstacle["placement_requires_actual_ego_footprint_feasibility"]
    assert reports[0]["geometry_group"] == reports[1]["geometry_group"]


def test_native_development_plan_excludes_test_only_town05(tmp_path):
    with pytest.raises(ValueError,match="Town05"):
        create_native_candidate_plan(tmp_path,tmp_path/"Town05.xodr",51)


def test_inner_curb_refinement_requires_bound_native_map_and_remains_unqualified(tmp_path):
    from pathlib import Path
    from lane_dataset.config import Config,sha256_file
    source=Path(Config().carla_root)/"CarlaUE4/Content/Carla/Maps/OpenDrive/Town04.xodr"
    if not source.exists():
        pytest.skip("installed native Town04 OpenDRIVE required")
    carla=pytest.importorskip("carla")
    path=tmp_path/'refinement.json'
    refinement=dict(proposed_edge_policy='inner_boundary_of_curb_shoulder',town='Town04',road_id=51,
        source_xodr_sha256=sha256_file(source),paired_edge_qualified=False)
    path.write_text(json.dumps(refinement))
    report,plan=create_native_candidate_plan(tmp_path,source,51,carla_module=carla,edge_refinement=path)
    assert 'inner-curb' in plan.name
    assert report['mapped_outer_curb_width_m']==pytest.approx(8.,abs=1e-4)
    assert not report['rendered_qualification']
    assert report['tasks'][0]['route']['mapped_edge_definition']=='inner_boundary_of_curb_shoulder'
    assert report['edge_reference_selection']['sha256']==sha256_file(path)
    refinement['source_xodr_sha256']='0'*64
    path.write_text(json.dumps(refinement))
    with pytest.raises(ValueError,match='does not bind'):
        create_native_candidate_plan(tmp_path,source,51,carla_module=carla,edge_refinement=path)
