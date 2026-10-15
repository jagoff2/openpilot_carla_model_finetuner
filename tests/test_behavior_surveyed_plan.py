import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lane_behavior import curb_surface
from lane_behavior.qualification_probe import load_probe_task
from lane_behavior.scenarios import obstacle_case, right_hand_corridor
from lane_behavior.surveyed_plan import SCOPE, create_surveyed_plans


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, *, rejected=()):
    xodr = tmp_path/"Town04.xodr"
    xodr.write_text("<OpenDRIVE/>", encoding="utf-8")
    xodr_sha = hashlib.sha256(xodr.read_bytes()).hexdigest()
    probe = tmp_path/"probe.json"
    probe_sha = write(probe, {"scope":"provenance_only_no_rendered_qualification"})
    stations = np.arange(0., 202., 2.)
    left = np.c_[stations, stations*0-4.635, stations*0]
    right = np.c_[stations, stations*0+4.635, stations*0]
    parents = []
    for lane in (-1, 1):
        l, r = (left, right) if lane == -1 else (right[::-1], left[::-1])
        geometry = right_hand_corridor(l, r, stations)
        start = dict(road_id=51, lane_id=lane, s=0. if lane == -1 else 200.)
        route = dict(arc_m=stations.tolist(), length_m=200.,
            left_xyz=geometry["inferred_divider_xyz"], right_xyz=geometry["road_right_edge_xyz"],
            center_xyz=geometry["right_lane_target_xyz"],
            driving_topology_center_xyz=geometry["right_lane_target_xyz"], unmarked_road=geometry,
            start=start, source_xodr=str(xodr), source_xodr_sha256=xodr_sha,
            section=dict(road_intervals=[dict(road_id="51",start_m=0.,end_m=200.)], direction="straight",family="straight"))
        tasks = []
        for case in ("clear_road", "parked_car", "roadside_pedestrian"):
            tasks.append(dict(episode_id=f"native-test-{case}-{lane}", group_id="native-test", split="development",
                native_unmarked=True, procedural=False, training_ready=False, rendered_qualification=False,
                capture_contract="unmarked_rht_obstacle_v1", unmarked_road=True,
                route=route, start=start, opendrive_path=str(xodr), nominal_speed_mps=2.5,
                capture_start_route_m=10., unmarked_behavior=obstacle_case(case,200.,4.5)))
        parent = dict(status="offline_candidate_only", source_kind="native_map", split="development",
            rendered_qualification=False, geometry_group="native-test", town="Town04", road_id=51,
            source_xodr=str(xodr), source_xodr_sha256=xodr_sha, tasks=tasks)
        path = tmp_path/f"parent-{lane}.json"
        sha = write(path,parent)
        parents.append((path,sha))
    bindings = dict(plan=str(parents[0][0]),plan_sha256=parents[0][1],xodr=str(xodr),xodr_sha256=xodr_sha,
        probe=str(probe),probe_sha256=probe_sha)
    rows = []
    offsets = np.unique(np.r_[[-.9,-.8,-.7,-.5,-.3], np.arange(-.21,-.139,.005),[-.1,.1,.2,.3,.4]])
    heights = np.clip(5.*(offsets+.192),0.,.15)
    fit = curb_surface.detect_curb_step(offsets,heights)
    assert fit["passed"]
    for station in stations:
        for side, sign in (("left",-1.),("right",1.)):
            nominal = np.array([station,sign*4.635,0.])
            outward = np.array([0.,sign,0.])
            samples=[]
            for offset,height in zip(offsets,heights):
                xyz=nominal+outward*offset;xyz[2]=height
                samples.append(dict(offset_m=float(offset),xyz=xyz.tolist(),label="Ground",
                    all_hits=[dict(xyz=xyz.tolist(),label="Ground")]))
            toe=nominal+outward*fit["toe_offset_m"];toe[2]=fit["toe_height_m"]
            row=dict(route_s=float(station),side=side,nominal_outer_edge_xyz=nominal.tolist(),
                outward_unit=outward.tolist(),samples=samples,**fit,toe_xyz=toe.tolist())
            if (side,float(station)) in rejected:
                row.update(passed=False,reason="collision_surface_not_road_or_curb:Poles")
                row.pop("toe_xyz")
            rows.append(row)
    base=dict(version="bound_raw_collision_survey",complete=True,budget_exhausted=False,training_ready=False,
        requested_stations=stations.tolist(),rows=rows,source_bindings=bindings)
    base_path=tmp_path/"raw.json";base_sha=write(base_path,base)
    replay=dict(base,version=curb_surface.VERSION,
        scope="independent_collision_curb_toe_geometry_not_rendered_admission",
        passed=not rejected,source_survey=str(base_path),source_survey_sha256=base_sha,
        replay_source_sha256=hashlib.sha256(Path(curb_surface.__file__).read_bytes()).hexdigest())
    replay_path=tmp_path/"replay.json";write(replay_path,replay)
    return replay_path,parents[1][0],tmp_path/"output"


def test_builds_both_directions_from_passed_toes_with_bounded_pole_gaps(tmp_path):
    survey,reverse,out=fixture(tmp_path,rejected=(("left",38.),("right",50.)))
    original_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.glob("*.json")}
    results=create_surveyed_plans(survey,out,reverse_plan=reverse)
    assert len(results)==2
    a,b=(value for value,_ in results)
    assert a["retained_collision_source_interval_m"] == [0.,200.]
    assert a["retained_teacher_metric_length_m"] == pytest.approx(200.)
    assert len(a["discarded_survey_rows"])==2
    assert a["collision_survey_support"]["left"]["maximum_interpolated_gap_m"]==4.
    assert a["collision_survey_support"]["right"]["withheld_plus_toe_uncertainty_m"]<.05
    first=a["tasks"][0]["route"];second=b["tasks"][0]["route"]
    np.testing.assert_allclose(first["unmarked_road"]["road_left_edge_xyz"],
        np.asarray(second["unmarked_road"]["road_right_edge_xyz"])[::-1])
    np.testing.assert_allclose(np.asarray(first["center_xyz"])[:,1],4.443/2,atol=1e-9)
    np.testing.assert_allclose(np.asarray(second["center_xyz"])[:,1],-4.443/2,atol=1e-9)
    for document,path in results:
        assert path.name.endswith("-candidate-plan.json") and "collision-survey-v2" in path.name
        assert document["geometry_group"]=="native-test"
        assert document["source_scope"]==SCOPE
        assert document["training_ready"] is document["rendered_qualification"] is False
        assert document["active_queue_modified"] is False and document["accepted_labels"]==0
        assert document["collision_survey_bindings"]["probe"]["sha256"]
        for case in ("clear_road","parked_car","roadside_pedestrian"):
            task=load_probe_task(path,case)
            assert task["split"]=="development" and not task["training_ready"]
            assert task["route"]["section"]["road_intervals"]==[dict(road_id="51",start_m=0.,end_m=200.)]
        source=np.asarray(document["tasks"][0]["route"]["unmarked_road"]["original_source_arc_m"])
        assert np.all(np.diff(source)>0) if document["tasks"][0]["start"]["lane_id"]==-1 else np.all(np.diff(source)<0)
    assert original_sources=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.glob("*.json")}


def test_new_plan_is_immutable_and_old_files_are_not_overwritten(tmp_path):
    survey,reverse,out=fixture(tmp_path)
    results=create_surveyed_plans(survey,out,reverse_plan=reverse)
    before={str(path):path.read_bytes() for _,path in results}
    with pytest.raises(FileExistsError,match="immutable surveyed plan already exists"):
        create_surveyed_plans(survey,out,reverse_plan=reverse)
    assert before=={str(path):path.read_bytes() for _,path in results}


def test_missing_endpoint_crops_and_rebases_scenario_stations_without_changing_group(tmp_path):
    survey,reverse,out=fixture(tmp_path,rejected=(("left",0.),))
    results=create_surveyed_plans(survey,out,reverse_plan=reverse)
    forward=results[0][0];backward=results[1][0]
    assert forward["retained_collision_source_interval_m"]==[2.,200.]
    assert forward["retained_teacher_metric_length_m"]==pytest.approx(198.)
    task=forward["tasks"][1]
    assert task["start"]["s"]==pytest.approx(2.)
    assert task["unmarked_behavior"]["obstacles"][0]["route_s_m"]==pytest.approx(98.)
    assert task["unmarked_behavior"]["planned_phase_ranges_m"]["approach"]==pytest.approx([58.,73.])
    assert task["capture_start_route_m"]==pytest.approx(8.)
    assert backward["tasks"][1]["unmarked_behavior"]["obstacles"][0]["route_s_m"]==pytest.approx(100.)
    assert backward["tasks"][0]["route"]["source_parent_route_interval_m"]==[0.,198.]


def test_long_missing_gap_cannot_be_filled_to_make_a_complete_route(tmp_path):
    rejected=tuple(("left",float(s)) for s in range(92,110,2))
    survey,reverse,out=fixture(tmp_path,rejected=rejected)
    with pytest.raises(ValueError,match="too short"):
        create_surveyed_plans(survey,out,reverse_plan=reverse)
    assert not out.exists()


@pytest.mark.parametrize("source",["parent--1.json","raw.json","probe.json","Town04.xodr"])
def test_modified_immutable_source_fails_before_any_output(tmp_path,source):
    survey,reverse,out=fixture(tmp_path)
    with (tmp_path/source).open("a",encoding="utf-8") as stream:stream.write(" ")
    with pytest.raises(ValueError,match="binding mismatch"):
        create_surveyed_plans(survey,out,reverse_plan=reverse)
    assert not out.exists()


def test_derivative_toe_coordinates_must_replay_from_actual_collision_samples(tmp_path):
    survey,reverse,out=fixture(tmp_path)
    doc=json.loads(survey.read_text());doc["rows"][0]["toe_xyz"][1]+=.1;write(survey,doc)
    with pytest.raises(ValueError,match="toe XYZ"):
        create_surveyed_plans(survey,out,reverse_plan=reverse)
    assert not out.exists()


def test_reverse_parent_must_have_matching_physical_group_and_geometry(tmp_path):
    survey,reverse,out=fixture(tmp_path)
    doc=json.loads(reverse.read_text());doc["geometry_group"]="other";write(reverse,doc)
    with pytest.raises(ValueError,match="bound road geometry|same geometry group"):
        create_surveyed_plans(survey,out,reverse_plan=reverse)
    assert not out.exists()


def test_interpolation_gap_limit_cannot_be_relaxed_by_caller(tmp_path):
    survey,reverse,out=fixture(tmp_path)
    with pytest.raises(ValueError,match="maximum 8 m"):
        create_surveyed_plans(survey,out,reverse_plan=reverse,maximum_gap_m=8.1)


def test_short_measured_span_rebudgets_and_keeps_nine_hundred_ticks(tmp_path):
    from lane_behavior.short_native import sequence_budget
    from lane_behavior.sequence_guard import phase_ranges
    survey,_,out=fixture(tmp_path)
    derivative=json.loads(survey.read_text())
    base_path=Path(derivative['source_survey']);base=json.loads(base_path.read_text())
    parent_path=Path(base['source_bindings']['plan']);parent=json.loads(parent_path.read_text())
    for task in parent['tasks']:
        budget=sequence_budget(task['route']['length_m'],obstacle_s=50.)
        task.update(sequence_budget=budget,capture_ticks=900,minimum_ticks=900)
        for actor in task['unmarked_behavior']['obstacles']:actor['route_s_m']=50.
        task['unmarked_behavior']['planned_phase_ranges_m']=phase_ranges(budget,
            clear=not task['unmarked_behavior']['obstacles'])
    parent['sequence_budget']=parent['tasks'][0]['sequence_budget']
    binding=write(parent_path,parent)
    for document in (base,derivative):
        document['source_bindings']['plan_sha256']=binding
        document['rows']=[r for r in document['rows'] if r['route_s']<=100.]
        document['requested_stations']=[s for s in document['requested_stations'] if s<=100.]
    derivative['source_survey_sha256']=write(base_path,base);write(survey,derivative)
    document,path=create_surveyed_plans(survey,out)[0]
    assert document['retained_teacher_metric_length_m']==pytest.approx(100.)
    assert document['sequence_budget']['physical_route_length_m']==pytest.approx(100.)
    for task in document['tasks']:
        budget=task['sequence_budget']
        assert budget['station_coordinate_frame']=='surveyed_physical_whole_road_arc_m'
        assert budget['start_s_m']==task['capture_start_route_m']==10.
        assert budget['obstacle_s_m']==50.
        assert task['minimum_ticks']==task['capture_ticks']==900
        assert task['unmarked_behavior']['planned_phase_ranges_m']==phase_ranges(budget,
            clear=not task['unmarked_behavior']['obstacles'])
        assert budget['required_suffix_end_m']<100.-budget['road_end_body_guard_m']
    with pytest.raises(FileExistsError):create_surveyed_plans(survey,out)
