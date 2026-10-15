"""Offline short native unmarked-road candidates with explicit sequence budgets.

This builder reads OpenDRIVE and calls carla.Map only. It never connects to a
server. Mapped curbs remain unqualified until collision survey and RGB/depth
replay establish their actual surfaces. All outputs remain development tasks.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from lane_dataset.config import sha256_file, write_json
from lane_dataset.inventory import route_document
from lane_dataset.scenario_planner import direct_junction_links
from lane_dataset.splits import (GeometryGroup, GeometrySection, RoadInterval,
    SplitAssignment, audit_split_overlap)
from .scenarios import CASES, REQUIRED_CAPTURE_FEATURES, right_hand_corridor


VERSION = "short_native_unmarked_candidate_v1"
ALLOWED_TOWNS = ("Town01", "Town02", "Town03", "Town04", "Town06", "Town07", "Town10HD")


def sequence_budget(length_m, *, start_s=10., obstacle_s=50., speed_mps=2.5,
                    ticks=900, dt=.05, maximum_transition_m=8.):
    """Budget conservative event envelopes, not fabricated physical success.

    The actual actor, ego and planned maneuver must satisfy these envelopes
    after independent physical-edge survey. Duration allowances do not prove a
    vehicle will achieve progress; episode QA must verify traversal and timing.
    """
    values = np.asarray([length_m, start_s, obstacle_s, speed_mps, dt, maximum_transition_m], float)
    if (not np.isfinite(values).all() or type(ticks) is not int or not 400 <= ticks <= 2400
            or speed_mps <= 0 or speed_mps > 2.5 or dt != .05 or start_s < 5.
            or not 6. <= maximum_transition_m <= 14.):
        raise ValueError("invalid bounded short native sequence budget")
    history_s, future_s, startup_allowance_s = 5., 10., 8.
    actor_half_length, ego_half_length, clearance, transition = 3., 2.5, 1.2, float(maximum_transition_m)
    half_pass = actor_half_length+ego_half_length+clearance
    first = obstacle_s-half_pass-transition
    last = obstacle_s+half_pass+transition
    recovery_end = last+5.
    road_end_guard = 3.
    suffix_end = recovery_end+speed_mps*future_s
    if (first-start_s < speed_mps*history_s or suffix_end > length_m-road_end_guard
            or obstacle_s <= start_s or ticks*dt < (suffix_end-start_s)/speed_mps+startup_allowance_s):
        raise ValueError("road/capture budget cannot retain history, complete avoidance, recovery and future")
    return dict(scope="offline_budget_requires_actual_actor_teacher_and_recorded_timing_recheck",
        station_coordinate_frame="candidate_route_arc_requires_rebudget_after_physical_edge_reparameterization",
        start_s_m=start_s, obstacle_s_m=obstacle_s, cruise_speed_mps=speed_mps,
        capture_ticks=ticks, capture_dt_s=dt, capture_duration_s=ticks*dt,
        required_history_s=history_s, required_future_s=future_s,
        startup_allowance_s=startup_allowance_s, maximum_actor_half_length_m=actor_half_length,
        maximum_ego_half_length_m=ego_half_length, maximum_obstacle_clearance_m=clearance,
        maximum_transition_m=transition, minimum_retained_nominal_recovery_m=5.,
        planned_event_envelope_m=[first, last], planned_recovery_end_m=recovery_end,
        required_suffix_end_m=suffix_end, road_end_body_guard_m=road_end_guard,
        actual_timing_qualified=False, actual_body_dimensions_qualified=False,
        actual_teacher_excursions_qualified=False,
        validation_requires="actual phases inside envelope; >=5s recorded prefix and >=10s suffix around supervised event")


def _road_xml(source, road_id):
    source = Path(source)
    if source.stem not in ALLOWED_TOWNS:
        raise ValueError("short native candidates exclude test-only Town05 and unknown towns")
    xml = source.read_text(encoding="utf-8")
    matches = [r for r in ET.fromstring(xml).findall("road") if r.get("id") == str(road_id)]
    if len(matches) != 1 or matches[0].get("junction") != "-1":
        raise ValueError("one native non-junction road required")
    road = matches[0]
    sections = road.findall("lanes/laneSection")
    if len(sections) != 1 or float(sections[0].get("s", "0")) != 0.:
        raise ValueError("short road needs one unchanged lane section")
    section = sections[0]
    center = section.find("center/lane")
    if center is None or any(m.get("type", "none") != "none" for m in center.findall("roadMark")):
        raise ValueError("road centerline is painted")
    edges, topology = {}, {}
    for side, sign in (("left", 1), ("right", -1)):
        lanes = sorted(section.findall(side+"/lane"), key=lambda lane:abs(int(lane.get("id"))))
        driving = [lane for lane in lanes if lane.get("type") == "driving"]
        if len(driving) != 1 or int(driving[0].get("id")) != sign:
            raise ValueError("exactly one driving lane per direction required")
        edge, topology[side] = None, []
        for lane in lanes:
            marks = [m.get("type", "none") for m in lane.findall("roadMark")]
            topology[side].append(dict(lane_id=int(lane.get("id")), type=lane.get("type"), marks=marks))
            if lane.get("type") == "sidewalk":
                break
            if lane.get("type") not in ("driving", "shoulder") or any(m not in ("none", "curb") for m in marks):
                raise ValueError("painted or unsupported road side")
            if "curb" in marks:
                if any(m != "curb" for m in marks):
                    raise ValueError("curb boundary changes within road")
                edge = int(lane.get("id"))
        if edge is None:
            raise ValueError("physical outer curb must be explicitly mapped")
        edges[side] = edge
    return source, xml, road, edges, topology


def _split_audit(root, xml, town, road, *, training_road_id=51):
    """Audit complete roads and directly connected junction dependencies."""
    root = Path(root)
    binding = json.loads((root/"qualification.json").read_text())["split_binding"]
    path = Path(binding["path"])
    if not path.is_absolute(): path = root/path
    if sha256_file(path) != binding["sha256"]:
        raise ValueError("frozen split binding changed")
    baseline = json.loads(path.read_text())
    if baseline.get("splits_frozen") is not True:
        raise ValueError("baseline splits must be frozen")
    def section(doc):
        return GeometrySection(**{**doc, "road_intervals":tuple(RoadInterval(**i) for i in doc["road_intervals"])})
    groups = [GeometryGroup(g["group_id"], tuple(section(s) for s in g["sections"])) for g in baseline["groups"]]
    assignments = [SplitAssignment(**a) for a in baseline["assignments"]]
    links = direct_junction_links(xml)
    declarations = []
    for current, split in ((road, "validation"), (next(r for r in ET.fromstring(xml).findall("road")
            if r.get("id") == str(training_road_id)), "train")):
        rid = current.get("id")
        js = tuple(sorted({l.get("elementId") for l in current.findall("link/*") if l.get("elementType") == "junction"}))
        s = GeometrySection("unmarked-proposed-"+town+"-"+rid, town, "constant_radius",
            (RoadInterval(rid, 0., float(current.get("length"))),), js, links, direction="left")
        groups.append(GeometryGroup(s.section_id, (s,)))
        assignments.append(SplitAssignment(s.section_id, split, s.family, 1, 20261015))
        declarations.append(dict(split=split, section=asdict(s)))
    # Supplemental published data must also retain its previous allocation.
    seen = set()
    for manifest_path in sorted((root/"accepted").glob("*/behavior-manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        split_path = manifest_path.parent/"behavior-split.json"
        if manifest.get("files", {}).get("behavior-split.json") != sha256_file(split_path):
            raise ValueError("published behavior split binding changed")
        s = section(json.loads(split_path.read_text())["candidate_section"])
        if s.section_id in seen: continue
        seen.add(s.section_id)
        groups.append(GeometryGroup(s.section_id, (s,)))
        assignments.append(SplitAssignment(s.section_id, manifest["split"], s.family, 1, 0))
    errors = audit_split_overlap(groups, assignments)
    return dict(passed=not errors, errors=errors, baseline_path=str(path), baseline_sha256=binding["sha256"],
        proposed_allocations=declarations, scope="candidate independence only; no production allocation changed")


def create_short_native_candidate(output, source_xodr, road_id=52, *, root=None, lane_id=-1,
                                  carla_module=None, start_s=10., obstacle_s=50., ticks=900):
    if lane_id not in (-1, 1): raise ValueError("traversal lane must be +/-1")
    source, xml, road, edge_ids, topology = _road_xml(source_xodr, road_id)
    budget = sequence_budget(float(road.get("length")), start_s=start_s, obstacle_s=obstacle_s, ticks=ticks)
    if carla_module is None:
        import carla as carla_module
    native = carla_module.Map(source.stem, xml)
    stations = np.arange(.01, float(road.get("length"))-.01, .5)
    if lane_id == 1: stations = stations[::-1]
    points = [native.get_waypoint_xodr(int(road_id), lane_id, float(s)) for s in stations]
    if any(p is None for p in points): raise ValueError("native driving topology has a hole")
    _, route = route_document(points, source.stem)
    left, right = [], []
    for station in stations:
        cross = []
        for side in ("left", "right"):
            p = native.get_waypoint_xodr(int(road_id), edge_ids[side], float(station))
            if p is None: raise ValueError("native mapped curb topology has a hole")
            loc, normal = p.transform.location, p.transform.get_right_vector()
            cross.append([loc.x+.5*p.lane_width*normal.x, loc.y+.5*p.lane_width*normal.y, loc.z+.5*p.lane_width*normal.z])
        left.append(cross[0 if lane_id == -1 else 1]); right.append(cross[1 if lane_id == -1 else 0])
    # Recheck the sampled route length, which may be shorter than XML by a step.
    budget = sequence_budget(route["length_m"], start_s=start_s, obstacle_s=obstacle_s, ticks=ticks)
    whole = right_hand_corridor(left, right, route["arc_m"])
    route.update(driving_topology_center_xyz=route["center_xyz"], left_xyz=whole["inferred_divider_xyz"],
        right_xyz=whole["road_right_edge_xyz"], center_xyz=whole["right_lane_target_xyz"], unmarked_road=whole,
        boundary_semantics={"left":"inferred_divider", "right":"mapped_outer_curb_pending_collision_and_rendered_qualification"},
        source_geometry_scope="offline_native_CARLA_map_parse_not_rendered_evidence", native_lane_topology=topology,
        mapped_outer_curb_lane_ids=edge_ids, mapped_edge_definition="outer_boundary_of_curb_shoulder",
        source_xodr=str(source.resolve()), source_xodr_sha256=sha256_file(source), observation_envelope_qualified=False)
    identity = dict(town=source.stem, road_id=int(road_id), source_sha256=sha256_file(source), interval_m=[0., float(road.get("length"))])
    group = "unmarked-native-"+hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    tasks = []
    for case in CASES:
        behavior = dict(case=case, baseline="inferred_right_lane_center", recovery_target="same_inferred_right_lane_center_after_obstacle_clear",
            crossing_divider_allowed_only_for_observed_obstruction=False, opposing_half_policy="whole_ego_body_stays_in_right_half_v1",
            oncoming_conflict_response="yield_before_conflict_do_not_force_pass", no_safe_corridor_response="yield_do_not_label_collision_path",
            rendered_qualification=False, training_ready=False, obstacles=[],
            planned_phase_ranges_m={"approach":[start_s, budget["planned_event_envelope_m"][0]],
                "avoidance_and_return":budget["planned_event_envelope_m"],
                "nominal_recovery":[budget["planned_event_envelope_m"][1], budget["planned_recovery_end_m"]]},
            phase_scope="bounded_intent_only_actual_surveyed_teacher_and_recorded_phases_must_replay")
        if case != "clear_road":
            pedestrian = case == "roadside_pedestrian"
            behavior["obstacles"] = [dict(kind="pedestrian" if pedestrian else "parked_vehicle",
                blueprint_filter="walker.pedestrian.*" if pedestrian else "vehicle.audi.a2", route_s_m=obstacle_s,
                maximum_road_intrusion_m=0. if pedestrian else .8, minimum_body_clearance_m=1.2 if pedestrian else .6,
                placement="measured_bbox_at_native_physical_road_edge_collision_ground_height", motion="stationary",
                pedestrian_controller=False, teleport_after_spawn=False, placement_requires_actual_bounding_box=True,
                placement_requires_actual_ego_footprint_feasibility=True, dimension_source="must_measure_spawned_actor_not_blueprint_guess",
                required_behavior="nonzero_avoidance_and_complete_recovery_inside_right_half")]
        else:
            behavior["planned_phase_ranges_m"] = {"right_lane_follow":[start_s+12.5, budget["planned_recovery_end_m"]]}
        tasks.append(dict(episode_id=f"{group}-{case}-lane{lane_id}-short-v1", group_id=group, split="development",
            family=route["section"]["family"], behavior_family="unmarked_rht_"+case, town=source.stem,
            procedural=False, native_unmarked=True, opendrive_path=str(source.resolve()), route=route, start=route["start"],
            branch=0, capture_start_route_m=start_s, appearance=0, seed=20261015, nominal_speed_mps=2.5,
            offset_m=0., heading_error_deg=0., planned_preview_m=25., warmup_distance_budget_m=20.,
            minimum_ticks=ticks, capture_ticks=ticks, sequence_budget=budget, direction=route["section"]["direction"],
            unmarked_behavior=behavior, unmarked_road=True, capture_contract="unmarked_rht_obstacle_v1",
            required_capture_features=list(REQUIRED_CAPTURE_FEATURES), existing_capture_compatible=False,
            rendered_qualification=False, training_ready=False, qualification_blockers=[
                "independent_collision_curb_survey_required", "same_frame_cropped_rgb_depth_edge_proof_required",
                "actual_actor_and_teacher_within_short_sequence_budget", "complete_physical_avoidance_and_recovery_required",
                "independent_visual_and_split_admission_required"]))
    report = dict(schema_version=1, builder_version=VERSION, status="offline_candidate_only", active_queue_modified=False,
        rendered_qualification=False, splits_frozen=False, geometry_group=group, split="development", proposed_split="validation",
        source_kind="native_map", town=source.stem, road_id=int(road_id), source_xodr=str(source.resolve()),
        source_xodr_sha256=sha256_file(source), mapped_outer_curb_width_m=float(np.median(np.linalg.norm(np.asarray(left)-right, axis=1))),
        driving_topology_lane_width_m=float(np.median([p.lane_width for p in points])), task_count=len(tasks), tasks=tasks,
        sequence_budget=budget, split_audit=None if root is None else _split_audit(root, xml, source.stem, road))
    report = json.loads(json.dumps(report, allow_nan=False))
    path = Path(output)/f"{group}-{source.stem}-road{road_id}-lane{lane_id}-short-v1-candidate-plan.json"
    if path.exists():
        if json.loads(path.read_text()) != report: raise FileExistsError("immutable candidate plan already differs")
    else: write_json(path, report)
    return report, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-xodr", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--road-id", type=int, default=52)
    parser.add_argument("--lane-id", type=int, choices=(-1, 1), default=-1)
    report, path = create_short_native_candidate(**vars(parser.parse_args()))
    print(json.dumps(dict(path=str(path), status=report["status"], split_audit=report["split_audit"], budget=report["sequence_budget"])))


if __name__ == "__main__": main()
