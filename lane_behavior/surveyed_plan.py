"""Build immutable native teacher plans from replayed collision curb toes.

Collision rays establish physical geometry for a demonstrator and its QA.
Interpolation is explicitly bounded and never establishes RGB visibility,
observed model support, accepted samples, or training qualification.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from lane_dataset.geometry import LaneReference
from . import curb_surface
from .scenarios import right_hand_corridor


VERSION = "collision_survey_native_teacher_plan_v2"
SCOPE = "collision_surveyed_curb_toes_with_bounded_interpolation_teacher_and_physical_qa_only"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(path, expected=None):
    path = Path(path).resolve()
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if expected is not None and actual != expected:
        raise ValueError("immutable source binding mismatch: "+str(path))
    return json.loads(payload), dict(path=str(path), sha256=actual)


def _verify_bindings(bindings):
    if not isinstance(bindings, dict) or not all(key in bindings for key in ("plan", "plan_sha256", "xodr", "xodr_sha256")):
        raise ValueError("survey must bind its parent plan and native XODR")
    verified = {}
    for name, value in bindings.items():
        if name.endswith("_sha256"):
            source = name[:-7]
            if source not in bindings or _sha(bindings[source]) != value:
                raise ValueError("immutable survey source binding mismatch: "+source)
            verified[source] = dict(path=str(Path(bindings[source]).resolve()), sha256=value)
    return verified


def _parent(plan, expected_xodr_sha):
    if (plan.get("status") != "offline_candidate_only" or plan.get("split") != "development"
            or plan.get("rendered_qualification") is not False or plan.get("source_kind") != "native_map"
            or plan.get("source_xodr_sha256") != expected_xodr_sha or plan.get("town") == "Town05"):
        raise ValueError("survey parent must be an unqualified native development plan")
    tasks = plan.get("tasks", [])
    if not tasks:
        raise ValueError("native parent has no tasks")
    first = tasks[0]
    route = first["route"]
    lane = first["start"]["lane_id"]
    if lane not in (-1, 1) or len(route["section"]["road_intervals"]) != 1:
        raise ValueError("one known native road and traversal lane required")
    for task in tasks:
        if (task.get("native_unmarked") is not True or task.get("procedural") is not False
                or task.get("split") != "development" or task.get("training_ready") is not False
                or task.get("group_id") != plan.get("geometry_group")
                or task["start"]["lane_id"] != lane or task["route"] != route
                or task["route"].get("source_xodr_sha256") != expected_xodr_sha):
            raise ValueError("native parent tasks do not share bound road geometry")
    edges = route["unmarked_road"]
    reference = LaneReference(edges["road_left_edge_xyz"], edges["road_right_edge_xyz"], route["arc_m"])
    return first, reference


def _survey_points(derivative, base, reference):
    if (derivative.get("version") != curb_surface.VERSION
            or derivative.get("scope") != "independent_collision_curb_toe_geometry_not_rendered_admission"
            or derivative.get("complete") is not True or derivative.get("budget_exhausted") is not False
            or derivative.get("training_ready") is not False
            or derivative.get("source_bindings") != base.get("source_bindings")
            or derivative.get("requested_stations") != base.get("requested_stations")
            or base.get("complete") is not True or base.get("budget_exhausted") is not False
            or base.get("training_ready") is not False):
        raise ValueError("complete bound collision survey replay required")
    if derivative.get("replay_source_sha256") != _sha(curb_surface.__file__):
        raise ValueError("curb detector implementation differs from survey replay binding")
    raw = {(row["side"], float(row["route_s"])):row for row in base["rows"]}
    rows = derivative["rows"]
    keys = [(row["side"], float(row["route_s"])) for row in rows]
    expected = {(side, float(station)) for side in ("left", "right") for station in derivative["requested_stations"]}
    if len(raw) != len(base["rows"]) or len(set(keys)) != len(keys) or set(keys) != set(raw) or set(keys) != expected:
        raise ValueError("collision survey stations missing or duplicated")
    points, rejected = {"left":[], "right":[]}, []
    for row in rows:
        side, station = row["side"], float(row["route_s"])
        source = raw[(side, station)]
        if any(row.get(key) != source.get(key) for key in ("nominal_outer_edge_xyz", "outward_unit", "samples")):
            raise ValueError("replayed curb row changed bound collision measurements")
        if row.get("passed") is not True:
            rejected.append(dict(side=side, source_route_s_m=station, reason=row.get("reason", "unqualified")))
            continue
        if not reference.s[0] <= station <= reference.s[-1]:
            raise ValueError("survey station outside parent road")
        point = getattr(reference, "_"+side)(station)
        outward = reference._right(station)-reference._left(station)
        outward[2] = 0.
        outward *= (-1. if side == "left" else 1.)/np.linalg.norm(outward)
        if not np.allclose(row["nominal_outer_edge_xyz"], point, atol=1e-8, rtol=0) or not np.allclose(row["outward_unit"], outward, atol=1e-8, rtol=0):
            raise ValueError("collision cross-section does not bind parent road geometry")
        samples = source["samples"]
        fit = curb_surface.detect_curb_step([s["offset_m"] for s in samples], [s["xyz"][2] for s in samples])
        if fit.get("passed") is not True:
            raise ValueError("claimed curb toe does not replay from collision samples")
        for key, value in fit.items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                if not math.isclose(float(row.get(key, math.nan)), float(value), abs_tol=1e-10, rel_tol=0):
                    raise ValueError("curb fit result differs from deterministic replay")
            elif row.get(key) != value:
                raise ValueError("curb fit result differs from deterministic replay")
        toe = point+outward*fit["toe_offset_m"]
        toe[2] = fit["toe_height_m"]
        if not np.allclose(row.get("toe_xyz"), toe, atol=1e-9, rtol=0):
            raise ValueError("curb toe XYZ disagrees with replayed physical cross-section")
        points[side].append((station, toe))
    for side in points:
        points[side].sort(key=lambda item:item[0])
        if len(points[side]) < 4:
            raise ValueError("insufficient qualified curb points on "+side)
    return points, rejected


def _runs(stations, maximum_gap):
    starts = np.r_[0, np.flatnonzero(np.diff(stations) > maximum_gap+1e-8)+1]
    ends = np.r_[starts[1:]-1, len(stations)-1]
    return [(float(stations[a]), float(stations[b])) for a, b in zip(starts, ends) if b-a >= 3]


def _paired_geometry(points, maximum_gap, spacing, maximum_toe_uncertainty, minimum_span_m=170.):
    per_side = {side:(np.array([p[0] for p in rows]), np.array([p[1] for p in rows])) for side, rows in points.items()}
    spans = []
    for left in _runs(per_side["left"][0], maximum_gap):
        for right in _runs(per_side["right"][0], maximum_gap):
            first, last = max(left[0], right[0]), min(left[1], right[1])
            if last > first:
                spans.append((first, last))
    if not spans:
        raise ValueError("no common curb span within the maximum interpolation gap")
    first, last = max(spans, key=lambda span:(span[1]-span[0], -span[0]))
    if last-first < minimum_span_m:
        raise ValueError("common qualified curb span too short for complete avoidance and recovery")
    query = np.unique(np.r_[first, np.arange(first, last, spacing), last,
        *(s[(s >= first) & (s <= last)] for s, _ in per_side.values())])
    curves, support = {}, {}
    for side, (s, xyz) in per_side.items():
        # Include exactly the bracketing passed points. No failed toe or outside
        # run can affect interpolation or the derivative at the cropped ends.
        a = max(0, int(np.searchsorted(s, first, side="right"))-1)
        b = min(len(s), int(np.searchsorted(s, last, side="left"))+1)
        knots, vertices = s[a:b], xyz[a:b]
        if len(knots) < 4 or np.diff(knots).max() > maximum_gap+1e-8:
            raise ValueError("curb interpolation uses an unsupported gap")
        errors = []
        for fold in (0, 1):
            fit = np.array([i for i in range(len(knots)) if i % 2 == fold or i in (0, len(knots)-1)])
            withheld = np.array([i for i in range(len(knots)) if i not in fit])
            if len(fit) < 4 or not len(withheld):
                raise ValueError("insufficient curb points for independent interpolation validation")
            prediction = CubicSpline(knots[fit], vertices[fit], axis=0)(knots[withheld])
            errors.extend(np.linalg.norm(prediction-vertices[withheld], axis=1).tolist())
        maximum_error = float(max(errors))
        if maximum_error+maximum_toe_uncertainty > .05:
            raise ValueError("held-out curb interpolation and toe uncertainty exceed five cm")
        curves[side] = CubicSpline(knots, vertices, axis=0, extrapolate=False)(query)
        support[side] = dict(passed_source_stations_m=knots.tolist(), passed_toe_xyz=vertices.tolist(),
            maximum_interpolated_gap_m=float(np.diff(knots).max()), maximum_withheld_error_m=maximum_error,
            maximum_toe_uncertainty_m=maximum_toe_uncertainty,
            withheld_plus_toe_uncertainty_m=maximum_error+maximum_toe_uncertainty,
            accuracy_scope="alternating_measured_station_validation_not_rendered_visibility")
    whole = LaneReference(curves["left"], curves["right"])
    return query, whole, support


def _make_plan(parent, parent_binding, base_plan, base_reference, query, whole, support, bindings, rejected, reverse):
    first, original = _parent(parent, bindings["xodr"]["sha256"])
    original_route = first["route"]
    if (parent["geometry_group"] != base_plan["geometry_group"] or parent["town"] != base_plan["town"]
            or str(parent["road_id"]) != str(base_plan["road_id"])
            or original_route["section"]["road_intervals"] != base_plan["tasks"][0]["route"]["section"]["road_intervals"]):
        raise ValueError("reverse parent does not bind the same geometry group and road interval")
    expected_left = base_reference.right_xyz[::-1] if reverse else base_reference.left_xyz
    expected_right = base_reference.left_xyz[::-1] if reverse else base_reference.right_xyz
    if (original.left_xyz.shape != expected_left.shape or not np.allclose(original.left_xyz, expected_left, atol=1e-4, rtol=0)
            or not np.allclose(original.right_xyz, expected_right, atol=1e-4, rtol=0)):
        raise ValueError("parent traversal does not correspond to surveyed physical road")
    length = float(whole.s[-1])
    if reverse:
        source_query = query[::-1]
        left, right, arc = whole.right_xyz[::-1], whole.left_xyz[::-1], length-whole.s[::-1]
        original_query = np.interp(source_query, base_reference.s, original.s[::-1])
        # np.interp accepts decreasing values, while the input xp remains ordered.
    else:
        source_query, left, right, arc = query, whole.left_xyz, whole.right_xyz, whole.s
        original_query = query.copy()
    geometry = right_hand_corridor(left, right, arc)
    geometry.update(physical_whole_road_center_xyz=((left+right)*.5).tolist(),
        source_scope=SCOPE, source_bindings=bindings,
        source_collision_route_stations_m=source_query.tolist(), original_source_arc_m=source_query.tolist(),
        interpolation_support=support)
    corridor = LaneReference(geometry["inferred_divider_xyz"], geometry["road_right_edge_xyz"], arc)
    _, yaw, curvature, _ = corridor.sample(arc)

    def remap(value, *, clip=False):
        value = float(value)
        if not clip and not original_query[0]-1e-8 <= value <= original_query[-1]+1e-8:
            raise ValueError("scenario station falls outside retained surveyed road")
        return float(np.interp(value, original_query, arc))

    route = deepcopy(original_route)
    route.update(left_xyz=geometry["inferred_divider_xyz"], right_xyz=geometry["road_right_edge_xyz"],
        center_xyz=geometry["right_lane_target_xyz"], arc_m=arc.tolist(), length_m=length,
        unmarked_road=geometry, curvature_1pm=curvature.tolist(), peak_curvature_1pm=float(abs(curvature).max()),
        yaw_change_rad=float(np.unwrap(yaw)[-1]-np.unwrap(yaw)[0]),
        boundary_semantics=dict(left="inferred_divider", right="collision_surveyed_curb_toe_pending_rendered_qualification"),
        source_geometry_scope=SCOPE, observation_envelope_qualified=False, marking_support_qualified=False,
        surveyed_plan_version=VERSION, geometry_interpolation_visibility_qualified=False,
        source_parent_route_interval_m=[float(original_query[0]), float(original_query[-1])],
        source_parent_plan=parent_binding, source_collision_survey=bindings["survey_replay"],
        native_road_interval_policy="original_group_interval_preserved_conservatively_for_split_audit")
    topology = np.asarray(original_route["driving_topology_center_xyz"], float)
    route["driving_topology_center_xyz"] = np.stack([np.interp(original_query, original.s, topology[:, axis]) for axis in range(3)], axis=1).tolist()
    interval = original_route["section"]["road_intervals"][0]
    native_first = float(original_route["start"]["s"])
    native_last = float(interval["end_m"] if first["start"]["lane_id"] == -1 else interval["start_m"])
    native_stations = np.linspace(native_first, native_last, len(original.s))
    route["start"] = {**original_route["start"], "s":float(np.interp(original_query[0], original.s, native_stations))}
    route["native_start_station_method"] = "parent_uniform_waypoint_grid_index_mapping_metadata_only"
    tasks = []
    for source in parent["tasks"]:
        task = deepcopy(source)
        short = "sequence_budget" in source
        behavior = task["unmarked_behavior"]
        phase_mapping = {}
        for name, bounds in behavior["planned_phase_ranges_m"].items():
            mapped = [remap(value, clip=True) for value in bounds]
            if mapped[1] <= mapped[0]:
                raise ValueError("retained survey span loses a complete scenario phase")
            phase_mapping[name] = dict(parent=bounds, surveyed=mapped,
                clipped=any(value < original_query[0] or value > original_query[-1] for value in bounds))
            behavior["planned_phase_ranges_m"][name] = mapped
        for actor in behavior["obstacles"]:
            previous = actor["route_s_m"]
            actor["route_s_m"] = remap(previous)
            actor["parent_route_s_m"] = previous
            actor["maximum_road_intrusion_m"] = .0 if actor["kind"] == "pedestrian" else .8
            actor["placement"] = "measured_bbox_at_collision_surveyed_curb_toe_ground_height"
            actor["placement_requires_actual_ego_footprint_feasibility"] = True
            actor["required_behavior"] = "nonzero_avoidance_and_complete_recovery_inside_right_half"
        behavior["station_remapping"] = phase_mapping
        task.update(episode_id=source["episode_id"]+"-collision-survey-v2-"+bindings["survey_replay"]["sha256"][:12],
            route=deepcopy(route), start=deepcopy(route["start"]), split="development",
            capture_start_route_m=remap(source["capture_start_route_m"], clip=not short),
            minimum_ticks=math.floor((length-60.)/(source["nominal_speed_mps"]*.05)),
            rendered_qualification=False, training_ready=False,
            qualification_blockers=["same_frame_rgb_depth_curb_toe_qualification_required",
                "actual_model_crop_causal_evidence_required", "physical_avoidance_recovery_capture_required",
                "existing_geometry_split_overlap_audit_required"])
        if short:
            from .sequence_guard import rebudget, phase_ranges
            budget = rebudget(source,length_m=length,start_s=task["capture_start_route_m"],
                obstacle_s=remap(source["sequence_budget"]["obstacle_s_m"]))
            task["sequence_budget"] = budget
            task["minimum_ticks"] = task["capture_ticks"] = budget["capture_ticks"]
            behavior["planned_phase_ranges_m"] = phase_ranges(budget,clear=not behavior["obstacles"])
            behavior["phase_scope"] = "surveyed_metric_budget_actual_bodies_teacher_and_recorded_timing_must_replay"
            behavior["station_remapping"]["recomputed_budget"] = budget["station_remapping"]
        elif task["capture_start_route_m"] < 5.:
            task["capture_start_route_m"] = 10.
            task["capture_start_replanned_after_cropping"] = True
        tasks.append(task)
    document = deepcopy(parent)
    document.update(surveyed_plan_version=VERSION, status="offline_candidate_only", source_scope=SCOPE,
        split="development", active_queue_modified=False, accepted_labels=0, rendered_qualification=False,
        training_ready=False, splits_frozen=False, tasks=tasks, task_count=len(tasks),
        source_parent_plan=parent_binding, collision_survey_bindings=bindings,
        retained_collision_source_interval_m=[float(query[0]), float(query[-1])],
        retained_teacher_metric_length_m=length, collision_survey_support=support,
        discarded_survey_rows=rejected, interpolated_geometry_visibility_qualified=False,
        interpolation_method="cubic_through_passed_collision_toes_only_no_extrapolation",
        surveyed_whole_road_width_range_m=[float(np.linalg.norm(right-left, axis=1).min()), float(np.linalg.norm(right-left, axis=1).max())])
    if "sequence_budget" in first:
        document["sequence_budget"] = deepcopy(tasks[0]["sequence_budget"])
    return json.loads(json.dumps(document, allow_nan=False))


def create_surveyed_plans(survey_replay, output_directory, *, reverse_plan=None, maximum_gap_m=8., sample_spacing_m=.5):
    """Replay bound collision points and write new plans without simulator calls."""
    if not 0 < maximum_gap_m <= 8. or not 0 < sample_spacing_m <= .5:
        raise ValueError("maximum 8 m interpolation gap and .5 m sample spacing required")
    survey, survey_binding = _load(survey_replay)
    base, base_binding = _load(survey["source_survey"], survey["source_survey_sha256"])
    bindings = _verify_bindings(survey["source_bindings"])
    bindings.update(survey_replay=survey_binding, raw_survey=base_binding,
        replay_detector=dict(path=str(Path(curb_surface.__file__).resolve()), sha256=_sha(curb_surface.__file__)))
    parent, parent_binding = _load(bindings["plan"]["path"], bindings["plan"]["sha256"])
    _, reference = _parent(parent, bindings["xodr"]["sha256"])
    points, rejected = _survey_points(survey, base, reference)
    maximum_uncertainty = max(float(row["toe_uncertainty_m"]) for row in survey["rows"] if row.get("passed") is True)
    short = all("sequence_budget" in task for task in parent["tasks"])
    if any("sequence_budget" in task for task in parent["tasks"]) != short:
        raise ValueError("parent tasks disagree on short sequence budget contract")
    query, whole, support = _paired_geometry(points, maximum_gap_m, sample_spacing_m, maximum_uncertainty,
        minimum_span_m=0. if short else 170.)
    inputs = [(parent, parent_binding, False)]
    if reverse_plan is not None:
        other, other_binding = _load(reverse_plan)
        if other["tasks"][0]["start"]["lane_id"] == parent["tasks"][0]["start"]["lane_id"]:
            raise ValueError("reverse parent must use the opposite native lane")
        inputs.append((other, other_binding, True))
    prepared = []
    output_directory = Path(output_directory).resolve()
    for source, source_binding, reverse in inputs:
        document = _make_plan(source, source_binding, parent, reference, query, whole, support, bindings, rejected, reverse)
        document["maximum_interpolated_gap_policy_m"] = float(maximum_gap_m)
        lane = document["tasks"][0]["start"]["lane_id"]
        filename = f"{document['geometry_group']}-{document['town']}-road{document['road_id']}-lane{lane}-collision-survey-v2-{survey_binding['sha256'][:12]}-candidate-plan.json"
        path = output_directory/filename
        if path.exists():
            raise FileExistsError("immutable surveyed plan already exists: "+str(path))
        prepared.append((document, path))
    output_directory.mkdir(parents=True, exist_ok=True)
    for document, path in prepared:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse-plan", type=Path)
    args = parser.parse_args()
    results = create_surveyed_plans(args.survey, args.output, reverse_plan=args.reverse_plan)
    print(json.dumps([dict(path=str(path), sha256=_sha(path),
        lane_id=document["tasks"][0]["start"]["lane_id"], length_m=document["retained_teacher_metric_length_m"],
        source_interval_m=document["retained_collision_source_interval_m"],
        rejected_survey_rows=len(document["discarded_survey_rows"]), training_ready=False) for document, path in results]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
