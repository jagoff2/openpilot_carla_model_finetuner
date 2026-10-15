"""Offline unmarked-road fixtures and explicit, unqualified behavior tasks.

OpenDRIVE driving lanes are topology, not visible paint. Both directions share
one unmarked paved road; raised sidewalks give the renderer physical road edges.
The inferred divider is never claimed to be a rendered boundary. This module
does not connect to, reset, or tick a running simulator.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from lane_dataset.config import sha256_file, write_json
from lane_dataset.geometry import LaneReference
from lane_dataset.inventory import route_document, sample_route
from lane_dataset.procedural import ProceduralParameters, build_opendrive


CASES = ("clear_road", "parked_car", "roadside_pedestrian")
REQUIRED_CAPTURE_FEATURES = (
    "observed_physical_road_edges_with_inferred_rht_corridor",
    "visible_obstacle_footprints_and_motion_history",
    "obstacle_conditioned_target_with_right_corridor_recovery",
    "full_body_clearance_and_oncoming_space_checks",
    "independent_unmarked_behavior_admission",
)


@dataclass(frozen=True)
class UnmarkedRoadParameters:
    shape: str = "straight"
    lane_width_m: float = 4.5
    straight_length_m: float = 240.
    radius_m: float = 32.
    turn: str = "left"
    approach_m: float = 100.
    exit_m: float = 100.
    sidewalk_width_m: float = 1.5
    curb_height_m: float = .15
    split: str = "development"

    def validate(self):
        if self.shape not in ("straight", "constant_radius", "s_bend", "tightening", "hairpin"):
            raise ValueError("unsupported unmarked road shape")
        if self.turn not in ("left", "right"):
            raise ValueError("turn must be left or right")
        if self.split not in ("development", "train", "validation", "test"):
            raise ValueError("invalid split")
        for name, value in asdict(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"nonfinite road parameter: {name}")
        if not 3. <= self.lane_width_m <= 4.5:
            raise ValueError("each virtual lane must be 3 to 4.5 m wide")
        if not 180. <= self.straight_length_m <= 1000.:
            raise ValueError("straight length must budget approach, recovery, and preview")
        if self.approach_m < 80. or self.exit_m < 80.:
            raise ValueError("curve fixtures require at least 80 m approach and exit")
        if not 10. <= self.radius_m <= 80.:
            raise ValueError("radius outside supported generator range")
        if not .75 <= self.sidewalk_width_m <= 3. or not .05 <= self.curb_height_m <= .3:
            raise ValueError("invalid physical sidewalk/curb dimensions")

    def canonical_geometry(self):
        self.validate()
        data = asdict(self)
        # Obstacles, appearance, split, and mirrored traversal do not create
        # independent geometry. A radius/turn is irrelevant to a straight.
        for name in ("split", "turn"):
            data.pop(name)
        if self.shape == "straight":
            for name in ("radius_m", "approach_m", "exit_m"):
                data.pop(name)
        else:
            data.pop("straight_length_m")
        return {"generator": "unmarked_rht_curbed_v1", **data}

    @property
    def geometry_group(self):
        payload = json.dumps(self.canonical_geometry(), sort_keys=True, separators=(",", ":"))
        return "unmarked-" + hashlib.sha256(payload.encode()).hexdigest()[:20]


def build_unmarked_opendrive(parameters: UnmarkedRoadParameters) -> str:
    """Create two topology lanes with no painted divider or edge markings.

    A 15 cm curb is a requested physical mesh feature, not proof that CARLA
    renders it correctly. Its RGB/depth agreement still requires a capture.
    """
    p = parameters
    p.validate()
    base = ProceduralParameters(
        family="constant_radius" if p.shape == "straight" else p.shape,
        radius_m=p.radius_m, turn=p.turn, lane_width_m=p.lane_width_m,
        shoulder_width_m=p.sidewalk_width_m, approach_m=p.approach_m,
        exit_m=p.exit_m, split=p.split,
    )
    root = ET.fromstring(build_opendrive(base))
    root.find("header").set("name", p.geometry_group)
    road = root.find("road")
    road.set("name", p.geometry_group)
    road.set("rule", "RHT")
    if p.shape == "straight":
        road.set("length", str(p.straight_length_m))
        plan = road.find("planView")
        plan.clear()
        geometry = ET.SubElement(plan, "geometry", s="0", x="0", y="0", hdg="0",
                                 length=str(p.straight_length_m))
        ET.SubElement(geometry, "line")
    for lane in road.findall("lanes/laneSection/*/lane"):
        for mark in list(lane.findall("roadMark")):
            lane.remove(mark)
        ET.SubElement(lane, "roadMark", sOffset="0", type="none", width="0", color="white")
        if abs(int(lane.attrib["id"])) == 2:
            lane.set("type", "sidewalk")
            ET.SubElement(lane, "height", sOffset="0", inner=str(p.curb_height_m),
                          outer=str(p.curb_height_m))
    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def right_hand_corridor(road_left_xyz, road_right_xyz, arc_m=None):
    """Derive the virtual right half from *paired physical road edges*.

    CARLA traversal coordinates use +Y to the right. The goal is 25% of the
    road width inward from the right edge, not the midpoint of the whole road.
    This geometry function makes no visibility or training-admission claim.
    """
    road = LaneReference(road_left_xyz, road_right_xyz, arc_m)
    widths = np.linalg.norm(road.right_xyz-road.left_xyz, axis=1)
    if np.any(widths < 6.-1e-4) or np.any(widths > 12.+1e-4):
        raise ValueError("unmarked two-way road requires supported 6 to 12 m pavement width")
    divider = (road.left_xyz+road.right_xyz)*.5
    corridor = LaneReference(divider, road.right_xyz, road.s)
    return dict(road_left_edge_xyz=road.left_xyz.tolist(),
                road_right_edge_xyz=road.right_xyz.tolist(),
                inferred_divider_xyz=divider.tolist(),
                right_lane_target_xyz=corridor.center_xyz.tolist(),
                traffic_convention="RHT", target_fraction_from_right_edge=.25,
                divider_evidence="inferred_from_paired_physical_road_edges_not_paint",
                rendered_qualification=False)


def route_from_opendrive(road_map, parameters, *, lane_id=-1, step_m=.5):
    """Read road edges from CARLA's offline map parser, in traversal order."""
    if lane_id not in (-1, 1) or not math.isfinite(step_m) or step_m <= 0:
        raise ValueError("lane_id must be +/-1 and sampling step must be positive")
    xml = build_unmarked_opendrive(parameters)
    length = float(ET.fromstring(xml).find("road").attrib["length"])
    start_s = .01 if lane_id == -1 else length-.01
    start = road_map.get_waypoint_xodr(1, lane_id, start_s)
    if start is None:
        raise ValueError("unmarked map has no requested driving lane")
    points = sample_route(start, length-.1, step_m)
    if len(points) < 4:
        raise ValueError("unmarked map has insufficient traversable road")
    _, route = route_document(points, "Town04")
    left_edges, right_edges = [], []
    for waypoint in points:
        opposite = road_map.get_waypoint_xodr(waypoint.road_id, -lane_id, waypoint.s)
        if opposite is None:
            raise ValueError("both directions must exist at each paired road coordinate")
        def outer_edge(wp):
            location, right = wp.transform.location, wp.transform.get_right_vector()
            return [location.x+right.x*wp.lane_width/2,
                    location.y+right.y*wp.lane_width/2,
                    location.z+right.z*wp.lane_width/2]
        left_edges.append(outer_edge(opposite))
        right_edges.append(outer_edge(waypoint))
    geometry = right_hand_corridor(left_edges, right_edges, route["arc_m"])
    if not np.allclose(route["center_xyz"], geometry["right_lane_target_xyz"], atol=1e-4, rtol=0):
        raise ValueError("parsed driving-lane center disagrees with paired physical road edges")
    route["left_xyz"] = geometry["inferred_divider_xyz"]
    route["right_xyz"] = geometry["road_right_edge_xyz"]
    route["center_xyz"] = geometry["right_lane_target_xyz"]
    route["unmarked_road"] = geometry
    route["boundary_semantics"] = {"left": "inferred_divider", "right": "physical_road_edge"}
    route["observation_envelope_qualified"] = False
    route["marking_support_qualified"] = False
    route["source_geometry_scope"] = "offline_CARLA_map_parse_not_rendered_evidence"
    return route


def obstacle_case(case, route_length_m, lane_width_m):
    """Scenario intent only; actor footprint/motion/visibility must be measured.

    Pedestrian avoidance here is a stationary person at the road edge. A moving
    crossing pedestrian needs trajectory prediction/yielding and is not falsely
    represented as a stationary-obstacle example.
    """
    if case not in CASES:
        raise ValueError("unknown unmarked-road behavior case")
    if not math.isfinite(lane_width_m) or not 3. <= lane_width_m <= 4.5:
        raise ValueError("invalid unmarked-road virtual lane width")
    if not math.isfinite(route_length_m) or route_length_m < 170.:
        raise ValueError("route too short for approach, avoidance, recovery, and preview")
    result = dict(case=case, baseline="inferred_right_lane_center",
                  recovery_target="same_inferred_right_lane_center_after_obstacle_clear",
                  crossing_divider_allowed_only_for_observed_obstruction=False,
                  opposing_half_policy="whole_ego_body_stays_in_right_half_v1",
                  oncoming_conflict_response="yield_before_conflict_do_not_force_pass",
                  no_safe_corridor_response="yield_do_not_label_collision_path",
                  rendered_qualification=False, training_ready=False)
    if case == "clear_road":
        result.update(obstacles=[], planned_phase_ranges_m={"right_lane_follow": [35., route_length_m-30.]})
        return result
    # Constant arc coordinates are planning events, never label timestamps.
    actor_s = min(100., route_length_m-70.)
    pedestrian = case == "roadside_pedestrian"
    result["obstacles"] = [dict(
        kind="pedestrian" if pedestrian else "parked_vehicle",
        blueprint_filter="walker.pedestrian.*" if pedestrian else "vehicle.audi.a2",
        route_s_m=actor_s, offset_from_right_lane_center_m=lane_width_m/2-(.0 if pedestrian else .5),
        placement="right_edge_with_measured_inward_body_intrusion",
        maximum_road_intrusion_m=.2 if pedestrian else 1.3,
        motion="stationary", pedestrian_controller=False, teleport_after_spawn=False,
        placement_requires_actual_bounding_box=True,
        minimum_body_clearance_m=1.2 if pedestrian else .6,
        dimension_source="must_measure_spawned_actor_not_blueprint_guess",
    )]
    result["planned_phase_ranges_m"] = {
        "approach": [actor_s-40., actor_s-25.],
        "avoidance_entry": [actor_s-25., actor_s-10.],
        "passing": [actor_s-10., actor_s+10.],
        "recovery": [actor_s+10., actor_s+30.],
        "right_lane_follow": [actor_s+30., route_length_m-30.],
    }
    result["phase_scope"] = "planning_envelope_actual_observed_actor_geometry_controls_labels"
    return result


def create_candidate_plan(output_directory, parameters, *, cases=CASES, lane_id=-1, carla_module=None):
    """Write isolated candidates without changing the active queue or corpus.

    Only ``carla.Map`` is used. No Client, world loading, ticking, or live actor
    calls occur. Existing v8 capture must not consume these tasks unchanged.
    """
    parameters.validate()
    cases = tuple(cases)
    if not cases or len(set(cases)) != len(cases) or any(case not in CASES for case in cases):
        raise ValueError("candidate cases must be nonempty, unique, and recognized")
    if carla_module is None:
        import carla as carla_module
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    xml = build_unmarked_opendrive(parameters)
    variant = "straight" if parameters.shape == "straight" else parameters.turn
    path = output_directory / f"{parameters.geometry_group}-{variant}.xodr"
    path.write_text(xml, encoding="utf-8")
    parsed = carla_module.Map(parameters.geometry_group, xml)
    route = route_from_opendrive(parsed, parameters, lane_id=lane_id)
    family = {"straight": "straight_recovery", "constant_radius": "constant_radius",
              "s_bend": "tightening_s_bend", "tightening": "tightening_s_bend", "hairpin": "severe_hairpin"}[parameters.shape]
    route.update(source_xodr=str(path.resolve()), source_xodr_sha256=sha256_file(path), branch=0)
    route["section"].update(section_id=parameters.geometry_group, town="OpenDrive", family=family,
        procedural=True, procedural_parameters=parameters.canonical_geometry(), geometry_key=parameters.geometry_group)
    start = dict(route["start"])
    start["s"] += 10. if lane_id == -1 else -10.
    tasks = []
    for case in cases:
        behavior = obstacle_case(case, route["length_m"], parameters.lane_width_m)
        tasks.append(dict(
            episode_id=f"{parameters.geometry_group}-{variant}-{case}-lane{lane_id}",
            group_id=parameters.geometry_group, split=parameters.split, family=family,
            behavior_family="unmarked_rht_"+case, town="OpenDrive", procedural=True,
            opendrive_path=str(path.resolve()), route=route, start=start, branch=0,
            capture_start_route_m=10., appearance=0, seed=20261015, nominal_speed_mps=2.5,
            offset_m=0., heading_error_deg=0., planned_preview_m=30., warmup_distance_budget_m=20.,
            minimum_ticks=math.floor((route["length_m"]-60.)/(2.5*.05)),
            direction=route["section"]["direction"], unmarked_behavior=behavior,
            unmarked_road=True, capture_contract="unmarked_rht_obstacle_v1",
            required_capture_features=list(REQUIRED_CAPTURE_FEATURES),
            existing_capture_compatible=False, rendered_qualification=False,
            training_ready=False, qualification_blockers=[
                "unmarked_capture_contract_integration_required",
                "physical_road_edge_rgb_depth_qualification_required",
                "actual_obstacle_and_oncoming_clearance_qualification_required",
                "independent_obstacle_conditioned_behavior_admission_required",
            ],
        ))
    report = dict(schema_version=1, status="offline_candidate_only", active_queue_modified=False,
                  rendered_qualification=False, splits_frozen=False,
                  geometry_group=parameters.geometry_group, split=parameters.split,
                  road_parameters=asdict(parameters), source_xodr=str(path.resolve()),
                  source_xodr_sha256=sha256_file(path), task_count=len(tasks),
                  split_policy="all appearances, obstacle cases, and directions of this geometry share one split",
                  required_capture_features=list(REQUIRED_CAPTURE_FEATURES), tasks=tasks)
    # Normalize dataclass tuples from route_document into the same pure JSON
    # shape returned to callers and persisted for downstream consumers.
    report = json.loads(json.dumps(report, allow_nan=False))
    plan_path = output_directory/f"{parameters.geometry_group}-{variant}-lane{lane_id}-candidate-plan.json"
    write_json(plan_path, report)
    return report, plan_path


def create_native_candidate_plan(output_directory, source_xodr, road_id, *, lane_id=-1, carla_module=None,
                                 edge_refinement=None):
    """Plan one native, entirely unmarked, symmetric two-way road for review.

    Physical-edge candidates follow outer curbs, including any unmarked paved
    shoulders between the driving topology and sidewalk. They are not silently
    replaced by the driving-lane boundary. Native RGB/depth must resolve the
    actual asphalt/curb edge before these coordinates can support a label.
    """
    source_xodr = Path(source_xodr)
    town = source_xodr.stem
    if town not in ("Town01","Town02","Town03","Town04","Town06","Town07","Town10HD"):
        raise ValueError("native development probes exclude test-only Town05")
    if lane_id not in (-1,1):
        raise ValueError("native road traversal must use lane +/-1")
    xml = source_xodr.read_text(encoding="utf-8")
    refinement_binding=None
    if edge_refinement is not None:
        refinement=json.loads(Path(edge_refinement).read_text(encoding="utf-8"))
        if (refinement.get('proposed_edge_policy')!='inner_boundary_of_curb_shoulder'
                or refinement.get('source_xodr_sha256')!=sha256_file(source_xodr)
                or refinement.get('town')!=town or str(refinement.get('road_id'))!=str(road_id)):
            raise ValueError('edge refinement does not bind this native map and road')
        refinement_binding=dict(path=str(Path(edge_refinement).resolve()),sha256=sha256_file(edge_refinement),
            policy='inner_boundary_of_curb_shoulder',scope='candidate_reference_selection_not_paired_rendered_qualification')
    matches = [r for r in ET.fromstring(xml).findall("road") if r.get("id") == str(road_id)]
    if len(matches) != 1 or matches[0].get("junction") != "-1":
        raise ValueError("one non-junction native road required")
    road = matches[0]
    sections = road.findall("lanes/laneSection")
    if len(sections) != 1 or float(sections[0].get("s","0")) != 0.:
        raise ValueError("native initial probe requires one unchanged lane section")
    section = sections[0]
    edge_lane_ids = {}
    topology = {}
    for side,sign in (("left",1),("right",-1)):
        lanes = sorted(section.findall(side+"/lane"),key=lambda lane:abs(int(lane.get("id"))))
        driving = [lane for lane in lanes if lane.get("type") == "driving"]
        if len(driving) != 1 or int(driving[0].get("id")) != sign:
            raise ValueError("native unmarked probe requires exactly one driving lane per direction")
        edge_lane = None
        topology[side] = []
        for lane in lanes:
            marks = [m.get("type","none") for m in lane.findall("roadMark")]
            topology[side].append(dict(lane_id=int(lane.get("id")),type=lane.get("type"),marks=marks))
            if lane.get("type") == "sidewalk":
                break
            if lane.get("type") not in ("driving","shoulder") or any(mark not in ("none","curb") for mark in marks):
                raise ValueError("native road contains painted, parking, or unsupported side lanes")
            if "curb" in marks:
                if any(mark != "curb" for mark in marks):
                    raise ValueError("native curb boundary changes within the road")
                edge_lane = int(lane.get("id"))
        if edge_lane is None:
            raise ValueError("native physical outer curb is not explicitly mapped")
        edge_lane_ids[side] = edge_lane
    center = section.find("center/lane")
    if center is None or any(mark.get("type","none") != "none" for mark in center.findall("roadMark")):
        raise ValueError("native road has painted centerline")
    if carla_module is None:
        import carla as carla_module
    parsed = carla_module.Map(town,xml)
    length = float(road.get("length"))
    if length < 180.:
        raise ValueError("native road too short for complete approach, avoidance, recovery and preview")
    query = np.arange(.01,length-.01,.5)
    if lane_id == 1:
        query = query[::-1]
    points = [parsed.get_waypoint_xodr(int(road_id),lane_id,float(s)) for s in query]
    if any(point is None for point in points):
        raise ValueError("native driving topology has a hole")
    _,route = route_document(points,town)
    road_left,road_right = [],[]
    for s in query:
        def outer(which):
            wp = parsed.get_waypoint_xodr(int(road_id),edge_lane_ids[which],float(s))
            if wp is None:
                raise ValueError("native outer curb topology has a hole")
            p,r = wp.transform.location,wp.transform.get_right_vector()
            fraction=-.5 if refinement_binding is not None else .5
            return [p.x+r.x*wp.lane_width*fraction,p.y+r.y*wp.lane_width*fraction,p.z+r.z*wp.lane_width*fraction]
        left,right = outer("left"),outer("right")
        road_left.append(left if lane_id == -1 else right)
        road_right.append(right if lane_id == -1 else left)
    geometry = right_hand_corridor(road_left,road_right,route["arc_m"])
    route["driving_topology_center_xyz"] = route["center_xyz"]
    route["left_xyz"] = geometry["inferred_divider_xyz"]
    route["right_xyz"] = geometry["road_right_edge_xyz"]
    route["center_xyz"] = geometry["right_lane_target_xyz"]
    route["unmarked_road"] = geometry
    route["boundary_semantics"] = {"left":"inferred_divider","right":"mapped_outer_curb_pending_rendered_qualification"}
    route["source_geometry_scope"] = "offline_native_CARLA_map_parse_not_rendered_evidence"
    route["native_lane_topology"] = topology
    route["mapped_outer_curb_lane_ids"] = edge_lane_ids
    route["mapped_edge_definition"] = 'inner_boundary_of_curb_shoulder' if refinement_binding else 'outer_boundary_of_curb_shoulder'
    if refinement_binding:
        route["edge_reference_selection"] = refinement_binding
    route["source_xodr"] = str(source_xodr.resolve())
    route["source_xodr_sha256"] = sha256_file(source_xodr)
    route["observation_envelope_qualified"] = False
    identity = dict(town=town,road_id=int(road_id),source_sha256=route["source_xodr_sha256"],interval_m=[0.,length])
    group = "unmarked-native-"+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    tasks=[]
    effective_lane_width = float(np.median(np.linalg.norm(np.array(road_right)-road_left,axis=1))*.5)
    # Existing intention builder limits its generic lane grid to4.5m. Actual
    # measured native width is retained separately; no actor dimensions inferred.
    for case in CASES:
        behavior = obstacle_case(case,route["length_m"],min(effective_lane_width,4.5))
        if behavior["obstacles"]:
            behavior["obstacles"][0].update(maximum_road_intrusion_m=0. if case=="roadside_pedestrian" else .8,
                placement="measured_bbox_at_native_physical_road_edge_collision_ground_height",
                placement_requires_actual_ego_footprint_feasibility=True,
                required_behavior="nonzero_avoidance_and_complete_recovery_inside_right_half")
        tasks.append(dict(episode_id=f"{group}-{case}-lane{lane_id}",group_id=group,split="development",
            family=route["section"]["family"],behavior_family="unmarked_rht_"+case,town=town,
            procedural=False,native_unmarked=True,opendrive_path=str(source_xodr.resolve()),route=route,
            start=route["start"],branch=0,capture_start_route_m=10.,appearance=0,seed=20261015,
            nominal_speed_mps=2.5,offset_m=0.,heading_error_deg=0.,planned_preview_m=30.,warmup_distance_budget_m=20.,
            minimum_ticks=math.floor((route["length_m"]-60.)/.125),direction=route["section"]["direction"],
            unmarked_behavior=behavior,unmarked_road=True,capture_contract="unmarked_rht_obstacle_v1",
            required_capture_features=list(REQUIRED_CAPTURE_FEATURES),existing_capture_compatible=False,
            rendered_qualification=False,training_ready=False,qualification_blockers=[
                "native_outer_curb_rgb_depth_qualification_required","observed_road_width_and_target_qualification_required",
                "physical_avoidance_recovery_capture_required","existing_geometry_split_overlap_audit_required"]))
    report=dict(schema_version=1,status="offline_candidate_only",active_queue_modified=False,
        rendered_qualification=False,splits_frozen=False,geometry_group=group,split="development",
        source_kind="native_map",town=town,road_id=int(road_id),source_xodr=str(source_xodr.resolve()),
        source_xodr_sha256=route["source_xodr_sha256"],mapped_outer_curb_width_m=2*effective_lane_width,
        driving_topology_lane_width_m=float(np.median([p.lane_width for p in points])),task_count=len(tasks),tasks=tasks)
    if refinement_binding:
        report["edge_reference_selection"] = refinement_binding
    report=json.loads(json.dumps(report,allow_nan=False))
    output_directory=Path(output_directory)
    # Keep prior probe-plan hashes replayable when obstruction intent changes.
    variant=('-inner-curb' if refinement_binding else '')+'-obstruction-v2'
    path=output_directory/f"{group}-{town}-road{road_id}-lane{lane_id}{variant}-candidate-plan.json"
    write_json(path,report)
    return report,path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", choices=("straight", "constant_radius", "s_bend", "tightening", "hairpin"), default="straight")
    parser.add_argument("--split", choices=("development", "train", "validation", "test"), default="development")
    parser.add_argument("--lane-width", type=float, default=4.5)
    parser.add_argument("--radius", type=float, default=32.)
    parser.add_argument("--lane-id", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--native-map",type=Path)
    parser.add_argument("--native-road",type=int)
    parser.add_argument("--edge-refinement",type=Path)
    args = parser.parse_args()
    if args.native_map is not None:
        if args.native_road is None:
            parser.error("--native-map requires --native-road")
        report,path=create_native_candidate_plan(args.output,args.native_map,args.native_road,lane_id=args.lane_id,
            edge_refinement=args.edge_refinement)
    else:
        if args.native_road is not None:
            parser.error("--native-road requires --native-map")
        report, path = create_candidate_plan(args.output,
            UnmarkedRoadParameters(shape=args.shape, split=args.split, lane_width_m=args.lane_width, radius_m=args.radius),
            lane_id=args.lane_id)
    print(json.dumps(dict(status=report["status"], task_count=report["task_count"], path=str(path),
                         active_queue_modified=False, rendered_qualification=False), indent=2))


if __name__ == "__main__":
    main()
