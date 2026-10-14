"""Offline independent route selection before geometry split grouping.

An exhaustive overlapping route catalogue is not a corpus: transitive overlap
can join an entire town. Select non-overlapping complete observation envelopes
first, then recompute grouping and freeze only after rendered qualification.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .config import DEFAULT_ROOT, Config, sha256_file, write_json
from .inventory import route_document, sample_route
from .procedural import ProceduralParameters, build_opendrive
from .maneuvers import propose_maneuver
from .splits import (APPROVED_TOWNS, FAMILIES, GeometrySection, RoadInterval, SplitAssignment,
                     audit_split_overlap, group_geometry_sections)


@dataclass(frozen=True)
class PlannerConfig:
    start_spacing_m: float = 10.
    route_envelope_m: float = 90.
    sample_step_m: float = 1.
    nominal_speed_mps: float = 2.5
    minimum_episode_s: float = 24.5
    planned_preview_m: float = 25.
    warmup_distance_budget_m: float | None = None
    overlap_margin_m: float = 5.
    seed: int = 20261013
    attempts: int = 12
    towns: tuple[str, ...] = APPROVED_TOWNS

    def validate(self):
        if (not self.towns or len(set(self.towns)) != len(self.towns)
                or not set(self.towns).issubset(APPROVED_TOWNS) or "Town05" not in self.towns):
            raise ValueError("planner requires unique approved maps including test-only Town05")
        warmup=0. if self.warmup_distance_budget_m is None else self.warmup_distance_budget_m
        if not math.isfinite(warmup) or warmup<0:
            raise ValueError("warmup distance budget must be nonnegative when supplied")
        if self.route_envelope_m < warmup+self.nominal_speed_mps*self.minimum_episode_s+self.planned_preview_m:
            raise ValueError("route envelope omits dense prefix/labels/suffix travel or planned preview")
        if self.start_spacing_m <= 0 or self.sample_step_m <= 0 or self.attempts < 1:
            raise ValueError("positive sampling distances and search attempts required")


def recording_ticks(route_length_m: float, config: PlannerConfig, *, procedural=False, capture_start_m=0.) -> int:
    """Budget recording separately from unrecorded warmup and forward preview.

    A structural plan with an unknown warmup budget stays explicitly unqualified.
    Supplying a budget enables capture planning and rejects truncated envelopes.
    """
    if not math.isfinite(route_length_m) or route_length_m <= 0:
        raise ValueError("route length must be finite and positive")
    if not math.isfinite(capture_start_m) or not 0<=capture_start_m<route_length_m:
        raise ValueError("capture start must lie within the route envelope")
    minimum = max(500, math.ceil(config.minimum_episode_s / Config().dt))
    if config.warmup_distance_budget_m is None:
        return minimum
    available = route_length_m - capture_start_m - config.warmup_distance_budget_m - config.planned_preview_m
    capacity = math.floor(available / (config.nominal_speed_mps * Config().dt))
    if capacity < minimum:
        raise ValueError("route envelope cannot contain warmup, recording, and preview")
    # Procedural curves have a long explicit exit: retain the complete bend and
    # its suffix while leaving the reserved preview beyond the last frame.
    return capacity if procedural else minimum


def direct_junction_links(opendrive: str) -> tuple[tuple[str,str], ...]:
    """Group directly joined junction roads and shared connection roads.

    Ordinary non-junction roads between distinct intersections remain interval
    overlap dependencies. Grouping *all reachable* intersections would make
    every connected town one geometry group, independent of recorded overlap.
    """
    root = ET.fromstring(opendrive)
    road_junction = {road.attrib["id"]:road.attrib.get("junction","-1") for road in root.findall("road")}
    links = set()
    owners = defaultdict(set)
    for junction in root.findall("junction"):
        for connection in junction.findall("connection"):
            owners[connection.attrib.get("connectingRoad")].add(junction.attrib["id"])
    for memberships in owners.values():
        values = sorted(memberships)
        links.update((values[0],other) for other in values[1:])
    for road in root.findall("road"):
        own = road_junction[road.attrib["id"]]
        if own == "-1":
            continue
        for link in road.findall("link/*"):
            other = (link.attrib.get("elementId") if link.attrib.get("elementType") == "junction"
                     else road_junction.get(link.attrib.get("elementId"),"-1"))
            if other not in (None,"-1",own):
                links.add(tuple(sorted((own,other))))
    return tuple(sorted(links))


def candidate_conflicts(sections: list[GeometrySection], margin_m=5.) -> list[set[int]]:
    """Build overlap dependencies without first merging unselected candidates."""
    graph = [set() for _ in sections]
    junction_parent = {}
    def find(key):
        junction_parent.setdefault(key,key)
        if junction_parent[key] != key:
            junction_parent[key] = find(junction_parent[key])
        return junction_parent[key]
    for section in sections:
        for left,right in section.junction_links:
            one,two = find((section.town,left)),find((section.town,right))
            junction_parent[one] = two
    roads = defaultdict(list)
    junctions = defaultdict(set)
    for index,section in enumerate(sections):
        for interval in section.road_intervals:
            roads[(section.town,interval.road_id)].append((interval.start_m,interval.end_m,index))
        for junction in section.junction_ids:
            junctions[find((section.town,junction))].add(index)
    for values in roads.values():
        active = []
        for start,end,index in sorted(values):
            active = [(old_end,old_index) for old_end,old_index in active if old_end+margin_m>=start]
            for _,old_index in active:
                if old_index != index:
                    graph[index].add(old_index);graph[old_index].add(index)
            active.append((end,index))
    for indices in junctions.values():
        for index in indices:
            graph[index].update(indices-{index})
    return graph


def choose_independent(sections: list[GeometrySection], graph: list[set[int]],
                       targets: dict[str,int], *, seed=0, attempts=12) -> list[int]:
    """Seeded bounded independent-set search, retaining real measured families.

    This is a heuristic, not a proof of maximum capacity. The report explicitly
    distinguishes observed search deficits from mathematical impossibility.
    """
    if len(graph) != len(sections):
        raise ValueError("one conflict set per section required")
    best=[];best_score=None
    for attempt in range(attempts):
        rng=np.random.default_rng(seed+attempt)
        noise=rng.random(len(sections))
        selected=[];blocked=set();counts=Counter();directions=Counter()
        while True:
            by_family={family:[i for i,section in enumerate(sections)
                if i not in blocked and section.family==family and counts[family]<targets.get(family,0)
                and (family=="straight_recovery" or section.direction not in ("left","right")
                     or directions[(family,section.direction)]<math.ceil(targets[family]/2))]
                for family in FAMILIES}
            available=[family for family,indices in by_family.items() if indices]
            if not available: break
            # Intersection/straight families cannot be manufactured by the
            # procedural curve fallback. Reserve their scarce support first.
            strategy=attempt%3
            def family_priority(f):
                scarcity=len(by_family[f])/max(1,targets[f]-counts[f])
                if strategy==0:
                    return (0 if f in ("intersection_turn","straight_recovery") else 1,scarcity,f)
                if strategy==1:
                    return (0 if f in ("severe_hairpin","tightening_s_bend") else 1,scarcity,f)
                return (0,scarcity,f)
            family=min(available,key=family_priority)
            def cost(index):
                section=sections[index]
                remaining_neighbors=sum(max(0,targets.get(sections[other].family,0)-counts[sections[other].family])/
                    max(1,len(by_family[sections[other].family])) for other in graph[index] if other not in blocked
                    and counts[sections[other].family]<targets.get(sections[other].family,0))
                imbalance=directions[(family,section.direction)]
                # Repeated attempts vary low-conflict alternatives, preserving
                # the fixed family label and lane/road identity throughout.
                return remaining_neighbors*(.8+.4*noise[index])+imbalance*.1
            index=min(by_family[family],key=lambda i:(cost(i),sections[i].section_id))
            selected.append(index);counts[family]+=1
            directions[(family,sections[index].direction)]+=1
            blocked.add(index);blocked.update(graph[index])
        deficits={family:max(0,targets[family]-counts[family]) for family in FAMILIES}
        score=(sum(deficits[f] for f in ("intersection_turn","straight_recovery")),
               sum(deficits.values()),-sum(len(sections[i].road_intervals)==1 for i in selected))
        if best_score is None or score<best_score:
            best,best_score=selected,score
    return sorted(best,key=lambda i:(sections[i].family,sections[i].section_id))


def optimize_native_selection(sections,graph,targets,heuristic,*,time_limit_s=40.):
    """Bounded binary optimization with the same hard overlap/family rules."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    row=[];column=[];values=[];lower=[];upper=[]
    for index,neighbors in enumerate(graph):
        for other in neighbors:
            if other<=index: continue
            number=len(lower)
            row.extend((number,number));column.extend((index,other));values.extend((1.,1.))
            lower.append(-np.inf);upper.append(1.)
    for family in FAMILIES:
        number=len(lower)
        for index,section in enumerate(sections):
            if section.family==family:
                row.append(number);column.append(index);values.append(1.)
        lower.append(float(targets[family]) if family in ("straight_recovery","intersection_turn") else 0.)
        upper.append(float(targets[family]))
        if family!="straight_recovery":
            for direction in ("left","right"):
                number=len(lower)
                for index,section in enumerate(sections):
                    if section.family==family and section.direction==direction:
                        row.append(number);column.append(index);values.append(1.)
                lower.append(0.);upper.append(float(math.ceil(targets[family]/2)))
    matrix=coo_matrix((values,(row,column)),shape=(len(lower),len(sections))).tocsc()
    result=milp(-np.ones(len(sections)),integrality=np.ones(len(sections)),bounds=Bounds(0.,1.),
        constraints=LinearConstraint(matrix,np.array(lower),np.array(upper)),
        options={"time_limit":time_limit_s,"mip_rel_gap":.001,"presolve":True})
    chosen=[] if result.x is None else np.flatnonzero(result.x>.5).tolist()
    # Never accept a fractional/time-limit artifact or regress a known solution.
    valid=bool(chosen) and all(other not in graph[index] for position,index in enumerate(chosen) for other in chosen[position+1:])
    counts=Counter(sections[index].family for index in chosen)
    valid &= all(counts[family]<=targets[family] for family in FAMILIES)
    valid &= all(counts[family]==targets[family] for family in ("straight_recovery","intersection_turn"))
    valid &= all(sum(sections[index].family==family and sections[index].direction==direction for index in chosen)
                 <=math.ceil(targets[family]/2) for family in FAMILIES if family!="straight_recovery"
                 for direction in ("left","right"))
    if not valid or len(chosen)<len(heuristic): chosen=heuristic
    return chosen,{"status":int(result.status),"message":result.message,
        "selected_native":len(chosen),"heuristic_native":len(heuristic),"time_limit_s":time_limit_s,
        "mip_gap":None if getattr(result,"mip_gap",None) is None else float(result.mip_gap),
        "proven_optimal":bool(result.status==0)}


def _procedural_parameters(split,family,index):
    shape="hairpin" if family=="severe_hairpin" else "tightening" if index%2 else "s_bend"
    if family=="constant_radius": shape="constant_radius"
    # Split-separated radius lattices, including 20 distinct train hairpins.
    offset={"train":0.,"validation":.75,"test":1.5}[split]
    params=ProceduralParameters(shape,11.+3.*index+offset,
        turn="left" if index%2==0 else "right",lane_width_m=(3.3,3.6,3.9)[index%3],
        grade=(0.,.025,-.025)[index%3],bank_rad=(0.,.04,-.04)[(index//3)%3],split=split)
    params.validate()
    return params


def plan_scenarios(dataset_root: str | Path, config: PlannerConfig=PlannerConfig()) -> dict:
    """Inspect installed CARLA OpenDRIVE through its actual offline Map parser."""
    import carla
    config.validate()
    root=Path(dataset_root)
    source=Path(Config().carla_root)/"CarlaUE4/Content/Carla/Maps/OpenDrive"
    sections=[];routes={};sources={};seen=set();rejected=Counter()
    for town in config.towns:
        path=source/f"{town}.xodr";xml=path.read_text();road_map=carla.Map(town,xml)
        sources[town]={"path":str(path),"sha256":sha256_file(path)}
        links=direct_junction_links(xml)
        starts=sorted(road_map.generate_waypoints(config.start_spacing_m),key=lambda w:(w.road_id,w.lane_id,w.s))
        for start in starts:
            if start.lane_type != carla.LaneType.Driving:
                rejected["non_driving_lane"]+=1;continue
            for branch in range(3):
                points=sample_route(start,config.route_envelope_m,config.sample_step_m,branch)
                if len(points)<int(config.route_envelope_m/config.sample_step_m):
                    rejected["short_envelope"]+=1;continue
                section,route=route_document(points,town)
                if route["length_m"]<config.route_envelope_m-1.:
                    rejected["short_measured_envelope"]+=1;continue
                # Hash the traversed geometry, not only the first/last roads.
                identity=hashlib.sha256((town+repr(np.round(np.array(route["center_xyz"]),2).tolist())).encode()).hexdigest()[:20]
                if identity in seen:
                    rejected["duplicate_route"]+=1;continue
                seen.add(identity)
                direction=section.direction
                if section.family!="straight_recovery" and direction=="straight":
                    # Small total heading change is not a straight direction
                    # for an S-bend. Attribute its first measurable bend.
                    active=[v for v in route["curvature_1pm"] if abs(v)>.002]
                    if active: direction="right" if active[0]>0 else "left"
                section=replace(section,section_id=identity,junction_links=links,direction=direction)
                route.update(branch=branch,section=asdict(section),nominal_speed_mps=config.nominal_speed_mps,
                             observation_envelope_qualified=False,planned_preview_m=config.planned_preview_m)
                sections.append(section);routes[identity]=route
        print(f"planner {town}: {sum(s.town==town for s in sections)} unique route candidates",flush=True)
    chosen=[];selection_reports={}
    for pool,towns,target in (("train_validation",set(config.towns)-{"Town05"},25),("test",{"Town05"},5)):
        candidates=[section for section in sections if section.town in towns]
        graph=candidate_conflicts(candidates,config.overlap_margin_m)
        indices=choose_independent(candidates,graph,{family:target for family in FAMILIES},seed=config.seed,attempts=config.attempts)
        indices,optimization=optimize_native_selection(candidates,graph,{family:target for family in FAMILIES},indices)
        values=[candidates[i] for i in indices];chosen.extend(values)
        counts=Counter(section.family for section in values)
        selection_reports[pool]={"selected_native":dict(counts),"targets_per_family":target,
            "native_deficits":{family:max(0,target-counts[family]) for family in FAMILIES},
            "candidate_count":len(candidates),"selected_count":len(values),
            "optimization":optimization,
            "search_method":"seeded independent-set initialization then bounded binary packing; optimum applies only to this sampled catalogue"}
    # Allocate already-disjoint native geometry across train/validation. Keep
    # holdout procedural use near its proportional share before applying caps.
    native_by_split={split:[] for split in ("train","validation","test")}
    for family in FAMILIES:
        pool=[section for section in chosen if section.family==family and section.town!="Town05"]
        pool.sort(key=lambda section:(section.direction,section.section_id))
        # Interleave left/right candidates so prefix slicing does not segregate
        # direction by split. No reverse traversal is a separate group.
        by_direction={direction:[section for section in pool if section.direction==direction]
                      for direction in ("left","right","straight")}
        interleaved=[]
        while any(by_direction.values()):
            for direction in by_direction:
                if by_direction[direction]: interleaved.append(by_direction[direction].pop(0))
        val_count=min(5,max(0,round(len(pool)/5)))
        if family=="straight_recovery":
            val_values=interleaved[:val_count]
        else:
            left=[section for section in pool if section.direction=="left"]
            right=[section for section in pool if section.direction=="right"]
            options=[(nl,nr) for nl in range(min(3,len(left))+1) for nr in range(min(3,len(right))+1)
                     if nl+nr<=5 and len(left)-nl<=10 and len(right)-nr<=10]
            if not options: raise RuntimeError("direction-constrained native solution cannot be split")
            nl,nr=min(options,key=lambda value:(abs(sum(value)-val_count),abs(value[0]-value[1]),-value[0]))
            val_values=left[:nl]+right[:nr]
        val_ids={section.section_id for section in val_values}
        native_by_split["validation"].extend(val_values)
        native_by_split["train"].extend(section for section in interleaved if section.section_id not in val_ids)
        native_by_split["test"].extend(section for section in chosen if section.family==family and section.town=="Town05")
    planned=[];assignments=[];tasks=[];deficits=[];procedural_counts={};generated_paths=[]
    for split,per_family,cap in (("train",20,30),("validation",5,7),("test",5,7)):
        native=native_by_split[split];counts=Counter(section.family for section in native)
        procedural_needed={family:max(0,per_family-counts[family]) for family in FAMILIES}
        unsupported=[family for family in ("straight_recovery","intersection_turn") if procedural_needed[family]]
        for family in unsupported:
            deficits.append(f"native_only_family_deficit:{split}/{family}:{procedural_needed[family]}")
        needed=sum(number for family,number in procedural_needed.items() if family not in unsupported)
        if needed>cap:
            deficits.append(f"procedural_cap_exceeded_if_filled:{split}:{needed}/{cap}")
        used=0
        selected=[(section,routes[section.section_id],None) for section in native]
        # Fill hardest shape first, preserving the cap even in an incomplete plan.
        for family in ("severe_hairpin","tightening_s_bend","constant_radius"):
            native_directions=Counter(section.direction for section in native if section.family==family)
            left_target=math.ceil(per_family/2)
            if native_directions["right"]>math.floor(per_family/2): left_target=math.floor(per_family/2)
            missing_directions={"left":left_target-native_directions["left"],
                                "right":per_family-left_target-native_directions["right"]}
            for index in range(procedural_needed[family]):
                if used>=cap: break
                params=_procedural_parameters(split,family,index)
                direction=max(missing_directions,key=lambda key:(missing_directions[key],key))
                params=replace(params,turn=direction)
                missing_directions[direction]-=1
                xml=build_opendrive(params);filename=f"{params.geometry_group}.xodr"
                output=root/"procedural"/split/filename
                output.parent.mkdir(parents=True,exist_ok=True);output.write_text(xml,encoding="utf-8")
                # Parse the actual generated road and retain CARLA's signed axes.
                road_map=carla.Map(params.geometry_group,xml)
                start=road_map.get_waypoint_xodr(1,-1,.01)
                total=float(ET.fromstring(xml).find("road").attrib["length"])
                points=sample_route(start,total-1.,config.sample_step_m,0)
                _,route=route_document(points,"Town03")
                parameters=asdict(params);parameters.pop("split");parameters.pop("turn")
                section=GeometrySection(params.geometry_group,"OpenDrive",family,procedural=True,
                    procedural_parameters=parameters,geometry_key=params.geometry_group,direction=params.turn)
                route.update(section=asdict(section),branch=0,nominal_speed_mps=config.nominal_speed_mps,
                    source_xodr=str(output),source_xodr_sha256=sha256_file(output),
                    observation_envelope_qualified=False,marking_support_qualified=False)
                selected.append((section,route,str(output)));used+=1;generated_paths.append(str(output))
        procedural_counts[split]=used
        for section,route,opendrive_path in selected:
            try:
                ticks=recording_ticks(route["length_m"],config,procedural=section.procedural,
                                      capture_start_m=10. if section.procedural else 0.)
            except ValueError as error:
                deficits.append(f"recording_envelope:{split}/{section.section_id}:{error}")
                continue
            group=group_geometry_sections([section],overlap_margin_m=config.overlap_margin_m)[0]
            planned.append(group)
            episode_seed=int(hashlib.sha256(f"{config.seed}:{group.group_id}:{split}".encode()).hexdigest()[:8],16)
            assignment=SplitAssignment(group.group_id,split,section.family,2 if split=="train" else 1,episode_seed)
            assignments.append(assignment)
            route["section"]=asdict(section)
            for appearance in range(assignment.appearances):
                task={"episode_id":f"{group.group_id}-{appearance}","group_id":group.group_id,
                    "split":split,"family":section.family,"town":section.town,"start":route["start"],
                    "branch":route["branch"],"route":route,"appearance":appearance,
                    "direction":section.direction,"seed":episode_seed+appearance,
                    "nominal_speed_mps":config.nominal_speed_mps,"procedural":section.procedural,
                    "minimum_ticks":ticks,
                    "warmup_distance_budget_m":config.warmup_distance_budget_m,
                    "planned_preview_m":config.planned_preview_m,
                    "rendered_qualification":False}
                if opendrive_path: task["opendrive_path"]=opendrive_path
                if section.procedural:
                    # Keep the complete vehicle on generated road mesh when it
                    # settles. Starting at s=.01 leaves its rear wheels in air.
                    task["start"]={**route["start"],"s":route["start"]["s"]+10.}
                    task["capture_start_route_m"]=10.
                task.update(propose_maneuver(route,section.family))
                tasks.append(task)
        actual=Counter(section.family for section,_,_ in selected)
        for family in FAMILIES:
            if actual[family]!=per_family:
                deficits.append(f"planned_group_quota:{split}/{family}:{actual[family]}/{per_family}")
            if family!="straight_recovery":
                direction_counts=Counter(section.direction for section,_,_ in selected if section.family==family)
                if abs(direction_counts["left"]-direction_counts["right"])>(0 if split=="train" else 1):
                    deficits.append(f"planned_direction_balance:{split}/{family}:{dict(direction_counts)}")
    overlap=audit_split_overlap(planned,assignments,overlap_margin_m=config.overlap_margin_m)
    # Independent selection must also preserve distinct groups WITHIN a split.
    regrouped=group_geometry_sections([section for group in planned for section in group.sections],overlap_margin_m=config.overlap_margin_m)
    if len(regrouped)!=len(planned): deficits.append(f"selected_group_overlap:{len(regrouped)}/{len(planned)}")
    deficits.extend(overlap)
    report={"schema_version":1,"status":"candidate_plan_complete" if not deficits else "candidate_plan_incomplete",
        "status_scope":"structural_candidate_selection_only",
        "rendered_qualification":False,"splits_frozen":False,"planner":asdict(config),"sources":sources,
        "observation_envelope_qualified":False,
        "warmup_distance_budget_status":"unknown_not_budgeted" if config.warmup_distance_budget_m is None else "declared_not_measured",
        "qualification_blockers":["physical_warmup_distance_must_be_measured_and_included_in_route_envelope",
            "actual_rendered_preview_and_other_visible_roads_require_cross_split_leakage_audit"],
        "candidate_count":len(sections),"candidate_families":dict(Counter(s.family for s in sections)),
        "candidate_families_by_town":{town:dict(Counter(s.family for s in sections if s.town==town)) for town in sources},
        "candidate_directions_by_family":{family:dict(Counter(s.direction for s in sections if s.family==family)) for family in FAMILIES},
        "selection":selection_reports,"selected_geometry_groups":len(planned),"episode_tasks":len(tasks),
        "procedural_groups_by_split":procedural_counts,"deficits":deficits,"rejections":dict(rejected),
        "groups":[asdict(group) for group in planned],"assignments":[asdict(item) for item in assignments],
        "grouping_policy":"complete planned route envelopes and 5m margin; shared or directly linked junction-road objects; reverse lanes overlap by road intervals",
        "unqualified_assumptions":["planned 25m useful preview must bound actually recorded model-visible lane evidence",
            "physical rolling warmup occurs before retained recording and its additional road travel is not budgeted when warmup_distance_budget_m is null",
            "road family classification requires rendered maneuver review","procedural marking continuity and physical clearance unqualified",
            "nominal speeds and episode durations require controller calibration and per-episode quality checks"],
        "native_direction_counts":{split:dict(Counter(section.direction for section in native_by_split[split])) for split in native_by_split}}
    write_json(root/"reports/scenario-plan.json",report)
    write_json(root/"reports/scenario-tasks.json",tasks)
    return report


if __name__=="__main__":
    import argparse
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--envelope",type=float,default=90.)
    parser.add_argument("--attempts",type=int,default=12)
    parser.add_argument("--start-spacing",type=float,default=10.)
    args=parser.parse_args()
    report=plan_scenarios(args.root,PlannerConfig(route_envelope_m=args.envelope,attempts=args.attempts,
                                               start_spacing_m=args.start_spacing))
    print(json.dumps({key:report[key] for key in ("status","candidate_count","candidate_families_by_town",
        "selection","selected_geometry_groups","episode_tasks","procedural_groups_by_split","deficits")},indent=2))
