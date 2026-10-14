"""Offline native-map route inventory, with full traversed road envelopes."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .config import Config, sha256_file, write_json
from .splits import APPROVED_TOWNS, GeometrySection, RoadInterval, group_geometry_sections, assign_splits


def sample_route(start, length_m=160., step_m=1., branch=0):
    points = [start]
    for _ in range(int(length_m/step_m)):
        choices = sorted(points[-1].next(step_m),key=lambda w:(w.road_id,w.section_id,w.lane_id,w.s))
        if not choices:
            break
        # Explicit deterministic topology choice. Route desire is derived later.
        selected = choices[branch % len(choices)]
        if selected.transform.location.distance(points[-1].transform.location) < .1:
            break
        points.append(selected)
    return points


def route_document(points, town):
    centers, left, right = [],[],[]
    roads=defaultdict(list); junctions=set()
    for wp in points:
        p=wp.transform.location;r=wp.transform.get_right_vector(); half=wp.lane_width/2
        centers.append([p.x,p.y,p.z]);left.append([p.x-half*r.x,p.y-half*r.y,p.z-half*r.z]);right.append([p.x+half*r.x,p.y+half*r.y,p.z+half*r.z])
        roads[str(wp.road_id)].append(wp.s)
        if wp.is_junction:
            junctions.add(str(wp.junction_id))
    xyz=np.asarray(centers); distances=np.linalg.norm(np.diff(xyz,axis=0),axis=1)
    arc=np.r_[0,np.cumsum(distances)]
    yaw=np.unwrap(np.arctan2(np.gradient(xyz[:,1]),np.gradient(xyz[:,0])))
    curvature=np.gradient(yaw,np.maximum.accumulate(arc+np.arange(len(arc))*1e-10))
    delta=float(yaw[-1]-yaw[0]);peak=float(np.quantile(np.abs(curvature),.95))
    active=curvature[np.abs(curvature)>.005]
    sign_changes=np.any(active>0) and np.any(active<0)
    if junctions and abs(delta)>math.radians(20):
        family="intersection_turn"
    elif peak>.075 or abs(delta)>math.radians(100):
        family="severe_hairpin"
    elif sign_changes or (len(active)>10 and np.std(np.abs(active))>.008):
        family="tightening_s_bend"
    elif peak>.004:
        family="constant_radius"
    else:
        family="straight_recovery"
    direction="straight" if abs(delta)<math.radians(5) else ("right" if delta>0 else "left")
    start=points[0]
    identity=f"{town}-{start.road_id}-{start.lane_id}-{start.s:.2f}-{points[-1].road_id}-{points[-1].s:.2f}"
    key=hashlib.sha256(identity.encode()).hexdigest()[:20]
    intervals=tuple(RoadInterval(road,min(values),max(values)) for road,values in sorted(roads.items()))
    section=GeometrySection(key,town,family,intervals,tuple(sorted(junctions)),direction=direction)
    junction_ranges=[]
    run_start=None
    for i,wp in enumerate(points):
        if wp.is_junction and run_start is None:
            run_start=i
        if run_start is not None and (not wp.is_junction or i==len(points)-1):
            junction_ranges.append([float(arc[run_start]),float(arc[i])])
            run_start=None
    doc=dict(section=asdict(section),start=dict(road_id=start.road_id,lane_id=start.lane_id,s=start.s),
             center_xyz=centers,left_xyz=left,right_xyz=right,arc_m=arc.tolist(),
             curvature_1pm=curvature.tolist(),length_m=float(arc[-1]),yaw_change_rad=delta,
             peak_curvature_1pm=peak,junction_ranges_m=junction_ranges,marking_support_qualified=False,
             geometry_classification="candidate_requires_rendered_maneuver_review")
    return section,doc


def inventory_maps(root, config=None):
    import carla
    config=config or Config();root=Path(root)
    source=Path(config.carla_root)/"CarlaUE4/Content/Carla/Maps/OpenDrive"
    sections=[];routes={};sources={};short=0
    for town in APPROVED_TOWNS:
        path=source/f"{town}.xodr";text=path.read_text()
        road_map=carla.Map(town,text);sources[town]=dict(path=str(path),sha256=sha256_file(path))
        # A sparse arc lattice avoids fabricating distinct groups out of
        # overlapping starts. Both directions are included before union.
        starts=sorted(road_map.generate_waypoints(180.),key=lambda w:(w.road_id,w.lane_id,w.s))
        for start in starts:
            for branch in range(3 if start.is_junction else 1):
                points=sample_route(start,160.,1.,branch)
                if len(points)<150:
                    short+=1;continue
                section,doc=route_document(points,town)
                if section.section_id in routes:
                    continue
                sections.append(section);routes[section.section_id]=doc
    groups=group_geometry_sections(sections)
    deficits=[];assignments=[]
    try:
        assignments=assign_splits(groups,seed=config.seed)
    except ValueError as error:
        deficits.append(str(error))
    result=dict(schema_version=1,sources=sources,route_count=len(routes),group_count=len(groups),
                family_candidates=dict(Counter(section.family for section in sections)),
                town_groups=dict(Counter(town for group in groups for town in group.towns)),
                rejected_short_routes=short,split_status="prepared" if assignments else "coverage_incomplete",
                deficits=deficits,assignments=[asdict(a) for a in assignments],groups=[asdict(g) for g in groups],
                junction_link_policy="shared junction IDs unioned; inter-junction adjacency still requires geometry review before split freeze",
                splits_frozen=False)
    write_json(root/"reports/inventory.json",result)
    write_json(root/"reports/routes.json",routes)
    return result
