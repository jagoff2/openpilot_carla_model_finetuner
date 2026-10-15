"""Independent admission of native behavior captures, separate from legacy data.

No passed flag from capture or a persisted observation audit grants admission.
Physical motion, actual actor envelopes, current raw observation replay, packed
images and the frozen native geometry split are checked afresh. Publication is
append-only into a new accepted shard; the legacy manifest/index is untouched.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import subprocess
import uuid
import xml.etree.ElementTree as ET

import numpy as np

from lane_dataset.config import sha256_file, write_json
from lane_dataset.geometry import LaneReference
from lane_dataset.loader import FrameArchiveReader
from lane_dataset.render_quality import validate_visual_capture
from lane_dataset.review_hold import assert_episode_not_held
from lane_dataset.splits import GeometryGroup, GeometrySection, RoadInterval, SplitAssignment, audit_split_overlap
from lane_dataset.traffic import convex_hull
from lane_dataset.visibility import decode_carla_depth_bgra
from .evidence import pack_model_rgb, decode_carla_instance_bgra
from .loader import ADMISSION_SCHEMA, REQUIRED_CHECKS, SCHEMA, AdmittedBehaviorSequence, evidence_file_bindings
from .planned import PlannedObstacle, planned_scene_geometry
from .passing import PlannedPassingSpace, ObservedPassingSpace
from .records import digest, replay_record
from .target import BehaviorConfig, EgoFootprint, build_behavior_target
from .teacher_quality import qualify_teacher_rollout
from .quality import _project
from .window import build_behavior_window
from .split_guard import publication_lock, supplemental_split_errors, assert_supplemental_split


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _member(directory, relative):
    if not isinstance(relative,str) or not relative:
        raise ValueError("invalid relative artifact name")
    path=(directory/relative).resolve()
    if directory not in path.parents:
        raise ValueError("artifact escapes capture directory")
    return path


def verify_inventory(directory, inventory):
    if not isinstance(inventory,dict) or not inventory:
        raise ValueError("capture inventory is missing")
    for relative,expected in inventory.items():
        if sha256_file(_member(directory,relative))!=expected:
            raise ValueError("capture inventory hash mismatch: "+relative)


def verify_record_evidence(directory,record,index,report,evidence):
    """Replay canonical passing claims separately from byte-bound file names."""
    files=evidence_file_bindings(record);verify_inventory(directory,files)
    if record.get('passing_space') is None:return None
    from .free_space import FreeSpaceConfig,verify_free_space_proof,observed_passing_space
    from .reprocess import passing_observations
    from .evidence import FrameStamp
    candidates=[path for path in files if Path(path).name==f'free-space-{index:05d}.json']
    if len(candidates)!=1:raise ValueError('passing record requires one exact retained free-space proof')
    relative=candidates[0];saved=_json(_member(directory,relative))
    if saved.get('config')!=asdict(FreeSpaceConfig()):raise ValueError('passing raw proof uses a different evidence contract')
    decision=FrameStamp(**saved['decision'])
    if (decision.frame_id!=record['label_frame_id'] or decision.timestamp_s!=record['label_timestamp_s']
            or decision.camera!='narrow' or saved.get('opposing_width_m')!=1.
            or saved.get('station_interval_m') is not None):
        raise ValueError('passing proof decision or bounded strip changed')
    requested=[(p['input']['stamp']['frame_id'],p['input']['stamp']['camera']) for p in saved['frame_proofs']]
    frame_ids=sorted(set(f for f,c in requested));by_id={r['frame_id']:r for r in evidence}
    if any(f not in by_id for f in frame_ids):raise ValueError('passing raw frame is missing')
    frames=[by_id[f] for f in frame_ids]
    if requested!=[(f['frame_id'],c) for f in frames for c in ('narrow','wide')]:
        raise ValueError('passing proof differs from its complete captured frame pair ordering')
    observations,scenes,raw_bindings=passing_observations(directory,frames,report['files'])
    if any(files.get(name)!=sha for name,sha in raw_bindings.items()):
        raise ValueError('passing raw evidence is not retained in record bindings')
    road,obstacles,_,_=replay_record(record,verified_evidence_bindings=record['evidence_bindings'])
    rebuilt=verify_free_space_proof(saved,observations,road,decision=decision,controlled_scenes=scenes,
        obstacle_ids=tuple(o.identity for o in obstacles),opposing_width_m=1.)
    typed=observed_passing_space(rebuilt,road)
    if typed!=ObservedPassingSpace.from_document(record['passing_space']):
        raise ValueError('raw free-space replay differs from observed passing permission')
    return dict(method='independent_saved_free_surface_replay',proof_file=relative,
        proof_file_sha256=files[relative],replayed_proof_document_sha256=digest(rebuilt))


def _check_capture_exclusion(directory):
    path=Path(directory)/"capture-exclusion.json"
    if path.exists():raise ValueError("capture explicitly excluded from admission: "+str(path))


def reconstruct_teacher(document):
    """Rebuild the plan from its source arrays, never trust serialized output."""
    source=document["source"]
    ref=source["reference"]
    target=planned_scene_geometry(LaneReference(ref["left_xyz"],ref["right_xyz"],ref["s"]),
        tuple(PlannedObstacle(**o) for o in source["obstacles"]),EgoFootprint(**source["ego"]),
        config=BehaviorConfig(**source["config"]),source_metadata=source["source_metadata"],
        minimum_retained_nominal_recovery_m=source["minimum_retained_nominal_recovery_m"],
        obstacle_tracking_reserve_m=source.get("obstacle_tracking_reserve_m",0.),
        passing_space=PlannedPassingSpace.from_document(source["passing_space"]) if "passing_space" in source else None)
    if digest(target.document())!=digest(document):
        raise ValueError("teacher target source/solution does not replay exactly")
    return target


def verify_actual_envelopes(target, evidence, telemetry):
    """Check saved real bbox geometry, actor identity and measured stationarity."""
    if not evidence:
        raise ValueError("actual actor envelope evidence missing")
    required=list(range(0,len(telemetry),5))
    if [f["index"] for f in evidence]!=required:
        raise ValueError("actual actor QA requires uninterrupted 4 Hz evidence")
    footprints={o.identity:o.polygon() for o in target.obstacles}
    actor_rows=[]
    for frame in evidence:
        row=telemetry[frame["index"]]
        if frame["frame_id"]!=row["frame_id"] or abs(frame["timestamp_s"]-row["timestamp"])>1e-6:
            raise ValueError("actor evidence is not aligned with telemetry")
        actor=frame["ego"]; box=actor["bounding_box"]
        measured=EgoFootprint(2*box["extent_m"][0],2*box["extent_m"][1],
            tuple(box["center_actor_m"][:2]),math.radians(box["rotation_degrees"][2]))
        if (abs(box["rotation_degrees"][0])+abs(box["rotation_degrees"][1])>1e-6
                or not np.allclose(measured.corners(),target.ego.corners(),atol=1e-6,rtol=0)
                or not np.allclose(actor["position_world_m"],row["position_world"],atol=1e-5,rtol=0)
                or abs((actor["yaw_rad"]-row["rotation_world"][2]+np.pi)%(2*np.pi)-np.pi)>1e-6):
            raise ValueError("actual ego envelope/pose differs from physical replay")
        obstacle=frame["obstacle"]
        if not footprints:
            if obstacle is not None:
                raise ValueError("unplanned obstacle in actual capture")
            continue
        if len(footprints)!=1 or obstacle is None or str(obstacle["actor_id"]) not in footprints:
            raise ValueError("actual obstacle identity differs from fixed teacher scene")
        actor_rows.append(obstacle)
    actual_sweep=None
    if actor_rows:
        from .grounding import stationary_envelope
        bound=stationary_envelope(actor_rows)
        expected=footprints[str(actor_rows[0]['actor_id'])]
        # The plan remains unchanged. Every measured box is separately checked
        # against its rigid pose, and admitted clearance uses their entire union.
        for actor in actor_rows:
            polygon=convex_hull(np.asarray(actor['bounding_box']['vertices_world_xyz'])[:,:2])
            distance=np.linalg.norm(polygon[:,None]-expected[None],axis=-1)
            if max(distance.min(0).max(),distance.min(1).max())>bound['between_observation_guard_m']+1e-6:
                raise ValueError('actual obstacle differs from bounded planned fixed envelope')
        positions=np.asarray([a["position_world_m"] for a in actor_rows],float)
        velocity=np.asarray([a["velocity_world_mps"] for a in actor_rows],float)
        yaw=np.unwrap([a["yaw_rad"] for a in actor_rows])
        rate=np.asarray([a["yaw_rate_rps"] for a in actor_rows],float)
        if (not all(np.isfinite(a).all() for a in (positions,velocity,yaw,rate))
                or np.linalg.norm(positions-positions[0],axis=1).max()>.03
                or np.linalg.norm(velocity,axis=1).max()>.05
                or abs(yaw-yaw[0]).max()>math.radians(.5)
                or abs(rate).max()>math.radians(.5)):
            raise ValueError("actual obstacle is not stationary")
        bound['polygon_xy']=convex_hull(np.r_[expected,np.asarray(bound['polygon_xy'])]).tolist()
        actual_sweep=_actual_obstacle_clearance(target,telemetry,bound)
    return dict(passed=True,observed_states=len(evidence),
        actual_obstacle_sweep=actual_sweep,
        scope="actual 4 Hz bbox/actor QA; physical trajectory is independently 20 Hz")


def _actual_obstacle_clearance(target,telemetry,bound):
    """Sweep the measured ego body against all measured stationary envelopes."""
    from .quality import QualityConfig,_body
    from .target import _polygon_distance
    cfg=QualityConfig();positions=np.asarray([r['position_world'] for r in telemetry],float)
    yaw=np.unwrap([r['rotation_world'][2] for r in telemetry])
    if len(positions)<2 or not np.isfinite(np.r_[positions.ravel(),yaw]).all():
        raise ValueError('actual obstacle sweep requires finite physical trajectory')
    corners=target.ego.corners();radius=float(np.linalg.norm(corners,axis=1).max())
    polygon=np.asarray(bound['polygon_xy']);guard=bound['between_observation_guard_m']
    minimum=math.inf;pieces=0;ego_rotation_guard=0.
    for index in range(len(positions)-1):
        distance=float(np.linalg.norm(positions[index+1,:2]-positions[index,:2]));angle=abs(float(yaw[index+1]-yaw[index]))
        count=max(1,math.ceil(distance/cfg.maximum_sweep_spacing_m),math.ceil(angle/cfg.maximum_sweep_heading_step_rad))
        rotation_guard=radius*(1-math.cos(angle/count/2));ego_rotation_guard=max(ego_rotation_guard,rotation_guard)
        before=_body(positions[index],yaw[index],corners)
        for fraction in np.arange(1,count+1)/count:
            point=positions[index]*(1-fraction)+positions[index+1]*fraction
            heading=yaw[index]*(1-fraction)+yaw[index+1]*fraction
            after=_body(point,heading,corners);swept=convex_hull(np.r_[before,after])
            minimum=min(minimum,_polygon_distance(swept,polygon)-rotation_guard-guard)
            before=after;pieces+=1
    obstacle=target.obstacles[0];required=target.config.obstacle_clearance(obstacle.kind)
    if required is None or minimum<required-1e-8:
        raise ValueError(f'actual recorded obstacle sweep clearance below unchanged limit: {minimum:.6f} < {required}')
    return dict(passed=True,minimum_clearance_m=float(minimum),required_clearance_m=required,
        obstacle_id=obstacle.identity,stationary_envelope=bound,ego_rotation_guard_m=ego_rotation_guard,
        swept_polygon_pieces=pieces,scope='20 Hz measured ego sweep against actual box union and bounded stationary interpolation')


def replay_physical(directory, report):
    inventory=report["files"]
    for name in ("teacher-target.json","telemetry.json","evidence-frames.jsonl","collisions.json","candidate-task.json"):
        if name not in inventory:
            raise ValueError("physical replay input not hash-bound: "+name)
    target=reconstruct_teacher(_json(directory/"teacher-target.json"))
    task=_json(directory/"candidate-task.json")
    if (target.contract_sha256!=report.get("teacher_contract_sha256")
            or target.source_metadata.get("task_sha256")!=inventory["candidate-task.json"]
            or target.source_metadata.get("plan_sha256")!=report.get("plan_sha256")):
        raise ValueError("teacher target does not bind the exact capture task/plan")
    edges=task["route"]["unmarked_road"]
    if (not np.array_equal(target.reference.left_xyz,edges["road_left_edge_xyz"])
            or not np.array_equal(target.reference.right_xyz,edges["road_right_edge_xyz"])
            or not np.array_equal(target.reference.s,task["route"]["arc_m"])):
        raise ValueError("teacher road differs from recorded native task")
    if target.passing_space is not None:
        if (task['unmarked_behavior'].get('passing_space_policy')
                !='observed_clear_surface_during_static_parked_avoidance_only'
                or inventory.get('controlled-scene.json')!=target.passing_space.controlled_scene_sha256
                or report.get('controlled_scene_sha256')!=target.passing_space.controlled_scene_sha256):
            raise ValueError('privileged passing teacher lacks exact controlled-scene/task binding')
    rows=_json(directory/"telemetry.json")
    evidence=[json.loads(line) for line in (directory/"evidence-frames.jsonl").read_text().splitlines() if line.strip()]
    collision=_json(directory/"collisions.json")
    if not isinstance(collision,list):
        raise ValueError("collision sensor result is not retained")
    envelopes=verify_actual_envelopes(target,evidence,rows)
    grounding=None
    if target.obstacles:
        if "obstacle-grounding.json" not in inventory:
            raise ValueError("stationary obstacle lacks hash-bound in-road grounding/support evidence")
        from .grounding import verify_grounding
        grounding=verify_grounding(_json(directory/"obstacle-grounding.json"),target.reference,task=task,
            recorded_actors=[frame["obstacle"] for frame in evidence])
    physical=qualify_teacher_rollout(target=target,target_sha256=target.contract_sha256,
        timestamps_s=[r["timestamp"] for r in rows],positions_xyz=[r["position_world"] for r in rows],
        yaw_rad=[r["rotation_world"][2] for r in rows],velocity_world=[r["velocity_world"] for r in rows],
        lateral_acceleration_mps2=[r["lateral_acceleration_mps2"] for r in rows],collision_count=len(collision))
    physical["actual_envelope_replay"]=envelopes
    physical["obstacle_grounding_replay"]=grounding
    if task["unmarked_behavior"]["obstacles"] and not target.excursions:
        physical["passed"]=False;physical["reasons"].append("required_nonzero_avoidance_not_demonstrated")
    return physical,target,rows,evidence


def _section(value):
    return GeometrySection(**{**value,"road_intervals":tuple(RoadInterval(**r) for r in value["road_intervals"])})


def verify_split(root, task, task_sha256, assignment):
    """Audit an explicit assignment against current frozen native geometry.

Assignment keys are candidate_task_sha256, split, frozen_split_sha256.
The baseline path comes from corpus qualification.json, never caller input.
The candidate consumes its entire native road plus attached junctions, so
appearances, reverse traversal and preview cannot fabricate independence.
"""
    if (not isinstance(assignment,dict) or assignment.get("candidate_task_sha256")!=task_sha256
            or assignment.get("split") not in ("train","validation","test")):
        raise ValueError("explicit exact-task train/validation/test assignment required")
    qualification=_json(root/"qualification.json")
    binding=qualification["split_binding"]
    baseline_path=Path(binding["path"])
    if not baseline_path.is_absolute():baseline_path=root/baseline_path
    baseline_sha=sha256_file(baseline_path)
    if baseline_sha!=binding["sha256"] or assignment.get("frozen_split_sha256")!=baseline_sha:
        raise ValueError("split approval does not bind current frozen baseline")
    baseline=_json(baseline_path)
    if baseline.get("splits_frozen") is not True:
        raise ValueError("native baseline splits are not frozen")
    groups=[GeometryGroup(g["group_id"],tuple(_section(s) for s in g["sections"])) for g in baseline["groups"]]
    assignments=[SplitAssignment(**a) for a in baseline["assignments"]]
    route=task["route"]
    xodr=Path(route["source_xodr"])
    if (task.get("native_unmarked") is not True or task.get("procedural") is not False
            or sha256_file(xodr)!=route["source_xodr_sha256"]):
        raise ValueError("native split geometry source is not bound")
    xml=ET.parse(xodr).getroot()
    ids={str(row["road_id"]) for row in route["section"]["road_intervals"]}
    if ids!={str(task["start"]["road_id"])}:
        raise ValueError("bounded native behavior assignment requires one complete road")
    roads=[r for r in xml.findall("road") if r.get("id") in ids]
    if len(roads)!=1 or roads[0].get("junction")!="-1":
        raise ValueError("native road topology does not match task")
    road=roads[0]
    junctions=tuple(sorted({link.get("elementId") for link in road.findall("link/*") if link.get("elementType")=="junction"}))
    section=GeometrySection("behavior-"+task_sha256,task["town"],task["family"],
        (RoadInterval(road.get("id"),0.,float(road.get("length"))),),junctions,
        direction=route["section"]["direction"])
    group_id="behavior-"+task_sha256
    groups.append(GeometryGroup(group_id,(section,)))
    assignments.append(SplitAssignment(group_id,assignment["split"],task["family"],1,int(task["seed"])))
    # Include previous supplemental publications; legacy grouping stays intact.
    previous_sections={}
    for path in sorted((root/"accepted").glob("*/behavior-manifest.json")):
        manifest=_json(path)
        saved=_json(path.parent/"behavior-split.json")
        if manifest.get("files",{}).get("behavior-split.json")!=sha256_file(path.parent/"behavior-split.json"):
            raise ValueError("previous behavior split binding changed")
        previous=_section(saved["candidate_section"])
        if previous.section_id==section.section_id:
            if manifest["split"]!=assignment["split"]:
                raise ValueError("same behavior task already assigned to another split")
            continue
        seen=previous_sections.get(previous.section_id)
        if seen is not None:
            if seen!=(digest(asdict(previous)),manifest["split"]):
                raise ValueError("prior behavior geometry has conflicting split bindings")
            continue
        previous_sections[previous.section_id]=(digest(asdict(previous)),manifest["split"])
        groups.append(GeometryGroup(previous.section_id,(previous,)))
        assignments.append(SplitAssignment(previous.section_id,manifest["split"],previous.family,1,0))
    errors=audit_split_overlap(groups,assignments)
    errors.extend(supplemental_split_errors(root,section=section,split=assignment["split"]))
    return dict(passed=not errors,errors=errors,split=assignment["split"],assignment=assignment,
        candidate_section=asdict(section),baseline_sha256=baseline_sha,
        baseline_path=str(baseline_path.resolve()),baseline_document=baseline,
        source_xodr_sha256=route["source_xodr_sha256"])


def verify_retention(directory, report, rows, evidence):
    frames=[json.loads(line) for line in (directory/"frames.jsonl").read_text().splitlines() if line.strip()]
    if len(frames)!=len(rows) or len(rows)<301:
        raise ValueError("packed sequence lacks aligned five-second history and ten-second future")
    for i,(frame,row) in enumerate(zip(frames,rows)):
        if (frame["index"]!=i or row["index"]!=i or frame["frame_id"]!=row["frame_id"]
                or abs(frame["timestamp"]-row["timestamp"])>1e-6
                or frame["desire"]!=[1.,0.,0.,0.,0.,0.,0.,0.] or frame["traffic_convention"]!=[1.,0.]
                or (i and (frame["frame_id"]!=frames[i-1]["frame_id"]+1
                    or abs(frame["timestamp"]-frames[i-1]["timestamp"]-.05)>1e-6))):
            raise ValueError("20 Hz camera/telemetry/frame metadata mismatch")
    for camera in ("narrow","wide"):
        entry=report["archives"][camera]
        if report["files"].get(entry["path"])!=entry["sha256"]:
            raise ValueError("packed archive not in capture inventory")
        reader=FrameArchiveReader(_member(directory,entry["path"]),expected_archive_sha256=entry["sha256"])
        try:
            if len(reader)!=len(frames):raise ValueError("packed archive count mismatch")
            for proof in evidence:
                meta=proof["cameras"][camera]
                if report["files"].get(meta["payload"])!=meta["payload_sha256"]:
                    raise ValueError("raw proof not in capture inventory")
                with np.load(_member(directory,meta["payload"]),allow_pickle=False) as data:
                    if not np.array_equal(data["packed"],reader[proof["index"]]):
                        raise ValueError("raw evidence differs from retained model frame")
                    if (not np.array_equal(data["rgb"],data["rgb_bgra"][...,[2,1,0]])
                            or not np.array_equal(data["packed"],pack_model_rgb(data["rgb"],data["model_to_source"]))
                            or not np.array_equal(data["depth_m"],decode_carla_depth_bgra(data["depth_bgra"]).astype(np.float32))):
                        raise ValueError("raw RGB/packing/depth encoding failed independent replay")
                    semantic,instance=decode_carla_instance_bgra(data["instance_bgra"])
                    if not np.array_equal(semantic,data["semantic_ids"]) or not np.array_equal(instance,data["instance_ids"]):
                        raise ValueError("raw instance/semantic encoding failed independent replay")
                    expected=dict(frame_id=proof["frame_id"],timestamp_s=proof["timestamp_s"],camera=camera)
                    if (any(value!=expected for value in meta["modality_stamps"].values())
                            or meta["separate_semantic_sensor_stamp"]!=expected):
                        raise ValueError("raw modality frame stamps do not align")
        finally:reader.close()
    return dict(passed=True,frame_count=len(frames),raw_proof_frames=len(evidence),
        raw_encoding_scope="all retained 4 Hz raw RGB/packing/depth/instance payloads and stamps, independent of 2 Hz road geometry cadence")


def verify_label_recovery(target, rows, records):
    indices=np.asarray([r["label_index"] for r in records],int)
    if not len(indices):return dict(passed=False,errors=["no_observed_behavior_labels"])
    if np.any(np.diff(indices)<=0) or np.any(indices%10) or indices[0]<100 or indices[-1]>len(rows)-201:
        raise ValueError("behavior candidate label retention/index invalid")
    # A label can see and supervise a phase long before the physical ego reaches
    # it. Coverage is the observed TARGET geometry, not ego-at-label position.
    # Privileged teacher station is used only to audit completed scene coverage.
    errors=[];intervals=[];nominal_intervals=[];per_label=[]
    nominal=target.nominal_xy
    nominal_arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(nominal,axis=0),axis=1))]
    for item in records:
        record=item["record"]
        road,obstacles,ego,config=replay_record(record,verified_evidence_bindings=record["evidence_bindings"])
        permission=None if record.get('passing_space') is None else ObservedPassingSpace.from_document(record['passing_space'])
        if record.get("target_mode") in ("observed_future_window","observed_future_geometry"):
            observed=build_behavior_window(road,obstacles,ego,config=config,source_ego_s=record["source_ego_s"],passing_space=permission)
        else:observed=build_behavior_target(road,obstacles,ego,config=config,passing_space=permission)
        if not observed.moving_supervision:
            raise ValueError("recovery coverage contains an unsupported target")
        q=np.linspace(observed.s[0],observed.s[-1],max(2,int(np.ceil((observed.s[-1]-observed.s[0])/.1))+1))
        xy,_,_,_,valid=observed.sample(q)
        if not valid.all():raise ValueError("observed recovery target has masked gaps")
        station,_,_=_project(xy,target.s,nominal)
        if np.any(np.diff(station)<=1e-7):
            raise ValueError("observed target phase correspondence is clipped, reversed or ambiguous")
        interval=[float(station[0]),float(station[-1])]
        intervals.append(interval)
        observed_nominal=.25*road.reference._left(q)[:,:2]+.75*road.reference._right(q)[:,:2]
        nominal_mask=np.linalg.norm(xy-observed_nominal,axis=1)<=1e-6
        starts=np.flatnonzero(nominal_mask&np.r_[True,~nominal_mask[:-1]])
        ends=np.flatnonzero(nominal_mask&np.r_[~nominal_mask[1:],True])
        nominal_intervals.extend([float(station[a]),float(station[b])] for a,b in zip(starts,ends) if b>a)
        per_label.append(dict(label_index=item["label_index"],record_sha256=record["record_sha256"],
            teacher_station_interval_m=interval,supported_points=len(q)))
    def merge(values):
        merged=[]
        for lo,hi in sorted(values):
            if merged and lo<=merged[-1][1]+1e-6:merged[-1][1]=max(merged[-1][1],hi)
            else:merged.append([lo,hi])
        return merged
    merged=merge(intervals);merged_nominal=merge(nominal_intervals);event_coverage=[]
    for number,event in enumerate(target.excursions):
        phases={}
        for phase,lo,hi in (("departure",event.start_s,event.plateau_start_s),
                ("pass",event.plateau_start_s,event.plateau_end_s),("return",event.plateau_end_s,event.end_s)):
            pieces=[[max(a,lo),min(b,hi)] for a,b in merged if min(b,hi)>max(a,lo)]
            lengths=[b-a for a,b in pieces]
            longest=max(lengths,default=0.);cursor=lo;gaps=[]
            for a,b in pieces:
                if a>cursor+1e-6:gaps.append([cursor,a])
                cursor=max(cursor,b)
            if cursor<hi-1e-6:gaps.append([cursor,hi])
            phases[phase]=dict(interval_m=[lo,hi],covered_intervals_m=pieces,covered_m=sum(lengths),
                covered_fraction=sum(lengths)/(hi-lo),maximum_contiguous_m=longest,gaps_m=gaps,
                minimum_required_contiguous_m=.25,passed=longest>=.25-1e-6)
            if not phases[phase]["passed"]:errors.append(f"event_{number}:no_contiguous_observed_{phase}_geometry")
        # Nominal recovery may become observable after its very first metre.
        # Require a genuinely nominal contiguous five-metre observed target
        # interval, while independent physical QA still checks full traversal.
        next_start=target.excursions[number+1].start_s if number+1<len(target.excursions) else target.s[-1]
        pieces=[[max(a,event.end_s),min(b,next_start)] for a,b in merged_nominal if min(b,next_start)>max(a,event.end_s)]
        distances=[float(np.interp(b,target.s,nominal_arc)-np.interp(a,target.s,nominal_arc)) for a,b in pieces]
        longest=max(distances,default=0.)
        if longest<max(5.,target.minimum_retained_nominal_recovery_m)-1e-6:
            errors.append(f"event_{number}:observed_recovery_does_not_retain_five_meters")
        event_coverage.append(dict(event_index=number,phases=phases,nominal_recovery_intervals_m=pieces,
            maximum_contiguous_nominal_recovery_m=longest,required_nominal_recovery_m=max(5.,target.minimum_retained_nominal_recovery_m)))
    return dict(passed=not errors,errors=errors,label_count=len(indices),events=len(target.excursions),
        observed_geometry_intervals_m=merged,observed_nominal_intervals_m=merged_nominal,labels=per_label,
        event_coverage=event_coverage,entire_event_visually_supervised=False,
        scope="independently observed body-supported partial phase geometry and five-metre nominal recovery; complete physical event is checked separately")


def current_split_assignment(root, assignment):
    """Refresh only a fixed task/split's baseline binding, never its identity."""
    if (not isinstance(assignment,dict) or assignment.get("split") not in ("train","validation","test")
            or not isinstance(assignment.get("candidate_task_sha256"),str)):
        raise ValueError("exact-task fixed split assignment required for refresh")
    binding=_json(Path(root)/"qualification.json")["split_binding"]
    return {**assignment,"frozen_split_sha256":binding["sha256"]}


def audit_candidate(capture, *, corpus_root, split_assignment=None, publish=False, loader_python=None,
                    refresh_split_binding=False, expected_source_hashes=None):
    """Run checks and save a new audit. Missing evidence never creates acceptance."""
    directory,root=Path(capture).resolve(),Path(corpus_root).resolve()
    output=directory/"admission-audits"/uuid.uuid4().hex
    output.mkdir(parents=True,exist_ok=False)
    audit=dict(schema="unmarked_candidate_admission_audit_v1",passed=False,training_ready=False,
        accepted_labels=0,checks={key:False for key in REQUIRED_CHECKS},errors=[],details={},
        visual_review="machine_checks_only_no_human_review_claim",capture=str(directory))
    audit["refresh_split_binding_allowed"]=bool(refresh_split_binding)
    audit["requested_split_assignment"]=split_assignment
    if expected_source_hashes is not None:audit["expected_source_hashes"]=dict(expected_source_hashes)
    records=[];report=None;split=None
    def attempt(name,operation):
        try:
            value=operation();audit["details"][name]=value
            return value
        except Exception as error:
            audit["errors"].append(name+": "+str(error));return None
    def inputs():
        _check_capture_exclusion(directory)
        value=_json(directory/"capture-report.json")
        if (value.get("scope")!="physical_teacher_capture_pending_observed_admission"
                or value.get("status")!="captured_pending_observed_admission" or value.get("errors")
                or value.get("cleanup_errors") or value.get("training_ready") is not False):
            raise ValueError("capture was incomplete or failed")
        verify_inventory(directory,value["files"])
        assert_episode_not_held(root/"accepted"/value["episode_id"]/"manifest.json",corpus=root)
        return value
    report=attempt("input_integrity",inputs)
    if report is not None:
        physical=attempt("physical_replay",lambda:replay_physical(directory,report))
        if physical is not None:
            qa,target,rows,evidence=physical
            audit["details"]["physical_replay"]=qa
            audit["checks"]["physical_quality"]=qa["passed"] is True
            retained=attempt("retention",lambda:verify_retention(directory,report,rows,evidence))
            audit["checks"]["retention_integrity"]=retained is not None and retained["passed"]
            visual=attempt("rendered",lambda:validate_visual_capture(directory,report,dict(speed=[r["speed"] for r in rows])))
            audit["checks"]["rendered_quality"]=visual is not None and visual["passed"] is True
            task=_json(directory/"candidate-task.json")
            def check_split():
                if not refresh_split_binding:
                    return verify_split(root,task,report["files"]["candidate-task.json"],split_assignment)
                with publication_lock(root):
                    assignment=current_split_assignment(root,split_assignment)
                    return verify_split(root,task,report["files"]["candidate-task.json"],assignment)
            split=attempt("split",check_split)
            audit["checks"]["split_integrity"]=split is not None and split["passed"] is True
            def observations():
                from .reprocess import reprocess
                result,replay_dir=reprocess(directory)
                candidate=_json(replay_dir/result["records_file"])
                for item in candidate:
                    record=item["record"]
                    passing_replay=verify_record_evidence(directory,record,item['label_index'],report,evidence)
                    if passing_replay is not None:item['passing_evidence_replay']=passing_replay
                    replay_record(record,verified_evidence_bindings=record["evidence_bindings"])
                    index=item["label_index"];row=rows[index]
                    if (record["label_frame_id"]!=row["frame_id"] or abs(record["label_timestamp_s"]-row["timestamp"])>1e-6
                            or not np.allclose(record["ego_position_world"],row["position_world"],atol=1e-6,rtol=0)):
                        raise ValueError("replayed record differs from actual frame")
                return result,candidate
            observed=attempt("observations",observations)
            if observed is not None:
                replay_report,records=observed
                audit["details"]["observations"]=replay_report
                for check in ("road_evidence_replayed","obstacle_evidence_replayed","behavior_target_replayed"):
                    audit["checks"][check]=bool(records)
                recovery=attempt("recovery",lambda:verify_label_recovery(target,rows,records))
                audit["checks"]["event_recovery"]=recovery is not None and recovery["passed"] is True
    audit["passed"]=all(audit["checks"].values()) and not audit["errors"]
    audit["remaining_checks"]=[key for key,value in audit["checks"].items() if not value]
    audit["candidate_label_count"]=len(records)
    write_json(output/"candidate-audit.json",audit)
    if publish and audit["passed"]:
        audit["published_manifest"]=str(_publish(directory,root,report,records,split,audit,output,loader_python=loader_python))
        audit["accepted_labels"]=len(records);audit["training_ready"]=True
        write_json(output/"publication-result.json",audit)
    return audit,output


def _publish(directory,root,report,records,split,audit,audit_output,*,loader_python=None):
    """Private publication only after fresh audit, serialized against split races."""
    if not audit["passed"] or not all(audit["checks"].get(key) is True for key in REQUIRED_CHECKS):
        raise ValueError("incomplete behavior admission cannot publish")
    if digest(_json(directory/"capture-report.json"))!=digest(report):
        raise ValueError("capture report changed after independent audit")
    name="behavior-"+report["episode_id"]+"-"+sha256_file(directory/"capture-report.json")[:12]
    if name!=Path(name).name or "/" in name or "\\" in name:
        raise ValueError("unsafe episode identity")
    final=root/"accepted"/name
    assert_episode_not_held(final/"manifest.json",corpus=root)
    with publication_lock(root):
        _check_capture_exclusion(directory)
        if audit.get("expected_source_hashes") is not None:
            source_root=Path(__file__).resolve().parents[1]
            current={p.relative_to(source_root).as_posix():sha256_file(p)
                for package in ("lane_behavior","lane_dataset") for p in (source_root/package).glob("*.py")}
            if current!=audit["expected_source_hashes"]:
                raise ValueError("queue source changed during capture or admission")
        original_assignment=split["assignment"]
        assignment=current_split_assignment(root,original_assignment) if audit.get("refresh_split_binding_allowed") else original_assignment
        split=verify_split(root,_json(directory/"candidate-task.json"),report["files"]["candidate-task.json"],assignment)
        split["publication_binding_history"]=dict(requested=audit.get("requested_split_assignment",original_assignment),
            audited=original_assignment,published=assignment,refresh_allowed=bool(audit.get("refresh_split_binding_allowed")))
        if not split["passed"]:raise ValueError("split changed before publication: "+str(split["errors"]))
        assert_supplemental_split(root,section=split["candidate_section"],split=split["split"])
        if final.exists():raise FileExistsError("behavior output is immutable and already exists")
        stage=root/"staging"/(name+"-"+uuid.uuid4().hex)
        stage.mkdir(parents=True,exist_ok=False)
        inventory=dict(report["files"])
        inventory["capture-report.json"]=sha256_file(directory/"capture-report.json")
        for item in records:inventory.update(evidence_file_bindings(item['record']))
        verify_inventory(directory,inventory)
        for relative,expected in inventory.items():
            destination=_member(stage,relative);destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(_member(directory,relative),destination)
            if sha256_file(destination)!=expected:raise ValueError("publication copy changed: "+relative)
        write_json(stage/"behavior-records.json",[r["record"] for r in records])
        np.savez_compressed(stage/"behavior-labels.npz",label_indices=np.asarray([r["label_index"] for r in records],np.int64))
        write_json(stage/"behavior-split.json",split)
        write_json(stage/"behavior-audit.json",audit)
        for relative in ("behavior-records.json","behavior-labels.npz","behavior-split.json","behavior-audit.json"):
            inventory[relative]=sha256_file(stage/relative)
        admission=dict(schema=ADMISSION_SCHEMA,passed=True,method="independent_raw_evidence_and_physical_replay",
            checks=audit["checks"],input_files=dict(inventory),
            validated_records=[dict(label_index=r["label_index"],record_sha256=r["record"]["record_sha256"],
                verified_evidence_bindings=r["record"]["evidence_bindings"],
                **({'passing_evidence_replay':r['passing_evidence_replay']} if 'passing_evidence_replay' in r else {})) for r in records],
            rendered_review="machine_only_no_human_claim")
        write_json(stage/"behavior-admission.json",admission)
        inventory["behavior-admission.json"]=sha256_file(stage/"behavior-admission.json")
        manifest=dict(schema=SCHEMA,status="accepted",training_ready=True,capture_scope="production",
            split=split["split"],episode_id=name,accepted_label_count=len(records),frames_file="frames.jsonl",
            labels_file="behavior-labels.npz",behavior_records_file="behavior-records.json",
            admission_file="behavior-admission.json",archives=report["archives"],files=inventory)
        write_json(stage/"behavior-manifest.json",manifest)
        if loader_python is None:
            with AdmittedBehaviorSequence(stage/"behavior-manifest.json",corpus_root=root) as sequence:
                sequence.assert_admitted()
        else:
            # Keep deterministic raw/physical replay in its capture interpreter;
            # the actual training interpreter must independently load the result.
            code="import sys\nfrom lane_behavior.loader import AdmittedBehaviorSequence\nwith AdmittedBehaviorSequence(sys.argv[1],corpus_root=sys.argv[2]) as sequence:\n sequence.assert_admitted()\nprint('behavior loader verification passed')\n"
            checked=subprocess.run([str(loader_python),"-c",code,str(stage/"behavior-manifest.json"),str(root)],
                cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,check=False)
            if checked.returncode:
                raise ValueError("training loader verification failed: "+(checked.stderr or checked.stdout)[-3000:])
        final.parent.mkdir(parents=True,exist_ok=True)
        # Windows rename fails if final exists; no replace and no prior data deletion.
        stage.rename(final)
    return final/"behavior-manifest.json"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture",type=Path);parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--split-assignment",type=Path);parser.add_argument("--publish",action="store_true")
    parser.add_argument("--loader-python",type=Path,help="optional Torch-enabled interpreter for final staged loader verification")
    parser.add_argument("--refresh-split-binding",action="store_true",help="refresh only the frozen baseline SHA and fully recheck the unchanged task/split at publication")
    args=parser.parse_args()
    assignment=_json(args.split_assignment) if args.split_assignment else None
    result,path=audit_candidate(args.capture,corpus_root=args.root,split_assignment=assignment,publish=args.publish,
        loader_python=args.loader_python,refresh_split_binding=args.refresh_split_binding)
    print(json.dumps(dict(audit=str(path),passed=result["passed"],accepted_labels=result["accepted_labels"],
        remaining_checks=result["remaining_checks"],errors=result["errors"])),flush=True)
    return 0 if result["passed"] else 2


if __name__=="__main__":raise SystemExit(main())
