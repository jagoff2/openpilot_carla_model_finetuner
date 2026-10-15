"""Replay physical behavior captures into candidate observed geometry records.

This step never publishes data, changes split assignments, or grants training
admission. It cheaply reuses saved camera evidence after capture has finished.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict
import json
import math
from pathlib import Path
import uuid

import numpy as np

from lane_dataset.config import sha256_file, write_json
from lane_dataset.geometry import EvidenceFrame, LaneReference
from lane_dataset.traffic import convex_hull
from .evidence import ActorState, FrameStamp, Observation, build_static_proof
from .records import build_record, digest
from .road_evidence import measure_road_frame, assemble_observed_road
from .target import BehaviorConfig, EgoFootprint, ObservedObstacle

ROAD_EVIDENCE_STRIDE_TICKS=10


def road_evidence_due(index):
    """Road geometry is replayed at every 2 Hz label decision, including warmup."""
    return index%ROAD_EVIDENCE_STRIDE_TICKS==0


class StaticObstacleHistory:
    """Causal visual history for one controlled, independently static fixture.

Fresh observed geometry expires after five seconds. A rear-clear certificate
may persist only as an exclusion, never as a target obstacle/evidence frame.
It requires fresh visual geometry at issuance and continuous actual stationary
actor identity plus forward route progress afterwards. Actual bbox geometry
never substitutes for a missing visual proof.
"""
    def __init__(self, category, *, maximum_age_s=5.):
        if not 0<maximum_age_s<=5.:raise ValueError("obstacle evidence age must be <=5 seconds")
        self.category=category;self.maximum_age_s=float(maximum_age_s)
        self.anchor=None;self.last_station=None;self.maximum_station=None;self.last_frame=None;self.failure=None
        self.observed=None;self.certificate=None

    def update_actual(self, frame, station):
        actor=frame.get("obstacle")
        if actor is None:
            self.failure="controlled_static_actor_missing";return
        if not np.isfinite(station):raise ValueError("invalid actual route station")
        if self.last_frame is not None and (frame["frame_id"]<=self.last_frame or station<self.maximum_station-.05):
            self.failure="static_exclusion_requires_monotone_continuing_route"
        self.last_frame=frame["frame_id"];self.last_station=float(station)
        self.maximum_station=float(station) if self.maximum_station is None else max(self.maximum_station,float(station))
        if self.anchor is None:self.anchor=json.loads(json.dumps(actor))
        old=self.anchor
        values=np.r_[actor["position_world_m"],actor["velocity_world_mps"],actor["yaw_rad"],actor["yaw_rate_rps"]]
        angle=abs((actor["yaw_rad"]-old["yaw_rad"]+np.pi)%(2*np.pi)-np.pi)
        if (not np.isfinite(values).all() or actor["actor_id"]!=old["actor_id"]
                or np.linalg.norm(np.asarray(actor["position_world_m"])-old["position_world_m"])>.03
                or np.linalg.norm(actor["velocity_world_mps"])>.05
                or angle>math.radians(.5) or abs(actor["yaw_rate_rps"])>math.radians(.5)):
            self.failure="controlled_obstacle_stationarity_or_identity_changed"

    def observe(self, proof, refs, bindings, *, current_frame):
        refs=tuple(refs)
        if (proof.get("supported") is not True or not refs or any(
                e.frame_id>current_frame["frame_id"] or e.timestamp_s>current_frame["timestamp_s"]+1e-8
                or current_frame["timestamp_s"]-e.timestamp_s>self.maximum_age_s+1e-8 for e in refs)):
            raise ValueError("static proof is not supported, causal and fresh")
        if self.anchor is None:raise ValueError("actual actor continuity must precede visual proof")
        if proof.get("actor_id")!=self.anchor["actor_id"] or proof.get("category")!=self.category:
            raise ValueError("visual proof identity/class differs from controlled actor")
        polygon=np.asarray(proof["observations"][-1]["conservative_qa_footprint_world_xy"],float)
        obstacle=ObservedObstacle(str(self.anchor["actor_id"]),polygon,self.category,refs,"static")
        obstacle.polygon()  # Validate the actual proof envelope before retaining it.
        self.observed=dict(obstacle=obstacle,proof_sha256=digest(proof),bindings=dict(bindings),
            observed_at=dict(frame_id=current_frame["frame_id"],timestamp_s=current_frame["timestamp_s"]))

    def decide(self, *, frame, ego, position_world, yaw_rad):
        base=dict(version="causal_controlled_static_obstacle_history_v1",actor_id=None if self.anchor is None else self.anchor["actor_id"],
            decision_frame_id=frame["frame_id"],decision_timestamp_s=frame["timestamp_s"],
            maximum_observed_evidence_age_s=self.maximum_age_s,
            scope="controlled stationary fixture only; persisted rear-clear assertion is exclusion, never observed target geometry")
        def result(mode,reason=None,obstacle=None):
            bindings={} if self.observed is None else dict(self.observed["bindings"])
            if self.certificate is not None:bindings.update(self.certificate["source_bindings"])
            return dict(mode=mode,reason=reason,obstacle=obstacle,bindings=dict(bindings),
                audit=dict(base,mode=mode,reason=reason,certificate=self.certificate,
                    observed_proof_sha256=None if self.observed is None else self.observed["proof_sha256"]))
        if self.failure:return result("unsupported",self.failure)
        if self.last_frame!=frame["frame_id"]:return result("unsupported","current_actor_continuity_not_checked")
        if self.observed is None:return result("unsupported","static_obstacle_never_visually_supported")
        obstacle=self.observed["obstacle"]
        age=max(frame["timestamp_s"]-e.timestamp_s for e in obstacle.evidence_frames)
        fresh=0<=age<=self.maximum_age_s+1e-8 and all(e.frame_id<=frame["frame_id"] for e in obstacle.evidence_frames)
        position=np.asarray(position_world,float)
        direction=np.array([math.cos(yaw_rad),math.sin(yaw_rad)])
        margin=BehaviorConfig().obstacle_clearance(self.category)
        if margin is None:raise ValueError("unsupported static obstacle category")
        rearmost_body=float(ego.corners()[:,0].min())
        foremost_obstacle=float(((obstacle.polygon()-position[:2])@direction).max())
        behind=foremost_obstacle+margin<rearmost_body-1e-6
        if self.certificate is None and fresh and behind:
            self.certificate=dict(issued_frame_id=frame["frame_id"],issued_timestamp_s=frame["timestamp_s"],
                observed_proof_sha256=self.observed["proof_sha256"],source_bindings=dict(self.observed["bindings"]),
                actor_id=self.anchor["actor_id"],actual_route_station_m=self.last_station,
                position_world=position.tolist(),yaw_rad=float(yaw_rad),observed_polygon_xy=obstacle.polygon().tolist(),
                rearmost_ego_coordinate_m=rearmost_body,foremost_obstacle_coordinate_m=foremost_obstacle,
                required_clearance_m=margin,geometry_source="retained fresh raw-camera obstacle proof",
                continuation_requires="same stationary actor, no reverse progress, observed envelope remains behind")
        if self.certificate is not None and behind:return result("rear_clear")
        if fresh:return result("current" if self.observed["observed_at"]["frame_id"]==frame["frame_id"] else "retained",obstacle=obstacle)
        return result("unsupported","static_obstacle_observation_expired_without_rear_clearance")


def verified_member(directory, relative, inventory):
    path = (directory/relative).resolve()
    if directory not in path.parents or relative not in inventory:
        raise ValueError("capture member is outside its bound inventory")
    if sha256_file(path) != inventory[relative]:
        raise ValueError("capture member hash mismatch: "+relative)
    return path


def replay_actor(directory, frames, camera, fixture, category, inventory):
    observations, states = [], []
    for row in frames:
        meta = row["cameras"][camera]
        payload = verified_member(directory, meta["payload"], inventory)
        if meta["payload_sha256"] != inventory[meta["payload"]]:
            raise ValueError("camera frame hash differs from capture inventory")
        stamps = {k:FrameStamp(**v) for k,v in meta["modality_stamps"].items()}
        stamp = stamps["rgb"]
        with np.load(payload,allow_pickle=False) as arrays:
            observations.append(Observation(stamp=stamp,modality_stamps=stamps,
                **{name:arrays[name].copy() for name in ("rgb","depth_m","semantic_ids","instance_ids",
                    "packed","intrinsics","camera_from_world","model_to_source","instance_bgra")},
                scene_actor_ids=tuple(row["scene_actor_ids"]),depth_mode=fixture[camera]["depth_mode"],
                depth_fixture_sha256=inventory["depth-fixture.json"],depth_qualified=fixture[camera]["qualified"]))
        actor = row["obstacle"]
        states.append(ActorState(stamp=stamp, **{k:actor[k] for k in
            ("actor_id","position_world_m","yaw_rad","velocity_world_mps","yaw_rate_rps")}))
    actor = frames[-1]["obstacle"]
    polygon = convex_hull(np.asarray(actor["bounding_box"]["vertices_world_xyz"])[:,:2])
    proof = build_static_proof(observations,states,actor_id=actor["actor_id"],instance_id=actor["actor_id"],
        category=category,conservative_footprint_world_xy=polygon,decision=observations[-1].stamp)
    return proof, tuple(EvidenceFrame(o.stamp.frame_id,o.stamp.timestamp_s,o.stamp.camera) for o in observations)


def passing_observations(directory, frames, inventory):
    """Load raw camera observations and their exact independent scene records."""
    from .free_space import load_saved_observation
    observations=[];scenes=[];bindings={}
    for row in frames:
        scene=row.get('controlled_scene')
        if not isinstance(scene,dict):raise ValueError('controlled passing scene inventory missing')
        for camera in ('narrow','wide'):
            meta=row['cameras'][camera];relative=meta['payload']
            if meta['payload_sha256']!=inventory.get(relative):raise ValueError('passing frame inventory mismatch')
            observations.append(load_saved_observation(verified_member(directory,relative,inventory),
                payload_sha256=inventory[relative],stamp=FrameStamp(row['frame_id'],row['timestamp_s'],camera),
                modality_stamps=meta['modality_stamps'],depth_fixture=verified_member(directory,'depth-fixture.json',inventory),
                depth_fixture_sha256=inventory['depth-fixture.json'],scene_actor_ids=row['scene_actor_ids']))
            scenes.append(scene);bindings[relative]=inventory[relative]
    for name in ('evidence-frames.jsonl','controlled-scene.json','depth-fixture.json'):
        verified_member(directory,name,inventory);bindings[name]=inventory[name]
    return observations,scenes,bindings


def observed_pass_permission(directory,frames,inventory,road,obstacles,destination,index):
    """A pixel-supported permission for a controlled static car, never a map gap."""
    from .free_space import build_free_space_proof,observed_passing_space
    observations,scenes,bindings=passing_observations(directory,frames,inventory)
    decision=FrameStamp(frames[-1]['frame_id'],frames[-1]['timestamp_s'],'narrow')
    proof=build_free_space_proof(observations,road,decision=decision,controlled_scenes=scenes,
        obstacle_ids=tuple(o.identity for o in obstacles),opposing_width_m=1.)
    path=destination/f'free-space-{index:05d}.json';write_json(path,proof)
    if not proof['supported']:return None,{},proof['reasons']
    permission=observed_passing_space(proof,road)
    bindings[path.relative_to(directory).as_posix()]=sha256_file(path)
    bindings['controlled_scene']=permission.controlled_scene_sha256
    bindings['rendered_free_space']=permission.rendered_free_space_sha256
    return permission,bindings,[]


def reprocess(directory):
    directory = Path(directory).resolve()
    capture_path = directory/"capture-report.json"
    report = json.loads(capture_path.read_text(encoding="utf-8"))
    if (report.get("scope") != "physical_teacher_capture_pending_observed_admission"
            or report.get("accepted_labels") != 0 or report.get("training_ready") is not False):
        raise ValueError("only explicit physical candidate captures can be replayed")
    inventory = report["files"]
    task = json.loads(verified_member(directory,"candidate-task.json",inventory).read_text(encoding="utf-8"))
    telemetry = json.loads(verified_member(directory,"telemetry.json",inventory).read_text(encoding="utf-8"))
    fixture_path = verified_member(directory,"depth-fixture.json",inventory)
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not all(fixture[c].get("qualified") is True for c in ("narrow","wide")):
        raise ValueError("capture lacks qualified aligned depth encoding")
    evidence = [json.loads(line) for line in verified_member(directory,"evidence-frames.jsonl",inventory).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not evidence or not telemetry:
        raise ValueError("empty capture evidence or telemetry")
    destination = directory/"replays"/uuid.uuid4().hex
    destination.mkdir(parents=True,exist_ok=False)
    edges = task["route"]["unmarked_road"]
    reference = LaneReference(edges["road_left_edge_xyz"],edges["road_right_edge_xyz"],task["route"]["arc_m"])
    curb_geometry=None;scene_bindings={};survey_path=None;survey_sha=None
    if edges.get("source_bindings"):
        from .curb_evidence import surveyed_reference
        retained=report.get("retained_scene_bindings",{})
        for name,binding in edges["source_bindings"].items():
            member=retained.get(name)
            if member is None or member["sha256"]!=binding["sha256"]:
                raise ValueError("surveyed scene source not retained in capture: "+name)
            verified_member(directory,member["path"],inventory)
            scene_bindings[member["path"]]=member["sha256"]
        original=np.asarray(edges.get("original_source_arc_m",edges.get("source_collision_route_stations_m",[])),float)
        if len(original)<2 or np.any(np.diff(original)<=0):
            raise ValueError("reverse surveyed curb replay requires explicit traversal-side transform")
        survey=retained["survey_replay"];raw=retained["raw_survey"]
        survey_path=directory/survey["path"];survey_sha=survey["sha256"]
        curb_geometry=surveyed_reference(survey_path,survey_sha,raw_survey_path=directory/raw["path"])
    source_records, recent_road, recent_actor, decisions, records = {}, deque(), deque(maxlen=3), [], []
    specifications=task["unmarked_behavior"]["obstacles"]
    if len(specifications)>1:raise ValueError("bounded behavior replay supports one controlled static fixture")
    actor_history=StaticObstacleHistory(specifications[0]["kind"]) if specifications else None
    failures = Counter()
    for number, frame in enumerate(evidence):
        index = frame["index"]
        row = telemetry[index]
        if (row["index"] != index or row["frame_id"] != frame["frame_id"]
                or abs(row["timestamp"]-frame["timestamp_s"])>1e-6):
            raise ValueError("camera and physical telemetry do not align")
        current_s,_ = reference.project(row["position_world"][:2])
        # Raw evidence and actor continuity remain 4 Hz. Expensive independent
        # curb geometry is replayed at 2 Hz, exactly including every label.
        for camera, meta in (frame["cameras"].items() if road_evidence_due(index) else ()):
            relative = meta["payload"]
            if meta["payload_sha256"] != inventory.get(relative):
                raise ValueError("frame payload has conflicting inventory binding")
            kwargs=dict(payload=directory/relative,payload_sha256=inventory[relative],
                stamp=FrameStamp(frame["frame_id"],frame["timestamp_s"],camera),modality_stamps=meta["modality_stamps"],
                semantic_sensor_stamp=meta["separate_semantic_sensor_stamp"],depth_fixture=fixture_path,
                depth_fixture_sha256=inventory["depth-fixture.json"])
            if curb_geometry is not None:
                from .curb_evidence import measure_curb_road_frame
                source_station=curb_geometry[0].project(row["position_world"][:2])[0]
                proof=measure_curb_road_frame(**kwargs,survey=survey_path,survey_sha256=survey_sha,
                    current_s=source_station,surveyed_geometry=curb_geometry)
            else:
                proof=measure_road_frame(**kwargs,reference=reference,current_s=current_s)
            recent_road.append(proof)
            source_records[proof.proof_sha256] = dict(relative=relative,sha256=inventory[relative])
        while recent_road and recent_road[0].stamp.timestamp_s < frame["timestamp_s"]-5.-1e-6:
            recent_road.popleft()
        recent_actor.append(frame)
        box = frame["ego"]["bounding_box"]
        ego = EgoFootprint(2*box["extent_m"][0],2*box["extent_m"][1],tuple(box["center_actor_m"][:2]),math.radians(box["rotation_degrees"][2]))
        actor_decision=None
        if actor_history is not None:
            actor_history.update_actual(frame,current_s)
            # Acquire history even when no label can yet be retained. Requiring
            # road assembly first would discard the last visible obstacle proof.
            if len(recent_actor)==3 and index%10==0 and not actor_history.failure:
                for camera in ("narrow","wide"):
                    proof,refs=replay_actor(directory,list(recent_actor),camera,fixture,actor_history.category,inventory)
                    if not proof["supported"]:continue
                    proof_path=destination/f"obstacle-seen-{index:05d}.json"
                    write_json(proof_path,proof)
                    proof_bindings={proof_path.relative_to(directory).as_posix():sha256_file(proof_path)}
                    for source in recent_actor:
                        meta=source["cameras"][camera];proof_bindings[meta["payload"]]=meta["payload_sha256"]
                    actor_history.observe(proof,refs,proof_bindings,current_frame=frame)
                    break
            actor_decision=actor_history.decide(frame=frame,ego=ego,position_world=row["position_world"],yaw_rad=row["rotation_world"][2])
        # Only decisions with a genuine five-second recurrent prefix and a
        # ten-second retained suffix are candidates. This does not encode speed.
        if index%10 or frame["timestamp_s"]-telemetry[0]["timestamp"] < 5.-1e-6 or telemetry[-1]["timestamp"]-frame["timestamp_s"] < 10.-1e-6:
            continue
        decision = dict(label_index=index,label_frame_id=frame["frame_id"],timestamp_s=frame["timestamp_s"],eligible=False,reasons=[])
        assembly = assemble_observed_road(recent_road,decision=FrameStamp(frame["frame_id"],frame["timestamp_s"],"narrow"),position_world=row["position_world"])
        if assembly.road is None:
            decision["reasons"].extend(assembly.reasons)
        else:
            audit_path = destination/f"road-{index:05d}.json"
            write_json(audit_path,dict(assembly.audit,road_evidence_stride_ticks=ROAD_EVIDENCE_STRIDE_TICKS))
            bindings = {audit_path.relative_to(directory).as_posix():sha256_file(audit_path),"depth-fixture.json":inventory["depth-fixture.json"]}
            bindings.update(scene_bindings)
            for sha in assembly.audit["used_frame_proofs"]:
                item = source_records[sha]; bindings[item["relative"]] = item["sha256"]
            obstacles = []
            if actor_decision is not None:
                decision["obstacle_evidence_mode"]=actor_decision["mode"]
                actor_audit=destination/f"obstacle-state-{index:05d}.json"
                write_json(actor_audit,actor_decision["audit"])
                bindings[actor_audit.relative_to(directory).as_posix()]=sha256_file(actor_audit)
                bindings.update(actor_decision["bindings"])
                # These raw actor/ego histories establish exclusion continuity;
                # they are independent QA, never visible obstacle geometry.
                for source in ("telemetry.json","evidence-frames.jsonl"):
                    bindings[source]=inventory[source]
                if actor_decision["mode"]=="unsupported":decision["reasons"].append(actor_decision["reason"])
                elif actor_decision["obstacle"] is not None:obstacles.append(actor_decision["obstacle"])
            elif frame["obstacle"] is not None:
                decision["reasons"].append("unplanned_obstacle_in_clear_road_case")
            if not decision["reasons"]:
                passing_space=None
                if (obstacles and all(o.kind=='parked_vehicle' for o in obstacles)
                        and task['unmarked_behavior'].get('passing_space_policy')
                        =='observed_clear_surface_during_static_parked_avoidance_only'):
                    try:
                        passing_space,passing_bindings,passing_reasons=observed_pass_permission(directory,
                            list(recent_actor),inventory,assembly.road,obstacles,destination,index)
                        bindings.update(passing_bindings)
                        decision['passing_evidence_reasons']=passing_reasons
                    except (ValueError,KeyError) as error:
                        decision['passing_evidence_reasons']=[str(error)]
                    # Missing permission leaves the ordinary right-half target
                    # evaluator in force. It can still admit a valid nominal
                    # path, but cannot cross an unobserved opposing strip.
                record = build_record(assembly.road,obstacles,ego,label_frame_id=frame["frame_id"],
                    label_timestamp_s=frame["timestamp_s"],evidence_bindings=bindings,
                    ego_position_world=row["position_world"],ego_yaw_rad=row["rotation_world"][2],
                    target_mode="observed_future_geometry",passing_space=passing_space)
                if not record["moving_supervision"]:
                    decision["reasons"].extend(record["reasons"])
                else:
                    decision.update(eligible=True,record_sha256=record["record_sha256"])
                    records.append(dict(label_index=index,record=record))
        failures.update(decision["reasons"])
        decisions.append(decision)
        if len(decisions)%10==0:
            print(json.dumps(dict(event="unmarked_observation_replay",frames=number+1,total=len(evidence),candidate_records=len(records),reasons=dict(failures))),flush=True)
    write_json(destination/"behavior-records.json",records)
    result = dict(schema="unmarked_candidate_observation_replay_v1",capture=str(directory),
        capture_report_sha256=sha256_file(capture_path),training_ready=False,accepted_labels=0,
        road_evidence_stride_ticks=ROAD_EVIDENCE_STRIDE_TICKS,raw_evidence_stride_ticks=5,
        road_evidence_frame_indices=[f["index"] for f in evidence if road_evidence_due(f["index"])],
        road_replay_scope="2 Hz label-aligned raw camera geometry; 4 Hz raw actor history retained; not every raw road frame measured",
        candidate_record_count=len(records),decisions=decisions,rejection_reasons=dict(failures),
        physical_capture_report=report.get("physical"),physical_replay_complete=False,
        remaining=["independent_physical_replay","geometry_split_assignment","complete_behavior_coverage","immutable_training_admission"],
        records_file="behavior-records.json",records_sha256=sha256_file(destination/"behavior-records.json"))
    write_json(destination/"replay-report.json",result)
    return result,destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture",type=Path)
    args = parser.parse_args()
    result,destination = reprocess(args.capture)
    print(json.dumps(dict(report=str(destination/"replay-report.json"),candidate_record_count=result["candidate_record_count"],accepted_labels=0)),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
