from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from lane_dataset.geometry import EvidenceFrame
from lane_behavior.reprocess import StaticObstacleHistory
from lane_behavior.target import EgoFootprint
from lane_dataset.config import sha256_file,write_json
from test_behavior_planned import simulator_road


def frame(number=10,time=1.,*,actor_id=7,speed=0.):
    return dict(frame_id=number,timestamp_s=time,obstacle=dict(actor_id=actor_id,
        position_world_m=[10.,3.,0.],velocity_world_mps=[speed,0.,0.],yaw_rad=0.,yaw_rate_rps=0.))


def observed(tracker, *, position=5.):
    current=frame()
    tracker.update_actual(current,position)
    proof=dict(supported=True,actor_id=7,category="parked_vehicle",observations=[dict(
        conservative_qa_footprint_world_xy=[[8.,2.],[12.,2.],[12.,4.],[8.,4.]])])
    refs=(EvidenceFrame(8,.5,"wide"),EvidenceFrame(9,.75,"wide"),EvidenceFrame(10,1.,"wide"))
    bindings={"raw/proof.npz":"a"*64,"proof.json":"b"*64}
    tracker.observe(proof,refs,bindings,current_frame=current)
    return current,proof,refs,bindings


def decide(tracker,current,position=5.,yaw=0.):
    return tracker.decide(frame=current,ego=EgoFootprint(),position_world=[position,3.,0.],yaw_rad=yaw)


def test_current_and_causal_retained_proof_keep_original_observation_stamps():
    tracker=StaticObstacleHistory("parked_vehicle")
    current,_,refs,bindings=observed(tracker)
    assert decide(tracker,current)["mode"]=="current"
    later=frame(20,3.);tracker.update_actual(later,8.)
    result=decide(tracker,later,8.)
    assert result["mode"]=="retained"
    assert result["obstacle"].evidence_frames==refs and result["bindings"]==bindings
    assert result["audit"]["certificate"] is None


def test_expired_observation_cannot_manufacture_target_or_rear_clearance():
    tracker=StaticObstacleHistory("parked_vehicle");observed(tracker)
    later=frame(150,8.);tracker.update_actual(later,20.)
    result=decide(tracker,later,20.)
    assert result["mode"]=="unsupported" and result["obstacle"] is None
    assert "expired_without_rear_clearance" in result["reason"]
    assert result["audit"]["certificate"] is None


def test_fresh_rear_clearance_persists_as_exclusion_and_retains_original_raw_bindings():
    tracker=StaticObstacleHistory("parked_vehicle");_,_,_,bindings=observed(tracker)
    passed=frame(60,4.);tracker.update_actual(passed,16.)
    first=decide(tracker,passed,16.)
    assert first["mode"]=="rear_clear" and first["obstacle"] is None
    cert=deepcopy(first["audit"]["certificate"])
    assert cert["foremost_obstacle_coordinate_m"]+cert["required_clearance_m"]<cert["rearmost_ego_coordinate_m"]
    later=frame(300,20.);tracker.update_actual(later,55.)
    result=decide(tracker,later,55.)
    assert result["mode"]=="rear_clear" and result["obstacle"] is None
    assert result["audit"]["certificate"]==cert and result["bindings"]==bindings


@pytest.mark.parametrize("failure",["reverse","moved","identity","turned_back"])
def test_rear_clearance_is_not_a_dynamic_actor_or_reversal_exemption(failure):
    tracker=StaticObstacleHistory("parked_vehicle");observed(tracker)
    passed=frame(60,4.);tracker.update_actual(passed,16.)
    assert decide(tracker,passed,16.)["mode"]=="rear_clear"
    later=frame(300,20.,actor_id=8 if failure=="identity" else 7,speed=.3 if failure=="moved" else 0.)
    position=15. if failure=="reverse" else 55.
    tracker.update_actual(later,position)
    result=decide(tracker,later,position,np.pi if failure=="turned_back" else 0.)
    assert result["mode"]=="unsupported" and result["obstacle"] is None


def test_never_seen_actor_is_not_omitted_just_because_simulator_says_behind():
    tracker=StaticObstacleHistory("parked_vehicle")
    current=frame();tracker.update_actual(current,20.)
    result=decide(tracker,current,20.)
    assert result["mode"]=="unsupported" and result["reason"]=="static_obstacle_never_visually_supported"


def test_retention_refuses_future_stale_unqualified_and_wrong_actor_proofs():
    tracker=StaticObstacleHistory("parked_vehicle")
    current,proof,refs,bindings=observed(tracker)
    for altered_refs in ((EvidenceFrame(11,1.1,"wide"),),(EvidenceFrame(1,-5.,"wide"),)):
        with pytest.raises(ValueError,match="causal and fresh"):
            tracker.observe(proof,altered_refs,bindings,current_frame=current)
    for changed in ({**proof,"supported":False},{**proof,"actor_id":8},{**proof,"category":"pedestrian"}):
        with pytest.raises(ValueError):tracker.observe(changed,refs,bindings,current_frame=current)


def test_oldest_observation_limits_retention_instead_of_last_observation():
    tracker=StaticObstacleHistory("parked_vehicle");observed(tracker)
    later=frame(110,5.6);tracker.update_actual(later,7.)
    assert decide(tracker,later,7.)["mode"]=="unsupported"


def test_bbox_overhang_and_class_clearance_are_required_before_exclusion():
    tracker=StaticObstacleHistory("parked_vehicle");observed(tracker)
    later=frame(60,4.);tracker.update_actual(later,14.5)
    result=decide(tracker,later,14.5)
    assert result["mode"]=="retained"  # Rear is 12.25 m; required obstacle margin reaches 12.6 m.
    assert result["audit"]["certificate"] is None


def test_small_reverse_steps_cannot_accumulate_past_certificate_tolerance():
    tracker=StaticObstacleHistory("parked_vehicle");observed(tracker)
    passed=frame(60,4.);tracker.update_actual(passed,16.)
    assert decide(tracker,passed,16.)["mode"]=="rear_clear"
    one=frame(65,4.25);tracker.update_actual(one,15.96)
    two=frame(70,4.5);tracker.update_actual(two,15.92)
    assert decide(tracker,two,15.92)["mode"]=="unsupported"


def test_road_replay_is_label_aligned_2hz_but_actor_proof_keeps_all_4hz_frames(tmp_path,monkeypatch):
    import lane_behavior.reprocess as module
    road=simulator_road(length=100.)
    task=dict(route=dict(arc_m=road.s.tolist(),unmarked_road=dict(road_left_edge_xyz=road.left_xyz.tolist(),
        road_right_edge_xyz=road.right_xyz.tolist())),unmarked_behavior=dict(obstacles=[dict(kind="parked_vehicle")]))
    rows=[dict(index=i,frame_id=1000+i,timestamp=i*.05,position_world=[5.+i*.125,3.,0.],rotation_world=[0.,0.,0.]) for i in range(301)]
    evidence=[];inventory={}
    for i in range(0,301,5):
        item=frame(1000+i,i*.05);item["index"]=i
        item["ego"]=dict(bounding_box=dict(extent_m=[2.25,.9,1.],center_actor_m=[0.,0.,0.],rotation_degrees=[0.,0.,0.]))
        item["cameras"]={}
        for camera in ("narrow","wide"):
            relative=f"raw/{i}-{camera}.npz";inventory[relative]="a"*64
            stamp=dict(frame_id=1000+i,timestamp_s=i*.05,camera=camera)
            item["cameras"][camera]=dict(payload=relative,payload_sha256="a"*64,
                modality_stamps={key:stamp for key in ("rgb","depth","packed","semantic","instance")},separate_semantic_sensor_stamp=stamp)
        evidence.append(item)
    for name,value in (("candidate-task.json",task),("telemetry.json",rows),
            ("depth-fixture.json",{camera:dict(qualified=True) for camera in ("narrow","wide")})):
        write_json(tmp_path/name,value);inventory[name]=sha256_file(tmp_path/name)
    (tmp_path/"evidence-frames.jsonl").write_text("\n".join(json.dumps(f) for f in evidence))
    inventory["evidence-frames.jsonl"]=sha256_file(tmp_path/"evidence-frames.jsonl")
    write_json(tmp_path/"capture-report.json",dict(scope="physical_teacher_capture_pending_observed_admission",
        accepted_labels=0,training_ready=False,files=inventory))
    road_calls=[];actor_calls=[];assemblies=[]
    def measure(**kwargs):
        stamp=kwargs["stamp"];road_calls.append((stamp.frame_id-1000,stamp.camera))
        return SimpleNamespace(stamp=stamp,proof_sha256=str(stamp.frame_id)+stamp.camera)
    def actor(directory,frames,camera,fixture,category,inventory):
        actor_calls.append(tuple(f["index"] for f in frames))
        return dict(supported=False),()
    def assemble(proofs,*,decision,position_world):
        frames=[p.stamp for p in proofs];assemblies.append(decision.frame_id-1000)
        assert any(p.frame_id==decision.frame_id for p in frames)
        assert all(0<=decision.timestamp_s-p.timestamp_s<=5.+1e-6 for p in frames)
        return SimpleNamespace(road=None,reasons=("synthetic_no_road",))
    monkeypatch.setattr(module,"measure_road_frame",measure)
    monkeypatch.setattr(module,"replay_actor",actor)
    monkeypatch.setattr(module,"assemble_observed_road",assemble)
    result,_=module.reprocess(tmp_path)
    assert road_calls==[(i,camera) for i in range(0,301,10) for camera in ("narrow","wide")]
    assert actor_calls and all(b-a==c-b==5 for a,b,c in actor_calls)
    assert (0,5,10) in actor_calls and (290,295,300) in actor_calls
    assert assemblies==[100] and result["road_evidence_stride_ticks"]==10
    assert result["raw_evidence_stride_ticks"]==5 and result["accepted_labels"]==0


def saved_passing_scene(tmp_path):
    """Actual encoded synthetic sensor payloads, no preaccepted claim flags."""
    from dataclasses import asdict
    from lane_dataset.visibility import decode_carla_depth_bgra
    from test_behavior_free_space import scene,history
    from lane_behavior.target import ObservedObstacle
    observations,scenes,road=history();inventory={};frames=[]
    fixture={c:dict(depth_mode='forward',qualified=True) for c in ('narrow','wide')}
    write_json(tmp_path/'depth-fixture.json',fixture)
    inventory['depth-fixture.json']=sha256_file(tmp_path/'depth-fixture.json')
    for i in range(3):
        cameras={}
        for camera in ('narrow','wide'):
            o,controlled=scene(20+5*i,.25*i,camera)
            encoded=np.rint(o.depth_m.astype(float)/1000.*(256**3-1)).astype(np.uint32)
            raw=np.stack(((encoded>>16)&255,(encoded>>8)&255,encoded&255,np.full_like(encoded,255)),axis=-1).astype(np.uint8)
            arrays={name:getattr(o,name) for name in ('rgb','semantic_ids','instance_ids','instance_bgra',
                'packed','intrinsics','camera_from_world','model_to_source')}
            arrays.update(depth_m=decode_carla_depth_bgra(raw).astype(np.float32),depth_bgra=raw,
                rgb_bgra=np.concatenate([o.rgb[...,[2,1,0]],np.full((*o.rgb.shape[:2],1),255,np.uint8)],axis=-1))
            name=f'raw-{i}-{camera}.npz';np.savez_compressed(tmp_path/name,**arrays)
            inventory[name]=sha256_file(tmp_path/name)
            cameras[camera]=dict(payload=name,payload_sha256=inventory[name],
                modality_stamps={k:asdict(v) for k,v in o.modality_stamps.items()})
        frames.append(dict(index=i*5,frame_id=20+5*i,timestamp_s=.25*i,cameras=cameras,
            scene_actor_ids=list(o.scene_actor_ids),controlled_scene=controlled))
    (tmp_path/'evidence-frames.jsonl').write_text('\n'.join(json.dumps(f) for f in frames))
    write_json(tmp_path/'controlled-scene.json',scenes[0])
    for name in ('evidence-frames.jsonl','controlled-scene.json'):inventory[name]=sha256_file(tmp_path/name)
    obstacle=ObservedObstacle('2',[[9.,2.8],[11.,2.8],[11.,4.],[9.,4.]],'parked_vehicle',
        tuple(EvidenceFrame(f['frame_id'],f['timestamp_s'],'wide') for f in frames),'static')
    return frames,inventory,road,(obstacle,)


def test_saved_passing_permission_uses_raw_frames_and_keeps_two_logical_digests(tmp_path):
    from lane_behavior.reprocess import observed_pass_permission
    frames,inventory,road,obstacles=saved_passing_scene(tmp_path)
    destination=tmp_path/'replay';destination.mkdir()
    permission,bindings,reasons=observed_pass_permission(tmp_path,frames,inventory,road,obstacles,destination,100)
    assert not reasons and permission is not None
    assert permission.allowed_intervals_m==((6.,14.),)
    assert bindings['rendered_free_space']==permission.rendered_free_space_sha256
    assert bindings['controlled_scene']==permission.controlled_scene_sha256
    assert bindings['controlled_scene']!=inventory['controlled-scene.json']
    assert all(meta['payload'] in bindings for f in frames for meta in f['cameras'].values())
    assert bindings['replay/free-space-00100.json']==sha256_file(destination/'free-space-00100.json')
    frames[-1]['controlled_scene']['actors'][1]['state']['velocity_world_mps'][0]=1.
    with pytest.raises(ValueError,match='moving'):
        observed_pass_permission(tmp_path,frames,inventory,road,obstacles,destination,110)
