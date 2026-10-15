from copy import deepcopy
from dataclasses import asdict
from itertools import product
import json

import numpy as np
import pytest

from lane_dataset.config import sha256_file, write_json
from lane_dataset.splits import GeometryGroup, GeometrySection, RoadInterval, SplitAssignment
from lane_dataset.geometry import EvidenceFrame, ObservableReference
from lane_behavior.admission import (audit_candidate, reconstruct_teacher, verify_actual_envelopes,
    verify_inventory, verify_label_recovery, verify_split, verify_retention, _publish, current_split_assignment)
from lane_behavior.planned import PlannedObstacle, planned_scene_geometry
from lane_behavior.records import build_record
from lane_behavior.target import ObservedRoad, ObservedObstacle
from lane_behavior.evidence import pack_model_rgb,decode_carla_instance_bgra
from test_behavior_planned import simulator_road, simulator_obstacle


def test_teacher_reconstruction_rejects_solution_and_source_changes():
    target=planned_scene_geometry(simulator_road(length=30.))
    document=target.document()
    assert reconstruct_teacher(document).contract_sha256==target.contract_sha256
    for section,key,value in (("solution","feasible",False),("source","source_metadata",{"changed":True})):
        altered=deepcopy(document);altered[section][key]=value
        with pytest.raises(ValueError,match="replay exactly"):reconstruct_teacher(altered)


def test_teacher_reconstruction_preserves_optional_tracking_reserve():
    target=planned_scene_geometry(simulator_road(length=30.),obstacle_tracking_reserve_m=.1)
    document=target.document()
    replay=reconstruct_teacher(document)
    assert replay.contract_sha256==target.contract_sha256
    assert replay.obstacle_tracking_reserve_m==.1
    altered=deepcopy(document)
    altered["source"]["obstacle_tracking_reserve_m"]+=.01
    with pytest.raises(ValueError,match="replay exactly"):
        reconstruct_teacher(altered)


def test_teacher_reconstruction_preserves_explicit_planned_passing_permission():
    from lane_behavior.passing import PlannedPassingSpace,reference_fingerprint
    road=simulator_road();obstacle=simulator_obstacle(y=4.3)
    permission=PlannedPassingSpace(reference_fingerprint(road),((0.,80.),),1.,(obstacle.identity,),"c"*64)
    target=planned_scene_geometry(road,[obstacle],passing_space=permission)
    document=target.document();replay=reconstruct_teacher(document)
    assert replay.contract_sha256==target.contract_sha256
    assert replay.passing_space==permission
    changed=deepcopy(document);changed["source"]["passing_space"]["controlled_scene_sha256"]="d"*64
    with pytest.raises(ValueError,match="replay exactly"):reconstruct_teacher(changed)


def passing_record_fixture(tmp_path):
    from test_behavior_reprocess import saved_passing_scene
    from lane_behavior.reprocess import observed_pass_permission
    frames,inventory,road,obstacles=saved_passing_scene(tmp_path)
    destination=tmp_path/'replay';destination.mkdir()
    permission,bindings,reasons=observed_pass_permission(tmp_path,frames,inventory,road,obstacles,destination,100)
    assert not reasons
    record=build_record(road,obstacles,label_frame_id=30,label_timestamp_s=.5,
        evidence_bindings=bindings,ego_position_world=[0.,2.,0.],ego_yaw_rad=0.,
        target_mode='observed_future_geometry',passing_space=permission)
    return record,dict(files=inventory),frames


def test_passing_admission_replays_raw_proof_instead_of_treating_claims_as_paths(tmp_path):
    from lane_behavior.admission import verify_record_evidence
    from lane_behavior.loader import evidence_file_bindings,AdmittedBehaviorSequence
    record,report,frames=passing_record_fixture(tmp_path)
    verified=verify_record_evidence(tmp_path,record,100,report,frames)
    assert verified['method']=='independent_saved_free_surface_replay'
    assert verified['proof_file']=='replay/free-space-00100.json'
    files=evidence_file_bindings(record)
    assert 'controlled_scene' not in files and 'rendered_free_space' not in files
    loader=object.__new__(AdmittedBehaviorSequence);loader.root=tmp_path
    loader._validate_passing_admission(record,dict(passing_evidence_replay=verified),files)
    with pytest.raises(ValueError,match='independent raw proof'):
        loader._validate_passing_admission(record,{},files)
    changed=deepcopy(record);changed['evidence_bindings']['controlled_scene']='a'*64
    with pytest.raises(ValueError,match='differ'):
        evidence_file_bindings(changed)
    changed=deepcopy(record);changed.pop('passing_space')
    with pytest.raises(ValueError,match='require'):
        evidence_file_bindings(changed)


def test_passing_admission_rejects_raw_inventory_changes_despite_fresh_file_hash(tmp_path):
    from lane_behavior.admission import verify_record_evidence
    record,report,frames=passing_record_fixture(tmp_path)
    frames[-1]['controlled_scene']['actors'][1]['state']['yaw_rad']+=.1
    with pytest.raises(ValueError,match='rotated|differs'):
        verify_record_evidence(tmp_path,record,100,report,frames)


def test_passing_admission_does_not_skip_unknown_named_bindings(tmp_path):
    from lane_behavior.admission import verify_record_evidence
    record,report,frames=passing_record_fixture(tmp_path)
    record['evidence_bindings']['invented_logical_claim']='f'*64
    with pytest.raises(FileNotFoundError):verify_record_evidence(tmp_path,record,100,report,frames)


def actual_rows(target, count=11):
    rows=[];evidence=[]
    for i in range(count):
        row=dict(index=i,frame_id=100+i,timestamp=i*.05,position_world=[10+i*.05,3.,0.],rotation_world=[0.,0.,0.])
        rows.append(row)
        if i%5:continue
        ego=dict(position_world_m=row["position_world"],yaw_rad=0.,bounding_box=dict(
            extent_m=[target.ego.length_m/2,target.ego.width_m/2,1.],center_actor_m=[*target.ego.center_xy,0.],
            rotation_degrees=[0.,0.,np.degrees(target.ego.yaw_rad)]))
        obstacle=None
        if target.obstacles:
            polygon=target.obstacles[0].polygon();center=np.r_[polygon.mean(0),0.]
            extent=np.r_[np.ptp(polygon,axis=0)/2,1.]
            transform=np.eye(4);transform[:3,3]=center
            vertices=np.asarray(list(product((-1.,1.),repeat=3)))*extent+center
            obstacle=dict(actor_id=7,position_world_m=[40.,4.3,0.],yaw_rad=0.,velocity_world_mps=[0.,0.,0.],yaw_rate_rps=0.,
                transform_world=transform.tolist(),bounding_box=dict(vertices_world_xyz=vertices.tolist(),
                    extent_m=extent.tolist(),center_actor_m=[0.,0.,0.],rotation_degrees=[0.,0.,0.]))
        evidence.append(dict(index=i,frame_id=row["frame_id"],timestamp_s=row["timestamp"],ego=ego,obstacle=obstacle))
    return rows,evidence


@pytest.fixture(scope="module")
def obstacle_target():
    obstacle=simulator_obstacle(y=4.3)
    return planned_scene_geometry(simulator_road(),[PlannedObstacle("7",obstacle.polygon(),obstacle.kind,"static")])


def test_actual_bbox_and_stationarity_are_replayed_not_trusted(obstacle_target):
    rows,evidence=actual_rows(obstacle_target)
    assert verify_actual_envelopes(obstacle_target,evidence,rows)["passed"]
    evidence[-1]["obstacle"]["bounding_box"]["vertices_world_xyz"][0][0]+=.1
    with pytest.raises(ValueError,match="fixed envelope"):verify_actual_envelopes(obstacle_target,evidence,rows)


@pytest.mark.parametrize("mutation,reason",[("velocity","not stationary"),("ego","ego envelope"),("missing","4 Hz"),("time","aligned")])
def test_actual_envelope_rejections(obstacle_target,mutation,reason):
    rows,evidence=actual_rows(obstacle_target)
    if mutation=="velocity":evidence[-1]["obstacle"]["velocity_world_mps"][0]=.2
    elif mutation=="ego":evidence[-1]["ego"]["bounding_box"]["extent_m"][1]+=.1
    elif mutation=="time":evidence[-1]["timestamp_s"]+=.1
    else:evidence.pop()
    with pytest.raises(ValueError,match=reason):verify_actual_envelopes(obstacle_target,evidence,rows)


def shift_obstacle(actor,dy=0.,angle=0.):
    matrix=np.asarray(actor['transform_world']);old=matrix.copy()
    rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    local=(np.asarray(actor['bounding_box']['vertices_world_xyz'])-old[:3,3])@old[:3,:3]
    matrix[:3,:3]=rotation;matrix[1,3]+=dy
    actor['position_world_m']=matrix[:3,3].tolist();actor['transform_world']=matrix.tolist();actor['yaw_rad']=angle
    actor['bounding_box']['vertices_world_xyz']=(local@rotation.T+matrix[:3,3]).tolist()


def test_actual_small_settling_drift_is_allowed_only_with_conservative_clearance(obstacle_target):
    rows,evidence=actual_rows(obstacle_target,101)
    for row in rows:row['position_world']=[35.+row['index']*.1,1.7,0.]
    for frame in evidence:
        frame['ego']['position_world_m']=rows[frame['index']]['position_world']
        fraction=frame['index']/100
        shift_obstacle(frame['obstacle'],-.02*fraction,np.radians(.4)*fraction)
    result=verify_actual_envelopes(obstacle_target,evidence,rows)
    sweep=result['actual_obstacle_sweep']
    assert sweep['passed'] and sweep['minimum_clearance_m']>=.6
    assert sweep['stationary_envelope']['maximum_translation_m']>.019
    assert sweep['stationary_envelope']['between_observation_guard_m']>.05
    # An apparent nominal clearance of .65m is insufficient once actual drift
    # and the bounded between-observation footprint are included.
    for row in rows:row['position_world'][1]=1.85
    for frame in evidence:frame['ego']['position_world_m']=rows[frame['index']]['position_world']
    with pytest.raises(ValueError,match='clearance below unchanged limit'):
        verify_actual_envelopes(obstacle_target,evidence,rows)


def test_stationary_reported_pose_cannot_leave_a_stale_saved_bbox(obstacle_target):
    rows,evidence=actual_rows(obstacle_target)
    actor=evidence[-1]['obstacle'];actor['position_world_m'][1]+=.02;actor['transform_world'][1][3]+=.02
    with pytest.raises(ValueError,match='fixed envelope disagrees'):
        verify_actual_envelopes(obstacle_target,evidence,rows)


def test_actual_obstacle_sweep_checks_between_ego_samples_and_pedestrian_margin():
    from lane_behavior.admission import _actual_obstacle_clearance
    from lane_behavior.target import EgoFootprint
    obstacle=simulator_obstacle(name='7',x=40.,y=3.,kind='pedestrian')
    target=planned_scene_geometry(simulator_road(),[obstacle],EgoFootprint())
    bound=dict(polygon_xy=obstacle.polygon().tolist(),between_observation_guard_m=.05)
    # Endpoints are clear but the swept body crosses the obstacle.
    rows=[dict(position_world=[x,3.,0.],rotation_world=[0.,0.,0.]) for x in (30.,50.)]
    with pytest.raises(ValueError,match='clearance below unchanged limit'):
        _actual_obstacle_clearance(target,rows,bound)
    rows=[dict(position_world=[x,.1,0.],rotation_world=[0.,0.,0.]) for x in (38.,42.)]
    with pytest.raises(ValueError,match='1.2'):_actual_obstacle_clearance(target,rows,bound)


def split_fixture(tmp_path, *, baseline_road="1",baseline_split="train",candidate_road="2",town="Town02"):
    source=tmp_path/(town+".xodr")
    source.write_text(f'<OpenDRIVE><road id="{candidate_road}" length="200" junction="-1"><link/></road></OpenDRIVE>')
    section=GeometrySection("existing",town,"straight_recovery",(RoadInterval(baseline_road,0,200),))
    baseline=dict(splits_frozen=True,groups=[asdict(GeometryGroup("old",(section,)))],
        assignments=[asdict(SplitAssignment("old",baseline_split,"straight_recovery",1,0))])
    frozen=tmp_path/"splits.json";write_json(frozen,baseline)
    write_json(tmp_path/"qualification.json",dict(split_binding=dict(path=str(frozen),sha256=sha256_file(frozen))))
    task=dict(native_unmarked=True,procedural=False,town=town,family="straight_recovery",seed=1,start=dict(road_id=candidate_road),
        route=dict(source_xodr=str(source),source_xodr_sha256=sha256_file(source),section=dict(
            direction="straight",road_intervals=[dict(road_id=candidate_road,start_m=30.,end_m=100.)])))
    approval=dict(candidate_task_sha256="a"*64,split="validation",frozen_split_sha256=sha256_file(frozen))
    return task,approval


def test_split_recomputed_against_full_road_not_just_capture_fragment(tmp_path):
    task,approval=split_fixture(tmp_path,baseline_road="2")
    # Different group/section IDs and disjoint apparent capture starts cannot
    # override the conservative full-road split claim.
    result=verify_split(tmp_path,task,"a"*64,approval)
    assert not result["passed"]
    assert any("geometry_leakage" in e for e in result["errors"])
    approval["split"]="train"
    result=verify_split(tmp_path,task,"a"*64,approval)
    assert result["passed"]
    interval=result["candidate_section"]["road_intervals"][0]
    assert interval["start_m"]==0 and interval["end_m"]==200


def test_refresh_keeps_task_and_split_and_still_rejects_geometry_overlap(tmp_path):
    task,assignment=split_fixture(tmp_path,baseline_road="2")
    original=dict(assignment);assignment["frozen_split_sha256"]="f"*64
    refreshed=current_split_assignment(tmp_path,assignment)
    assert refreshed["candidate_task_sha256"]==original["candidate_task_sha256"]
    assert refreshed["split"]==original["split"]
    assert refreshed["frozen_split_sha256"]==original["frozen_split_sha256"]
    result=verify_split(tmp_path,task,"a"*64,refreshed)
    assert not result["passed"] and any("geometry_leakage" in e for e in result["errors"])


def test_explicit_capture_exclusion_blocks_before_any_physical_or_raw_replay(tmp_path,monkeypatch):
    import lane_behavior.admission as module
    capture=tmp_path/"capture";capture.mkdir()
    write_json(capture/"capture-exclusion.json",dict(reason="parked wheels elevated by curb placement"))
    monkeypatch.setattr(module,"replay_physical",lambda *args:pytest.fail("excluded capture was replayed"))
    result,path=audit_candidate(capture,corpus_root=tmp_path,publish=True)
    assert not result["passed"] and result["accepted_labels"]==0
    assert "explicitly excluded" in result["errors"][0]
    assert not (tmp_path/"accepted").exists()


def test_publisher_refreshes_under_commit_lock_and_retains_old_and_new_bindings(tmp_path,monkeypatch):
    import lane_behavior.admission as module
    task,assignment=split_fixture(tmp_path)
    capture=tmp_path/"capture";capture.mkdir();write_json(capture/"candidate-task.json",task)
    assignment["candidate_task_sha256"]=sha256_file(capture/"candidate-task.json")
    split=verify_split(tmp_path,task,assignment["candidate_task_sha256"],assignment)
    assert split["passed"]
    # A concurrent legacy publication creates a new frozen version of the same
    # compatible geometry. A manual stale approval still fails; opt-in may refresh.
    frozen=tmp_path/"splits.json";baseline=json.loads(frozen.read_text());baseline["revision"]=2
    write_json(frozen,baseline)
    write_json(tmp_path/"qualification.json",dict(split_binding=dict(path=str(frozen),sha256=sha256_file(frozen))))
    report=dict(episode_id="test",files={"candidate-task.json":assignment["candidate_task_sha256"]},archives={})
    write_json(capture/"capture-report.json",report)
    audit=dict(passed=True,checks={name:True for name in module.REQUIRED_CHECKS},requested_split_assignment=dict(assignment))
    with pytest.raises(ValueError,match="current frozen"):
        _publish(capture,tmp_path,report,[],split,audit,capture)
    seen=[]
    class StopBeforeAdmission:
        def __init__(self,path,**kwargs):self.path=path
        def __enter__(self):
            saved=json.loads((self.path.parent/"behavior-split.json").read_text());seen.append(saved)
            raise RuntimeError("stop before synthetic fixture admission")
        def __exit__(self,*args):pass
    monkeypatch.setattr(module,"AdmittedBehaviorSequence",StopBeforeAdmission)
    audit["refresh_split_binding_allowed"]=True
    with pytest.raises(RuntimeError,match="synthetic fixture"):
        _publish(capture,tmp_path,report,[],split,audit,capture)
    history=seen[0]["publication_binding_history"]
    assert history["requested"]==history["audited"]==assignment
    assert history["published"]["frozen_split_sha256"]==sha256_file(frozen)
    assert history["published"]["candidate_task_sha256"]==assignment["candidate_task_sha256"]
    assert history["published"]["split"]==assignment["split"]
    assert not (tmp_path/"accepted").exists()


def test_split_approval_is_exact_and_current_and_town05_test_only(tmp_path):
    task,approval=split_fixture(tmp_path,town="Town05",baseline_split="test")
    assert not verify_split(tmp_path,task,"a"*64,approval)["passed"]
    approval["split"]="test"
    assert verify_split(tmp_path,task,"a"*64,approval)["passed"]
    for key in ("candidate_task_sha256","frozen_split_sha256"):
        changed={**approval,key:"b"*64}
        with pytest.raises(ValueError):verify_split(tmp_path,task,"a"*64,changed)


def test_previous_behavior_appearance_prevents_different_split(tmp_path):
    from test_behavior_split_guard import published
    task,approval=split_fixture(tmp_path)
    previous=verify_split(tmp_path,task,"a"*64,approval)
    from lane_behavior.admission import _section
    published(tmp_path,_section(previous["candidate_section"]))
    changed={**approval,"candidate_task_sha256":"b"*64,"split":"train"}
    result=verify_split(tmp_path,task,"b"*64,changed)
    assert not result["passed"] and any("geometry_leakage" in e for e in result["errors"])


def test_inventory_escape_and_modified_payload_rejected(tmp_path):
    path=tmp_path/"bound.json";path.write_text("{}")
    inventory={"bound.json":sha256_file(path)}
    verify_inventory(tmp_path,inventory)
    path.write_text('{"changed":1}')
    with pytest.raises(ValueError,match="hash mismatch"):verify_inventory(tmp_path,inventory)
    with pytest.raises(ValueError,match="escapes"):verify_inventory(tmp_path,{"../escape.json":"a"*64})


def test_failed_capture_yields_audit_and_cannot_create_accepted_data(tmp_path):
    capture=tmp_path/"candidate";capture.mkdir()
    write_json(capture/"capture-report.json",dict(status="capture_failed"))
    result,path=audit_candidate(capture,corpus_root=tmp_path,publish=True)
    assert not result["passed"] and not result["training_ready"] and result["accepted_labels"]==0
    assert (path/"candidate-audit.json").is_file()
    assert not (tmp_path/"accepted").exists()
    with pytest.raises(ValueError,match="incomplete"):_publish(capture,tmp_path,{},[],{},result,path)


def geometry_record(target,*,lo=0.,hi=80.,index=100):
    q=np.arange(lo,hi+.01,.5)
    evidence=(EvidenceFrame(1000+index,index*.05,"wide"),)
    road=ObservedRoad(ObservableReference(target.reference._left(q),target.reference._right(q),
        q,[evidence]*len(q),index*.05))
    obstacles=[ObservedObstacle(o.identity,o.polygon(),o.kind,evidence,"static") for o in target.obstacles]
    record=build_record(road,obstacles,label_frame_id=1000+index,label_timestamp_s=index*.05,
        evidence_bindings={"raw/proof.npz":"a"*64},ego_position_world=[5.,3.,0.],ego_yaw_rad=0.,
        target_mode="observed_future_geometry")
    assert record["moving_supervision"],record["reasons"]
    return dict(label_index=index,record=record)


def test_future_geometry_coverage_passes_even_when_ego_has_not_reached_the_phases(obstacle_target):
    rows=[dict(position_world=[5.,3.,0.]) for i in range(801)]
    result=verify_label_recovery(obstacle_target,rows,[geometry_record(obstacle_target)])
    assert result["passed"],result["errors"]
    assert result["observed_geometry_intervals_m"][0][0]>0
    assert result["observed_geometry_intervals_m"][0][1]>70


def test_missing_phase_geometry_does_not_inherit_physical_event_pass(obstacle_target):
    rows=[dict(position_world=[i*.1,3.,0.]) for i in range(801)]
    # The observed whole-road crop can include the obstacle but end before the
    # return and nominal recovery; movement through those phases is insufficient.
    partial=[geometry_record(obstacle_target,hi=50.)]
    result=verify_label_recovery(obstacle_target,rows,partial)
    assert not result["passed"]
    assert any("five_meters" in e for e in result["errors"])


def test_partial_observed_departure_is_retained_without_claiming_entire_event_visibility(obstacle_target):
    rows=[dict(position_world=[5.,3.,0.]) for i in range(801)]
    event=obstacle_target.excursions[0]
    partial=geometry_record(obstacle_target,lo=event.start_s-2.)
    result=verify_label_recovery(obstacle_target,rows,[partial])
    assert result["passed"],result["errors"]
    phase=result["event_coverage"][0]["phases"]["departure"]
    assert .25<=phase["maximum_contiguous_m"] and 0<phase["covered_fraction"]<1
    assert phase["gaps_m"] and result["entire_event_visually_supervised"] is False


def test_entirely_missing_departure_fails_even_with_later_complete_recovery(obstacle_target):
    rows=[dict(position_world=[5.,3.,0.]) for i in range(801)]
    event=obstacle_target.excursions[0]
    partial=geometry_record(obstacle_target,lo=event.plateau_start_s-1.)
    result=verify_label_recovery(obstacle_target,rows,[partial])
    assert not result["passed"]
    assert any("no_contiguous_observed_departure_geometry" in e for e in result["errors"])


@pytest.mark.parametrize("mutation",[None,"rgb","depth","instance","stamp"])
def test_all_retained_raw_encodings_replay_even_without_road_geometry_at_that_frame(tmp_path,monkeypatch,mutation):
    import lane_behavior.admission as module
    # The archive reader is isolated here; its real hash/count checks have their
    # own suite. This test proves raw encoding replay, including non-label index5.
    rgb=np.indices((256,512))[1].astype(np.uint8)[...,None].repeat(3,axis=2)
    raw=np.concatenate((rgb[...,::-1],np.zeros((256,512,1),np.uint8)),axis=2)
    instance_raw=np.zeros_like(raw);semantic,instance=decode_carla_instance_bgra(instance_raw)
    packed=pack_model_rgb(rgb,np.eye(3))
    values=dict(rgb=rgb,rgb_bgra=raw,depth_m=np.zeros((256,512),np.float32),depth_bgra=np.zeros_like(raw),
        packed=packed,model_to_source=np.eye(3),instance_bgra=instance_raw,semantic_ids=semantic,instance_ids=instance)
    if mutation=="rgb":values["rgb"]=rgb.copy();values["rgb"][0,0,0]^=1
    elif mutation=="depth":values["depth_m"][0,0]=1.
    elif mutation=="instance":values["instance_bgra"][0,0,0]=1
    rows=[dict(index=i,frame_id=1000+i,timestamp=i*.05,desire=[1.,0.,0.,0.,0.,0.,0.,0.],traffic_convention=[1.,0.]) for i in range(301)]
    (tmp_path/"frames.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    report=dict(files={},archives={});proof=dict(index=5,frame_id=1005,timestamp_s=.25,cameras={})
    for camera in ("narrow","wide"):
        name=camera+".npz";np.savez(tmp_path/name,**values);report["files"][name]=sha256_file(tmp_path/name)
        archive=camera+".zip";report["archives"][camera]=dict(path=archive,sha256="a"*64);report["files"][archive]="a"*64
        stamp=dict(frame_id=1005,timestamp_s=.25,camera=camera)
        proof["cameras"][camera]=dict(payload=name,payload_sha256=report["files"][name],
            modality_stamps={key:dict(stamp) for key in ("rgb","depth","packed","semantic","instance")},separate_semantic_sensor_stamp=stamp)
    if mutation=="stamp":proof["cameras"]["wide"]["modality_stamps"]["depth"]["frame_id"]+=1
    class Reader:
        def __init__(self,*args,**kwargs):pass
        def __len__(self):return 301
        def __getitem__(self,index):return packed
        def close(self):pass
    monkeypatch.setattr(module,"FrameArchiveReader",Reader)
    if mutation is None:
        result=verify_retention(tmp_path,report,rows,[proof])
        assert result["passed"] and "all retained 4 Hz" in result["raw_encoding_scope"]
    else:
        with pytest.raises(ValueError,match="encoding|stamps"):verify_retention(tmp_path,report,rows,[proof])
