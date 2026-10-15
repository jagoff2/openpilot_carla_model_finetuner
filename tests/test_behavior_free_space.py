from copy import deepcopy
from dataclasses import replace
from dataclasses import asdict
import json

import numpy as np
import pytest

from lane_behavior.evidence import FrameStamp,Observation,MODALITIES,pack_model_rgb
from lane_behavior.free_space import (FreeSpaceConfig,measure_free_surface,build_free_space_proof,
    verify_free_space_proof,observed_passing_space,load_saved_observation)
from lane_behavior.target import ObservedRoad
from lane_dataset.geometry import ObservableReference,EvidenceFrame,LaneReference
from lane_dataset.config import sha256_file
from lane_dataset.visibility import decode_carla_depth_bgra


def scene(frame=20,time=0.,camera='wide'):
    stamp=FrameStamp(frame,time,camera);rgb=np.full((256,512,3),80,np.uint8)
    k=np.array([[320.,0,256.],[0,320.,100.],[0,0,1.]])
    transform=np.array([[0.,1.,0.,0.],[0.,0.,-1.,1.5],[1.,0.,0.,0.],[0.,0.,0.,1.]])
    y=np.arange(256)[:,None]+.5;depth=np.broadcast_to(np.divide(480.,y-100.,out=np.full_like(y,120.),where=y>100.),(256,512)).astype(np.float32).copy()
    semantic=np.broadcast_to(np.where(y>100,1,0),(256,512)).astype(np.uint8).copy()
    instance=np.zeros((256,512),np.uint16);raw=np.zeros((256,512,4),np.uint8);raw[...,2]=semantic;raw[...,3]=255
    observation=Observation(stamp,{name:stamp for name in MODALITIES},rgb,depth,semantic,instance,
        pack_model_rgb(rgb,np.eye(3)),k,transform,np.eye(3),'forward','a'*64,True,
        instance_bgra=raw,scene_actor_ids=(1,2,3))
    state=dict(actor_id=2,source='simulator_actor_state_qa',position_world_m=[10.,2.,0.],
        velocity_world_mps=[0.,0.,0.],yaw_rate_rps=0.,yaw_rad=0.)
    inventory=dict(schema='controlled_native_static_scene_v1',scope='simulator_scene_QA_not_visual_free_space',
        frame_id=frame,timestamp_s=time,ego_actor_id=1,static_obstacle_actor_ids=[2],actors=[
            dict(actor_id=1,type_id='vehicle.tesla.model3',state=dict(state,actor_id=1)),
            dict(actor_id=2,type_id='vehicle.audi.a2',state=state),dict(actor_id=3,type_id='sensor.camera.rgb',state=None)])
    return observation,inventory


def history():
    pairs=[scene(20+5*i,.25*i) for i in range(3)]
    observations,scenes=map(list,zip(*pairs));decision=observations[-1].stamp
    q=np.arange(6.,14.01,.5);left=np.c_[q,q*0-4,q*0];right=np.c_[q,q*0+4,q*0]
    ref=ObservableReference(left,right,q,[(EvidenceFrame(decision.frame_id,decision.timestamp_s,decision.camera),)]*len(q),decision.timestamp_s)
    return observations,scenes,ObservedRoad(ref)


def arguments(observations,scenes):
    return dict(decision=observations[-1].stamp,controlled_scenes=scenes,obstacle_ids=('2',),opposing_width_m=1.)


def obstacle(observation,tag=14):
    semantic=observation.semantic_ids.copy();instance=observation.instance_ids.copy();raw=observation.instance_bgra.copy()
    # Obscure the entire opposing strip without modifying camera geometry.
    semantic[130:185,190:260]=tag;instance[130:185,190:260]=2
    raw[...,2]=semantic;raw[...,1]=(instance%256).astype(np.uint8);raw[...,0]=(instance//256).astype(np.uint8)
    return replace(observation,semantic_ids=semantic,instance_ids=instance,instance_bgra=raw)


def test_complete_rendered_strip_replays_into_typed_passing_permission():
    observations,scenes,road=history();args=arguments(observations,scenes)
    proof=build_free_space_proof(observations,road,**args)
    assert proof['supported'] and proof['allowed_intervals_m']==[[6.,14.]]
    assert not proof['training_ready'] and not proof['dynamic_traffic_permitted']
    assert verify_free_space_proof(proof,observations,road,**args)==proof
    permission=observed_passing_space(proof,road)
    assert permission.opposing_width_m==1. and permission.obstacle_ids==('2',)
    assert permission.rendered_free_space_sha256==proof['proof_sha256']
    assert all(cell['cell']['maximum_ground_residual_m']<1e-6 for cell in proof['used_cells'])


@pytest.mark.parametrize('tag',[12,14,2,0])
def test_pedestrian_vehicle_curb_and_unknown_semantic_occluders_do_not_grant_cells(tag):
    observations,scenes,road=history();observations=[obstacle(o,tag) for o in observations]
    proof=build_free_space_proof(observations,road,**arguments(observations,scenes))
    assert not proof['supported'] and not proof['allowed_intervals_m']
    assert any(c.get('reason')=='cell_occluded_or_nonroad' for p in proof['frame_proofs'] for c in p['cells'])


def test_depth_foreground_cannot_be_mislabeled_as_free_ground():
    observations,scenes,road=history()
    observations=[replace(o,depth_m=o.depth_m*.9) for o in observations]
    proof=build_free_space_proof(observations,road,**arguments(observations,scenes))
    assert not proof['supported']
    assert any(c.get('reason')=='cell_depth_not_observed_ground' for p in proof['frame_proofs'] for c in p['cells'])


def test_history_cannot_bridge_an_unobserved_station_gap():
    observations,scenes,road=history();changed=[]
    for o in observations:
        semantic=o.semantic_ids.copy();raw=o.instance_bgra.copy()
        semantic[151:156,:]=14;raw[...,2]=semantic
        changed.append(replace(o,semantic_ids=semantic,instance_bgra=raw))
    proof=build_free_space_proof(changed,road,**arguments(changed,scenes))
    assert proof['supported'] and len(proof['allowed_intervals_m'])==2
    assert proof['allowed_intervals_m'][0][1]<proof['allowed_intervals_m'][1][0]


def test_raw_visible_but_absent_from_actual_crop_is_not_free_permission():
    observations,scenes,road=history();changed=[]
    for o in observations:
        warp=np.eye(3);warp[0,2]=300
        changed.append(replace(o,model_to_source=warp,packed=pack_model_rgb(o.rgb,warp)))
    proof=build_free_space_proof(changed,road,**arguments(changed,scenes))
    assert not proof['supported']


@pytest.mark.parametrize('change',['extra_vehicle','unknown_type','unknown_inventory','moving','displacement','rotation','wrong_frame','pedestrian','missing_history'])
def test_inventory_is_separate_mandatory_static_scene_qualification(change):
    observations,scenes,road=history();scenes=deepcopy(scenes)
    if change=='extra_vehicle':
        scenes[0]['actors'][2]['type_id']='vehicle.other'
    elif change=='unknown_type':scenes[0]['actors'][2]['type_id']='static.unqualified_dynamic_prop'
    elif change=='unknown_inventory':scenes[0]['actors'].pop()
    elif change=='moving':scenes[0]['actors'][1]['state']['velocity_world_mps'][0]=.1
    elif change=='displacement':scenes[0]['actors'][1]['state']['position_world_m'][0]+=.1
    elif change=='rotation':scenes[0]['actors'][1]['state']['yaw_rad']+=.1
    elif change=='wrong_frame':scenes[0]['frame_id']+=1
    elif change=='pedestrian':scenes[0]['actors'][1]['type_id']='walker.pedestrian.001'
    elif change=='missing_history':scenes=scenes[-1:];observations=observations[-1:]
    with pytest.raises(ValueError):build_free_space_proof(observations,road,**arguments(observations,scenes))


@pytest.mark.parametrize('change',['future','stale','packed','unaligned','raw_instance','extrapolated','hidden_road'])
def test_causal_raw_payload_crop_and_observed_road_binding_are_required(change):
    observations,scenes,road=history();args=arguments(observations,scenes)
    if change in ('future','stale'):
        stamp=replace(observations[0].stamp,timestamp_s=1. if change=='future' else -5.)
        observations[0]=replace(observations[0],stamp=stamp)
    elif change=='packed':
        packed=observations[0].packed.copy();packed[0,0,0]^=1;observations[0]=replace(observations[0],packed=packed)
    elif change=='unaligned':
        stamps=dict(observations[0].modality_stamps);stamps['depth']=replace(observations[0].stamp,frame_id=0)
        observations[0]=replace(observations[0],modality_stamps=stamps)
    elif change=='raw_instance':
        raw=observations[0].instance_bgra.copy();raw[140,250,2]=14;observations[0]=replace(observations[0],instance_bgra=raw)
    elif change=='extrapolated':args['station_interval_m']=(5.,15.)
    elif change=='hidden_road':road=ObservedRoad(LaneReference(road.reference.left_xyz,road.reference.right_xyz,road.reference.s))
    with pytest.raises((ValueError,TypeError)):build_free_space_proof(observations,road,**args)


def test_proof_mutation_and_changed_camera_bytes_fail_replay():
    observations,scenes,road=history();args=arguments(observations,scenes)
    proof=build_free_space_proof(observations,road,**args)
    proof['allowed_intervals_m'][0][1]+=1
    with pytest.raises(ValueError,match='binding changed'):observed_passing_space(proof,road)
    with pytest.raises(ValueError,match='raw replay'):verify_free_space_proof(proof,observations,road,**args)


def test_surface_measurement_alone_does_not_assert_safe_traffic_gap():
    observations,_,road=history()
    result=measure_free_surface(observations[-1],road)
    assert all(c['supported'] for c in result['cells'])
    assert result['scope']=='rendered_surface_cells_only_no_traffic_permission'
    assert not result['training_ready']


def test_saved_raw_loader_binds_file_hashes_and_replays_encoded_depth(tmp_path):
    observations,_,road=history();o=observations[-1]
    fixture=tmp_path/'depth-fixture.json';fixture.write_text(json.dumps({'wide':{'depth_mode':'forward','qualified':True}}))
    encoded=np.rint(o.depth_m.astype(float)/1000.*(256**3-1)).astype(np.uint32)
    raw=np.stack(((encoded>>16)&255,(encoded>>8)&255,encoded&255,np.full_like(encoded,255)),axis=-1).astype(np.uint8)
    arrays={name:getattr(o,name) for name in ('rgb','depth_m','semantic_ids','instance_ids','instance_bgra','packed','intrinsics','camera_from_world','model_to_source')}
    arrays.update(depth_m=decode_carla_depth_bgra(raw).astype(np.float32),depth_bgra=raw,
        rgb_bgra=np.concatenate([o.rgb[...,[2,1,0]],np.full((*o.rgb.shape[:2],1),255,np.uint8)],axis=-1))
    payload=tmp_path/'raw.npz';np.savez_compressed(payload,**arrays)
    args=dict(payload_sha256=sha256_file(payload),stamp=o.stamp,modality_stamps={k:asdict(v) for k,v in o.modality_stamps.items()},
        depth_fixture=fixture,depth_fixture_sha256=sha256_file(fixture),scene_actor_ids=o.scene_actor_ids)
    loaded=load_saved_observation(payload,**args)
    assert all(c['supported'] for c in measure_free_surface(loaded,road)['cells'])
    arrays['depth_m']=arrays['depth_m']*.9;np.savez_compressed(payload,**arrays)
    with pytest.raises(ValueError,match='byte hash'):load_saved_observation(payload,**args)
    args['payload_sha256']=sha256_file(payload)
    with pytest.raises(ValueError,match='raw camera bytes'):load_saved_observation(payload,**args)
