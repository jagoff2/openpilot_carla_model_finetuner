from copy import deepcopy
from dataclasses import asdict
from itertools import product
import json
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from lane_behavior.grounding import (GroundingConfig,SCHEMA,body_road_clearance,road_digest,
    verify_grounding,wheel_bottom)
from test_behavior_planned import simulator_road


def ray(point,height=0.,label="Roads"):
    p=np.asarray(point,float)
    return dict(start_world_m=(p+[0,0,2]).tolist(),end_world_m=(p-[0,0,3]).tolist(),
        hits=[dict(point_world_m=[p[0],p[1],height],label=label)])


def actor(z=0.,y=3.,index=60):
    transform=np.eye(4);transform[:3,3]=[40.,y,z]
    center=np.array([0.,0.,.72]);extent=np.array([2.,1.,.7])
    corners=np.array(list(product([-1,1],repeat=3)))*extent+center+transform[:3,3]
    return dict(actor_id=7,frame_id=100+index,timestamp_s=index*.05,blueprint="vehicle.audi.a2",
        yaw_rad=0.,yaw_rate_rps=0.,
        position_world_m=transform[:3,3].tolist(),velocity_world_mps=[0.,0.,-.25 if index<20 else 0.],
        angular_velocity_world_rps=[0.,0.,0.],transform_world=transform.tolist(),
        control=dict(throttle=0.,brake=1.,hand_brake=True),bounding_box=dict(extent_m=extent.tolist(),
            center_actor_m=center.tolist(),rotation_degrees=[0.,0.,0.],vertices_world_xyz=corners.tolist()))


def bind_query_audit(document):
    final=document['settle_states'][-1];rays=[s['ground_ray'] for s in document['support']]
    points=[(np.asarray(r['start_world_m'])-[0,0,2]).tolist() for r in rays]
    document['placement_ground_rays']=deepcopy(rays)
    document['collision_query_policy']=dict(version='own_fixture_exclusion_without_world_tick_v1',queries=[
        dict(stage=stage,actor_id=7,before=deepcopy(final),after=deepcopy(final),collisions_restored=True,
            world_ticks=0,physics_setting_changed=False,query_points_world_m=points,rays=deepcopy(rays))
        for stage in ('initial_placement','settled_support')])


@pytest.fixture
def grounded():
    reference=simulator_road()
    rows=[actor(max(0.,.25-i*.0125),index=i) for i in range(61)]
    support=[]
    for i,(x,y) in enumerate(product([-1.2,1.2],[-.8,.8])):
        center=np.array([40+x,3+y,.3]);bottom=center-[0,0,.3]
        support.append(dict(wheel_index=i,position_world_cm=(center*100).tolist(),radius_cm=30.,ground_ray=ray(bottom)))
    document=dict(schema=SCHEMA,actor_id=7,kind="parked_vehicle",road_geometry_sha256=road_digest(reference),
        config=asdict(GroundingConfig()),settle_states=rows,support=support,physics_enabled_during_settling=True,
        physics_disabled_after_settling=False,support_frame_id=160,support_timestamp_s=3.)
    bind_query_audit(document)
    return document,reference


def test_physically_settled_all_road_supported_car_replays_without_trusting_passed(grounded):
    document,reference=grounded
    document["qualification"]={"passed":False}
    result=verify_grounding(document,reference,recorded_actors=[document["settle_states"][-1]])
    assert result["passed"] and result["road_clearance_m"]>.15
    assert max(abs(v) for v in result["support_gap_m"])<1e-10
    assert "no contact force claim" in result["scope"]
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize("failure",["floating_wheel","curb_support","frozen","no_fall","wrong_world_units",
    "moving","missing_wheel","stale_support","teleport","brake_released","off_road"])
def test_grounding_failures_are_not_overridden_by_passed_flag(grounded,failure):
    document,reference=grounded;document["qualification"]={"passed":True}
    if failure=="floating_wheel":document["support"][0]["ground_ray"]["hits"][0]["point_world_m"][2]=-.18
    elif failure=="curb_support":document["support"][0]["ground_ray"]["hits"][0]["label"]="Sidewalks"
    elif failure=="frozen":document["physics_disabled_after_settling"]=True
    elif failure=="no_fall":
        for row in document["settle_states"]:row["position_world_m"][2]=0.
    elif failure=="wrong_world_units":document["support"][0]["position_world_cm"]=[-120.,-80.,30.]
    elif failure=="moving":document["settle_states"][-1]["velocity_world_mps"][0]=.1
    elif failure=="missing_wheel":document["support"].pop()
    elif failure=="stale_support":document["support_frame_id"]-=1
    elif failure=="teleport":document["settle_states"][10]["position_world_m"][0]+=1
    elif failure=="brake_released":document["settle_states"][-1]["control"]["hand_brake"]=False
    else:document["settle_states"][-1]["bounding_box"]["vertices_world_xyz"]=actor(y=5.5)["bounding_box"]["vertices_world_xyz"]
    with pytest.raises(ValueError):verify_grounding(document,reference)


def test_full_road_inset_checks_whole_body_not_only_actor_origin():
    road=simulator_road()
    assert body_road_clearance(actor(y=4.8)["bounding_box"]["vertices_world_xyz"],road)>.15
    assert body_road_clearance(actor(y=5.95)["bounding_box"]["vertices_world_xyz"],road)<0


def test_recorded_actor_cannot_move_after_grounding(grounded):
    document,reference=grounded
    recorded=deepcopy(document["settle_states"][-1]);recorded["position_world_m"][2]+=.18
    with pytest.raises(ValueError,match="pose/transform"):
        verify_grounding(document,reference,recorded_actors=[recorded])


def test_grounding_allows_bounded_rigid_settling_with_vertices_beyond_translation_cap(grounded):
    from test_behavior_admission import shift_obstacle
    document,reference=grounded;recorded=deepcopy(document['settle_states'][-1])
    shift_obstacle(recorded,dy=.02,angle=np.radians(.4))
    displacement=np.linalg.norm(np.asarray(recorded['bounding_box']['vertices_world_xyz'])-
        document['settle_states'][-1]['bounding_box']['vertices_world_xyz'],axis=1)
    assert displacement.max()>.03
    result=verify_grounding(document,reference,recorded_actors=[recorded])
    assert result['passed'] and result['recorded_stationary_envelope']['maximum_translation_m']<.03
    recorded['bounding_box']['vertices_world_xyz']=document['settle_states'][-1]['bounding_box']['vertices_world_xyz']
    with pytest.raises(ValueError,match='fixed envelope disagrees'):
        verify_grounding(document,reference,recorded_actors=[recorded])


def test_wheel_bottom_uses_actual_roll_and_rejects_bad_axes():
    axis=np.array([0.,np.cos(.1),np.sin(.1)])
    point=wheel_bottom([0,0,1],.3,axis)
    assert abs(np.linalg.norm(point-[0,0,1])-.3)<1e-10
    assert abs((point-[0,0,1])@axis)<1e-10
    with pytest.raises(ValueError):wheel_bottom([0,0,1],.3,[0,2,0])


def test_missing_ground_preserves_exact_query_and_hit_labels():
    from lane_behavior.grounding import _ground_point,GroundSupportError
    probe=ray([40.,3.,.2],height=.9,label='Vehicles')
    with pytest.raises(GroundSupportError) as error:_ground_point(probe,{'Roads'})
    assert error.value.details==dict(allowed_labels=['Roads'],ray=probe,first_surface_label='Vehicles')
    assert 'Vehicles' in str(error.value) and 'start_world_m' in str(error.value)


def test_ground_tag_requires_independent_road_xy_and_surface_agreement():
    from lane_behavior.grounding import _ground_point
    road=simulator_road()
    assert _ground_point(ray([40,3,.2],label='Ground'),{'Roads'},road_reference=road)[2]==0
    with pytest.raises(ValueError,match='no qualifying'):
        _ground_point(ray([40,3,.2],label='Ground'),{'Roads'})
    with pytest.raises(ValueError,match='surface'):
        _ground_point(ray([40,3,.2],height=.041,label='Ground'),{'Roads'},road_reference=road)
    with pytest.raises(ValueError,match='outside'):
        _ground_point(ray([40,6.1,.2],label='Ground'),{'Roads'},road_reference=road)
    with pytest.raises(ValueError,match='no qualifying'):
        _ground_point(ray([40,3,.2],label='Sidewalks'),{'Roads'},road_reference=road)


@pytest.mark.parametrize('label',['Vehicles','Sidewalks','Ground'])
def test_first_surface_cannot_be_skipped_to_hidden_road_below(label):
    from lane_behavior.grounding import _ground_point
    probe=ray([40,3,.2],height=.18,label=label)
    probe['hits'].append(dict(point_world_m=[40,3,0.],label='Roads'))
    with pytest.raises(ValueError):_ground_point(probe,{'Roads'},road_reference=simulator_road())


def test_wheel_ground_tag_is_replayed_only_inside_surveyed_pavement(grounded):
    document,reference=grounded
    for wheel in document['support']:wheel['ground_ray']['hits'][0]['label']='Ground'
    bind_query_audit(document)
    assert verify_grounding(document,reference)['passed']
    document['support'][0]['ground_ray']['hits'][0]['point_world_m'][2]=.05
    bind_query_audit(document)
    with pytest.raises(ValueError,match='surface'):verify_grounding(document,reference)


def test_pedestrian_ground_envelope_is_explicitly_static_and_needs_all_support(grounded):
    document,reference=grounded
    document.update(kind="pedestrian",fixture_scope="stationary_pedestrian_grounded_envelope")
    final=document["settle_states"][-1]
    points=np.asarray(final["bounding_box"]["vertices_world_xyz"]);points=points[np.argsort(points[:,2])[:4]]
    document["support"]=[dict(bottom_world_m=p.tolist(),ground_ray=ray(p,label="Sidewalks")) for p in points]
    bind_query_audit(document)
    assert verify_grounding(document,reference)["passed"]
    document["support"][0]["ground_ray"]["hits"][0]["point_world_m"][2]=-.1
    bind_query_audit(document)
    with pytest.raises(ValueError,match="floats"):verify_grounding(document,reference)


def test_live_helper_sequence_keeps_vehicle_physical_and_fully_in_road(monkeypatch):
    import lane_behavior.grounding as module
    from lane_dataset.geometry import LaneReference
    class Vector:
        def __init__(self,x=0.,y=0.,z=0.):self.x,self.y,self.z=x,y,z
    class Rotation:
        def __init__(self,yaw=0.):self.yaw=yaw
    class Transform:
        def __init__(self,location=None,rotation=None):self.location=location or Vector();self.rotation=rotation or Rotation()
    carla=SimpleNamespace(Location=Vector,Rotation=Rotation,Transform=Transform,VehicleControl=lambda **kw:SimpleNamespace(**kw))
    monkeypatch.setitem(sys.modules,"carla",carla)
    class Box:
        def get_world_vertices(self,transform):
            p=np.array([transform.location.x,transform.location.y,transform.location.z])
            points=np.array(list(product([-1,1],repeat=3)))*[2.,1.,.7]+[0,0,.72]+p
            return [Vector(*point) for point in points]
    class Vehicle:
        id=7;type_id="vehicle.audi.a2"
        def __init__(self,transform):
            self.transform=transform;self.bounding_box=Box();self.physical=True;self.events=[];self.vz=0.
            self.collisions=True
            self.control=SimpleNamespace(throttle=0.,brake=0.,hand_brake=False)
        def set_simulate_physics(self,value):self.physical=value;self.events.append(("physics",value))
        def set_collisions(self,value):self.collisions=value;self.events.append(('collisions',value))
        def set_transform(self,value):
            assert not self.physical,"no pose replacement after physics is enabled"
            self.transform=value;self.events.append(("placement",value.location.y))
        def get_transform(self):return self.transform
        def apply_control(self,value):self.control=value
        def get_physics_control(self):
            p=self.transform.location
            return SimpleNamespace(wheels=[SimpleNamespace(position=Vector((p.x+x)*100,(p.y+y)*100,(p.z+.3)*100),radius=30.)
                for x,y in product([-1.2,1.2],[-.8,.8])])
    bp=SimpleNamespace(id="vehicle.audi.a2",has_attribute=lambda name:False)
    class World:
        def __init__(self):self.frame=0;self.actor=None
        def get_blueprint_library(self):return SimpleNamespace(filter=lambda expression:[bp])
        def get_settings(self):return SimpleNamespace(synchronous_mode=True)
        def spawn_actor(self,blueprint,transform):self.actor=Vehicle(transform);return self.actor
        def tick(self):
            assert self.actor.collisions,'physics must not tick with fixture excluded'
            self.frame+=1
            if self.actor.physical:
                old=self.actor.transform.location.z;self.actor.transform.location.z=max(0.,old-.05)
                self.actor.vz=(self.actor.transform.location.z-old)/.05
            else:self.actor.vz=0.
            return self.frame
    world=World()
    def state(vehicle,world):
        p=vehicle.transform.location;row=actor(z=p.z,y=p.y,index=world.frame)
        row["velocity_world_mps"]=[0,0,vehicle.vz]
        row["control"]=dict(throttle=vehicle.control.throttle,brake=vehicle.control.brake,hand_brake=vehicle.control.hand_brake)
        return row
    monkeypatch.setattr(module,"_state",state)
    monkeypatch.setattr(module,"_ray",lambda world,point,carla:ray(point))
    road=simulator_road();corridor=LaneReference((road.left_xyz+road.right_xyz)/2,road.right_xyz,road.s)
    task=dict(unmarked_behavior=dict(obstacles=[dict(kind="parked_vehicle",motion="stationary",route_s_m=40.,blueprint_filter="vehicle.audi.a2",
        placement_policy="fully_on_road_physically_settled",curb_clearance_m=.15)]))
    actors=[]
    parked,spec,document=module.spawn_grounded_obstacle(world,task,corridor,actors,road_reference=road)
    assert parked.physical and parked.events[0]==("physics",False)
    assert [e for e in parked.events if e[0]=='physics'][-1]==('physics',True)
    assert parked.collisions and [e for e in parked.events if e[0]=='collisions']==[
        ('collisions',False),('collisions',True),('collisions',False),('collisions',True)]
    assert len([x for x in parked.events if x[0]=="placement"])==1
    assert spec["actual_road_clearance_m"]>=.15
    assert document["qualification"]["passed"] and document["physics_disabled_after_settling"] is False
    assert actors==[parked]


def test_fixture_only_collision_exclusion_restores_even_when_ray_raises(monkeypatch):
    import lane_behavior.grounding as module
    world=SimpleNamespace(get_settings=lambda:SimpleNamespace(synchronous_mode=True))
    changes=[];fixture=SimpleNamespace(id=7,set_collisions=lambda value:changes.append(value))
    monkeypatch.setattr(module,'_state',lambda actor,world:globals()['actor']())
    monkeypatch.setattr(module,'_ray',lambda *args:(_ for _ in ()).throw(RuntimeError('ray failure')))
    with pytest.raises(RuntimeError,match='ray failure'):
        module._ground_queries(world,fixture,[[40,3,0]],None,stage='settled_support')
    assert changes==[False,True]


@pytest.mark.parametrize('level_sidewalk',[True,False])
def test_pedestrian_placement_moves_past_curb_lip_without_weakening_support(monkeypatch,level_sidewalk):
    import lane_behavior.grounding as module
    points=np.array([[39.7,4.,1.],[40.3,4.,1.],[39.7,4.6,1.],[40.3,4.6,1.]])
    queries=[]
    def query(world,actor,points,carla,*,stage):
        queries.append(points.copy())
        heights=[(.15 if p[1]>=4.12 else 0.) if level_sidewalk else (.15 if p[0]>40 else 0.) for p in points]
        rays=[ray(p,height=h,label='Sidewalks') for p,h in zip(points,heights)]
        return rays,dict(stage=stage,query_points_world_m=points.tolist(),rays=rays)
    monkeypatch.setattr(module,'_ground_queries',query)
    if not level_sidewalk:
        with pytest.raises(module.GroundSupportError,match='within 0.6 m'):
            module._pedestrian_standing_placement(None,None,points,np.array([0.,1.,0.]),None,GroundingConfig())
        assert len(queries)==13
    else:
        offset,rays,audit,heights,attempts=module._pedestrian_standing_placement(
            None,None,points,np.array([0.,1.,0.]),None,GroundingConfig())
        assert offset==pytest.approx(.15) and heights==[.15]*4
        assert not any(a['supported'] for a in attempts[:-1]) and attempts[-1]['supported'] is True
        assert np.allclose(queries[-1],points+[0,.15,0])
        json.dumps(attempts,allow_nan=False)


@pytest.mark.parametrize('change',['tick','physics','restore','pose','ray'])
def test_collision_query_replay_rejects_scope_or_pose_changes(grounded,change):
    document,reference=grounded;query=document['collision_query_policy']['queries'][-1]
    if change=='tick':query['world_ticks']=1
    elif change=='physics':query['physics_setting_changed']=True
    elif change=='restore':query['collisions_restored']=False
    elif change=='pose':query['after']['position_world_m'][0]+=.1
    else:query['rays'][0]['start_world_m'][0]+=.1
    with pytest.raises(ValueError):verify_grounding(document,reference)
