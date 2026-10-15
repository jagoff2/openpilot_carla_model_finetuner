"""Measured road placement and bounded settling of stationary scene fixtures.

Vehicles remain physical, braked and wholly inside the road. Wheel support is
replayed from actual wheel centres/radii and collision rays; it is not an engine
contact-force measurement. Pedestrians are static grounded fixtures, not moving
actor demonstrations. Call only while owning the synchronous simulator world.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
from itertools import product
import math
import json

import numpy as np

from lane_dataset.traffic import convex_hull
from .records import digest
from .target import _boundary_distance,_inside


SCHEMA="native_static_obstacle_grounding_v1"
WHEEL_API_SOURCE="https://raw.githubusercontent.com/carla-simulator/carla/0.9.16/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Vehicle/CarlaWheeledVehicle.cpp"


class GroundSupportError(ValueError):
    """Keep failed physical query inputs in the caller's ordinary error report."""
    def __init__(self,message,details):
        self.details=details
        super().__init__(message+': '+json.dumps(details,sort_keys=True,separators=(',',':'),allow_nan=False))


@dataclass(frozen=True)
class GroundingConfig:
    road_clearance_m: float=.15
    maximum_support_gap_m: float=.04
    maximum_support_penetration_m: float=.04
    minimum_gravity_fall_m: float=.08
    initial_drop_m: float=.25
    maximum_settle_ticks: int=240
    minimum_settle_ticks: int=60
    stable_frames: int=20
    maximum_stable_displacement_m: float=.001
    maximum_stable_speed_mps: float=.01
    maximum_stable_angular_speed_rps: float=.005

    def __post_init__(self):
        if any(not math.isfinite(v) or v<=0 for v in asdict(self).values()):raise ValueError("invalid grounding limits")
        if (self.road_clearance_m<.15 or self.maximum_support_gap_m>.04 or self.maximum_support_penetration_m>.04
                or self.minimum_gravity_fall_m<.08 or self.maximum_settle_ticks>240 or self.stable_frames<20
                or not self.stable_frames<=self.minimum_settle_ticks<=self.maximum_settle_ticks or not .2<=self.initial_drop_m<=.75
                or self.maximum_stable_displacement_m>.001 or self.maximum_stable_speed_mps>.01
                or self.maximum_stable_angular_speed_rps>.005):
            raise ValueError("grounding limits cannot weaken in-road support or settling checks")
        if any(type(getattr(self,n)) is not int for n in ("maximum_settle_ticks","minimum_settle_ticks","stable_frames")):
            raise ValueError("grounding tick counts must be integers")


def road_digest(reference):
    return digest(dict(left_xyz=reference.left_xyz.tolist(),right_xyz=reference.right_xyz.tolist(),s=reference.s.tolist()))


def _ribbon(reference):
    q=np.unique(np.r_[reference.s,np.arange(reference.s[0],reference.s[-1],.1)])
    left,right=reference._left(q)[:,:2],reference._right(q)[:,:2]
    second=np.maximum(np.linalg.norm(reference._left(q,2)[:,:2],axis=1),np.linalg.norm(reference._right(q,2)[:,:2],axis=1))
    guard=float(np.max(np.maximum(second[:-1],second[1:])*np.diff(q)**2/8))
    return np.r_[left,right[::-1]],guard


def body_road_clearance(vertices_xyz,reference):
    vertices=np.asarray(vertices_xyz,float)
    if vertices.shape!=(8,3) or not np.isfinite(vertices).all():raise ValueError("actual eight-corner 3D bbox required")
    polygon=convex_hull(vertices[:,:2]);ribbon,guard=_ribbon(reference)
    return float(_boundary_distance(polygon,ribbon)-guard) if _inside(polygon,ribbon).all() else -1.


def actor_envelope(actor):
    """Verify the measured 3D box against its independently recorded rigid pose."""
    matrix=np.asarray(actor['transform_world'],float);position=np.asarray(actor['position_world_m'],float)
    if (matrix.shape!=(4,4) or position.shape!=(3,) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3],[0,0,0,1],atol=1e-6,rtol=0)
            or not np.allclose(matrix[:3,3],position,atol=1e-5,rtol=0)
            or not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-5,rtol=0)
            or abs(np.linalg.det(matrix[:3,:3])-1)>1e-5):
        raise ValueError('actual obstacle pose/transform mismatch')
    yaw=float(actor['yaw_rad']);matrix_yaw=math.atan2(matrix[1,0],matrix[0,0])
    if not math.isfinite(yaw) or abs((yaw-matrix_yaw+np.pi)%(2*np.pi)-np.pi)>1e-5:
        raise ValueError('actual obstacle heading differs from rigid transform')
    box=actor['bounding_box'];extent=np.asarray(box['extent_m'],float);center=np.asarray(box['center_actor_m'],float)
    angles=np.radians(np.asarray(box['rotation_degrees'],float))
    vertices=np.asarray(box['vertices_world_xyz'],float)
    if (extent.shape!=(3,) or center.shape!=(3,) or angles.shape!=(3,) or vertices.shape!=(8,3)
            or not np.isfinite(np.r_[extent,center,angles,vertices.ravel()]).all() or np.any(extent<=0)):
        raise ValueError('actual obstacle has invalid measured body dimensions')
    roll,pitch,yaw=angles;cr,sr=np.cos(roll),np.sin(roll);cp,sp=np.cos(pitch),np.sin(pitch);cy,sy=np.cos(yaw),np.sin(yaw)
    rotation=np.array([[cp*cy,cy*sp*sr-sy*cr,-cy*sp*cr-sy*sr],
        [cp*sy,sy*sp*sr+cy*cr,-sy*sp*cr+cy*sr],[sp,-cp*sr,cp*cr]])
    local=np.asarray(list(product((-1.,1.),repeat=3)))*extent
    local=local@rotation.T+center
    expected=local@matrix[:3,:3].T+position
    distances=np.linalg.norm(vertices[:,None,:]-expected[None,:,:],axis=-1)
    if max(distances.min(0).max(),distances.min(1).max())>1e-4:
        raise ValueError('actual obstacle fixed envelope disagrees with measured pose and dimensions')
    return vertices,matrix,local


def stationary_envelope(actors,*,anchor=None):
    """Actual box union plus the full allowed stationary translation/rotation bound.

    The guard conservatively covers any intermediate pose remaining inside the
    same 3 cm / 0.5 degree stationary envelope. It is never deducted from the
    required obstacle clearance; it is deducted from measured free distance.
    """
    actors=list(actors)
    if not actors:raise ValueError('stationary obstacle envelope requires measured actors')
    anchor=actors[0] if anchor is None else anchor
    initial,matrix,local=actor_envelope(anchor);position=matrix[:3,3]
    vertices=[initial];maximum_translation=0.;maximum_angle=0.
    for actor in actors:
        actual,current,shape=actor_envelope(actor)
        delta=float(np.linalg.norm(current[:3,3]-position))
        angle=float(np.arccos(np.clip((np.trace(current[:3,:3]@matrix[:3,:3].T)-1)/2,-1,1)))
        velocity=np.asarray(actor['velocity_world_mps'],float);rate=float(actor['yaw_rate_rps'])
        if (actor['actor_id']!=anchor['actor_id'] or not np.allclose(shape,local,atol=1e-5,rtol=0)
                or delta>.03+1e-8 or angle>math.radians(.5)+1e-8
                or velocity.shape!=(3,) or not np.isfinite(velocity).all() or not math.isfinite(rate)
                or np.linalg.norm(velocity)>.05 or abs(rate)>math.radians(.5)):
            raise ValueError('actual obstacle is not stationary within unchanged pose/orientation limits')
        maximum_translation=max(maximum_translation,delta);maximum_angle=max(maximum_angle,angle);vertices.append(actual)
    radius=float(np.linalg.norm(local,axis=1).max())
    guard=.03+2*radius*math.sin(math.radians(.5)/2)+1e-4
    return dict(polygon_xy=convex_hull(np.concatenate(vertices)[:,:2]).tolist(),
        between_observation_guard_m=guard,maximum_translation_m=maximum_translation,
        maximum_rotation_rad=maximum_angle,maximum_allowed_translation_m=.03,
        maximum_allowed_rotation_rad=math.radians(.5),actor_radius_m=radius,
        scope='actual recorded box union with full bounded stationary pose envelope')


def wheel_bottom(center,radius,axle):
    center,axis=np.asarray(center,float),np.asarray(axle,float)
    if center.shape!=(3,) or axis.shape!=(3,) or not np.isfinite(np.r_[center,axis,radius]).all() or not .15<=radius<=.8:
        raise ValueError("invalid measured wheel geometry")
    if abs(np.linalg.norm(axis)-1)>1e-5:raise ValueError("wheel axle must be unit world vector")
    down=np.array([0.,0.,-1.])+axis*axis[2]
    if np.linalg.norm(down)<.9:raise ValueError("unsupported wheel/body tilt")
    return center+radius*down/np.linalg.norm(down)


def _ground_point(ray,allowed,*,road_reference=None,maximum_road_surface_residual_m=.04):
    start,end=np.asarray(ray["start_world_m"],float),np.asarray(ray["end_world_m"],float)
    if (start.shape!=(3,) or end.shape!=(3,) or not np.isfinite(np.r_[start,end]).all()
            or np.linalg.norm(start[:2]-end[:2])>1e-6 or start[2]<=end[2]):
        raise ValueError("ground support requires finite downward vertical collision ray")
    found=[]
    for hit in ray["hits"]:
        point=np.asarray(hit["point_world_m"],float)
        if (point.shape!=(3,) or not np.isfinite(point).all() or np.linalg.norm(point[:2]-start[:2])>.002
                or not end[2]-.002<=point[2]<=start[2]+.002):
            raise ValueError("ground collision hit does not lie on retained ray")
        found.append((point,hit['label']))
    details=dict(allowed_labels=sorted(allowed),ray=ray)
    if not found:raise GroundSupportError('collision ray has no qualifying ground support',details)
    # A descending ray must use its first physical surface. An accepted road
    # beneath a sidewalk, vehicle or raised object cannot qualify the fixture.
    point,label=max(found,key=lambda item:item[0][2])
    surveyed_ground=label=='Ground' and road_reference is not None
    if label not in allowed and not surveyed_ground:
        raise GroundSupportError('collision ray has no qualifying ground support',dict(details,first_surface_label=label))
    if road_reference is not None:
        if not 0<maximum_road_surface_residual_m<=.04:
            raise ValueError('surveyed ground surface agreement cannot exceed four centimetres')
        ribbon,_=_ribbon(road_reference)
        if not _inside(point[None,:2],ribbon).all():
            raise GroundSupportError('ground support lies outside independently surveyed road',details)
        station,_=road_reference.project(point[:2])
        left,right=road_reference._left(station),road_reference._right(station)
        cross=right[:2]-left[:2]
        fraction=float(np.dot(point[:2]-left[:2],cross)/np.dot(cross,cross))
        expected=float(left[2]+fraction*(right[2]-left[2]))
        if not 0<=fraction<=1 or abs(point[2]-expected)>maximum_road_surface_residual_m+1e-8:
            raise GroundSupportError('ground support disagrees with independently surveyed road surface',
                dict(details,surveyed_surface_z_m=expected,actual_surface_z_m=float(point[2]),
                    maximum_residual_m=maximum_road_surface_residual_m))
    return point


def _ray(world,point,carla):
    point=np.asarray(point,float);start=point+np.array([0.,0.,2.]);end=point-np.array([0.,0.,3.])
    hits=world.cast_ray(carla.Location(*map(float,start)),carla.Location(*map(float,end)))
    return dict(start_world_m=start.tolist(),end_world_m=end.tolist(),hits=[dict(
        point_world_m=[h.location.x,h.location.y,h.location.z],label=str(h.label).split('.')[-1]) for h in hits])


def _ground_queries(world,actor,points,carla,*,stage):
    """Query ground with only this fixture excluded, without a physics tick.

    CARLA0.9.16 SetActorCollisions and CastRay both wait for synchronous RPC
    completion. SetActorCollisions calls SetActorEnableCollision only. The
    fixture's simulation, pose and velocity are never changed by this scope.
    """
    if not world.get_settings().synchronous_mode:
        raise ValueError('fixture collision exclusion requires owned synchronous world')
    before=_state(actor,world);rays=[]
    try:
        actor.set_collisions(False)
        rays=[_ray(world,p,carla) for p in points]
    finally:
        # A ray/RPC failure must not leave the fixture intangible. This call
        # itself waits for restoration before the caller can advance physics.
        actor.set_collisions(True)
    after=_state(actor,world)
    if digest(before)!=digest(after):
        raise GroundSupportError('fixture changed during no-tick collision query',
            dict(stage=stage,before=before,after=after,rays=rays))
    return rays,dict(stage=stage,actor_id=int(actor.id),before=before,after=after,
        collisions_restored=True,world_ticks=0,physics_setting_changed=False,
        query_points_world_m=np.asarray(points,float).tolist(),rays=rays,
        scope='own fixture excluded from collision rays only; simulator ground QA, not camera evidence')


def _verify_collision_queries(document,final):
    policy=document.get('collision_query_policy',{})
    if policy.get('version')!='own_fixture_exclusion_without_world_tick_v1':
        raise ValueError('fixture ground proof lacks exact collision-query policy')
    queries=policy.get('queries',[])
    if [q.get('stage') for q in queries]!=['initial_placement','settled_support']:
        raise ValueError('both placement and settled support queries must be retained')
    for query in queries:
        before,after=query['before'],query['after']
        if (query.get('actor_id')!=document['actor_id'] or before.get('actor_id')!=document['actor_id']
                or query.get('collisions_restored') is not True or query.get('world_ticks')!=0
                or query.get('physics_setting_changed') is not False or digest(before)!=digest(after)):
            raise ValueError('collision-query exclusion changed actor, ticked physics or was not restored')
        points=np.asarray(query['query_points_world_m'],float)
        if points.shape!=(4,3) or len(query['rays'])!=4:
            raise ValueError('all four exact ground-support query points are required')
        for point,ray in zip(points,query['rays']):
            if not np.allclose(ray['start_world_m'],point+[0,0,2],atol=1e-8,rtol=0):
                raise ValueError('collision query ray origin differs from declared support point')
    if digest(queries[-1]['before'])!=digest(final):
        raise ValueError('settled collision queries do not bind final measured actor')
    if digest(queries[0]['rays'])!=digest(document.get('placement_ground_rays')):
        raise ValueError('placement rays differ from collision-query audit')
    if digest(queries[-1]['rays'])!=digest([item['ground_ray'] for item in document['support']]):
        raise ValueError('wheel/envelope rays differ from collision-query audit')


def _state(actor,world):
    from .qualification_probe import actor_document
    snapshot=world.get_snapshot();row=actor_document(actor,snapshot)
    angular=snapshot.find(actor.id).get_angular_velocity()
    row.update(frame_id=int(snapshot.frame),timestamp_s=float(snapshot.timestamp.elapsed_seconds),
        angular_velocity_world_rps=np.radians([angular.x,angular.y,angular.z]).tolist())
    if actor.type_id.startswith("vehicle."):
        control=actor.get_control()
        row["control"]=dict(throttle=float(control.throttle),brake=float(control.brake),hand_brake=bool(control.hand_brake))
    return row


def _stable(rows,cfg):
    if len(rows)<cfg.stable_frames:return False
    tail=rows[-cfg.stable_frames:]
    p=np.asarray([r["position_world_m"] for r in tail]);v=np.asarray([r["velocity_world_mps"] for r in tail])
    w=np.asarray([r["angular_velocity_world_rps"] for r in tail])
    return bool(np.isfinite(np.r_[p.ravel(),v.ravel(),w.ravel()]).all()
        and np.linalg.norm(p-p[0],axis=1).max()<=cfg.maximum_stable_displacement_m
        and np.linalg.norm(v,axis=1).max()<=cfg.maximum_stable_speed_mps
        and np.linalg.norm(w,axis=1).max()<=cfg.maximum_stable_angular_speed_rps)


def _fixture_spec(task,cfg):
    specs=task["unmarked_behavior"]["obstacles"]
    if len(specs)!=1:raise ValueError("grounding supports one stationary fixture")
    spec=dict(specs[0])
    if spec["kind"]=="parked_vehicle":
        if (spec.get("placement_policy")!="fully_on_road_physically_settled"
                or spec.get("curb_clearance_m")!=cfg.road_clearance_m
                or "maximum_road_intrusion_m" in spec):
            raise ValueError("parked task must explicitly require fully-on-road physical placement and curb clearance")
    return spec


def verify_grounding(document,reference,*,task=None,recorded_actors=()):
    """Recompute support and stationarity from retained inputs, ignoring passed."""
    if document.get("schema")!=SCHEMA or document.get("road_geometry_sha256")!=road_digest(reference):
        raise ValueError("missing or mismatched native obstacle grounding contract")
    if task is not None and document.get("task_sha256")!=digest(task):raise ValueError("grounding task binding mismatch")
    cfg=GroundingConfig(**document["config"]);kind=document["kind"];rows=document["settle_states"]
    if task is not None and _fixture_spec(task,cfg)["kind"]!=kind:raise ValueError("grounded fixture kind differs from task")
    if kind not in ("parked_vehicle","pedestrian") or not cfg.minimum_settle_ticks<=len(rows)<=cfg.maximum_settle_ticks+1:
        raise ValueError("bounded measured fixture settling history is missing")
    frames=np.asarray([r["frame_id"] for r in rows]);timestamps=np.asarray([r["timestamp_s"] for r in rows])
    if (not np.isfinite(timestamps).all() or np.any(np.diff(frames)!=1)
            or not np.allclose(np.diff(timestamps),.05,rtol=0,atol=1e-5) or not _stable(rows,cfg)):
        raise ValueError("fixture did not remain stationary over contiguous settling frames")
    identity=document["actor_id"]
    if any(r["actor_id"]!=identity for r in rows):raise ValueError("fixture identity changed while settling")
    final=rows[-1];actor_envelope(final);support=document["support"];gaps=[]
    _verify_collision_queries(document,final)
    if (document.get("support_frame_id")!=final["frame_id"]
            or abs(document.get("support_timestamp_s",math.inf)-final["timestamp_s"])>1e-6):
        raise ValueError("ground support is not aligned with final settled actor")
    positions=np.asarray([r["position_world_m"] for r in rows],float)
    velocity=np.asarray([r["velocity_world_mps"] for r in rows],float)
    if (not np.isfinite(np.r_[positions.ravel(),velocity.ravel()]).all()
            or np.max(np.linalg.norm(np.diff(positions,axis=0)-.5*(velocity[:-1]+velocity[1:])*np.diff(timestamps)[:,None],axis=1))>.03):
        raise ValueError("settling pose/velocity disagreement or teleport")
    if kind=="parked_vehicle":
        if document.get("physics_enabled_during_settling") is not True or document.get("physics_disabled_after_settling") is not False:
            raise ValueError("parked car must remain physical during and after grounding")
        fall=float(rows[0]["position_world_m"][2]-final["position_world_m"][2])
        if fall<cfg.minimum_gravity_fall_m:raise ValueError("no measured gravity settling of parked car")
        if body_road_clearance(final["bounding_box"]["vertices_world_xyz"],reference)<cfg.road_clearance_m-1e-6:
            raise ValueError("parked car is not fully inside physical road with curb clearance")
        if len(support)!=4 or sorted(x["wheel_index"] for x in support)!=[0,1,2,3]:raise ValueError("all four measured wheels require support")
        matrix=np.asarray(final["transform_world"],float)
        if matrix.shape!=(4,4) or not np.isfinite(matrix).all():raise ValueError("actual vehicle transform missing")
        local=[]
        for wheel in support:
            center=np.asarray(wheel["position_world_cm"],float)/100.;radius=float(wheel["radius_cm"])/100.
            bottom=wheel_bottom(center,radius,matrix[:3,1]);local.append((np.linalg.inv(matrix)@np.r_[center,1.])[:3])
            ray=wheel["ground_ray"];ground=_ground_point(ray,{"Roads"},road_reference=reference,
                maximum_road_surface_residual_m=min(cfg.maximum_support_gap_m,cfg.maximum_support_penetration_m))
            if np.linalg.norm(np.asarray(ray["start_world_m"])[:2]-bottom[:2])>1e-5:
                raise ValueError("wheel support ray is not below measured wheel")
            gaps.append(float(bottom[2]-ground[2]))
        local=np.asarray(local);extent=np.asarray(final["bounding_box"]["extent_m"]);center=np.asarray(final["bounding_box"]["center_actor_m"])
        if np.any(abs(local-center)>extent+.35) or np.ptp(local[:,0])<1. or np.ptp(local[:,1])<.8:
            raise ValueError("wheel coordinates do not match actual vehicle world pose")
        for row in rows[-cfg.stable_frames:]:
            c=row.get("control",{})
            if c.get("hand_brake") is not True or c.get("brake",0)<.99 or c.get("throttle",1)>.001:
                raise ValueError("settled parked car requires measured brake/handbrake hold")
    else:
        # This certifies a grounded static bounding envelope, never articulated
        # foot contact forces, walking physics, or a dynamic pedestrian target.
        if document.get("fixture_scope")!="stationary_pedestrian_grounded_envelope":raise ValueError("unsupported pedestrian fixture scope")
        if len(support)!=4:raise ValueError("pedestrian ground support needs four bottom-envelope samples")
        vertices=np.asarray(final["bounding_box"]["vertices_world_xyz"],float)
        expected=vertices[np.argsort(vertices[:,2])[:4]]
        points=np.asarray([x["bottom_world_m"] for x in support],float)
        if points.shape!=(4,3) or not np.allclose(points,expected,atol=1e-5,rtol=0):raise ValueError("pedestrian support differs from actual bottom envelope")
        for point,item in zip(points,support):
            ground=_ground_point(item["ground_ray"],{"Roads","Sidewalks","Ground","Terrain"})
            if np.linalg.norm(ground[:2]-point[:2])>.002:raise ValueError("pedestrian support ray mismatch")
            gaps.append(float(point[2]-ground[2]))
    if max(gaps)>cfg.maximum_support_gap_m or min(gaps)<-cfg.maximum_support_penetration_m:
        raise ValueError("fixture floats above or penetrates measured ground support")
    recorded_actors=list(recorded_actors)
    recorded_bound=stationary_envelope(recorded_actors,anchor=final) if recorded_actors else None
    for actor in recorded_actors:
        if (actor is None or actor["actor_id"]!=identity
                or np.linalg.norm(np.asarray(actor["position_world_m"])-final["position_world_m"])>.03):
            raise ValueError("recorded obstacle differs from grounded stationary actor")
        observed=np.asarray(actor["bounding_box"]["vertices_world_xyz"])
        if kind=="parked_vehicle" and body_road_clearance(observed,reference)<cfg.road_clearance_m-1e-6:
            raise ValueError("recorded parked car moved onto curb or outside road")
    return dict(passed=True,actor_id=identity,kind=kind,support_gap_m=gaps,settle_frames=len(rows),
        road_clearance_m=body_road_clearance(final["bounding_box"]["vertices_world_xyz"],reference) if kind=="parked_vehicle" else None,
        recorded_stationary_envelope=recorded_bound,
        scope="measured stationary actor geometry, collision-ray ground support and vehicle gravity settling; no contact force claim")


def _pedestrian_standing_placement(world,actor,bottom,right,carla,cfg):
    """Find nearby level support beyond the curb toe, before any recording."""
    attempts=[]
    for offset in np.arange(0.,.6001,.05):
        points=bottom+right*offset
        rays,query=_ground_queries(world,actor,points,carla,stage='initial_placement')
        try:
            heights=[float(_ground_point(ray,{"Roads","Sidewalks","Ground","Terrain"})[2]) for ray in rays]
            spread=max(heights)-min(heights)
            supported=spread<=min(cfg.maximum_support_gap_m,cfg.maximum_support_penetration_m)
            attempts.append(dict(outward_offset_m=float(offset),heights_m=heights,
                height_spread_m=spread,supported=supported,query=query))
            if supported:return float(offset),rays,query,heights,attempts
        except GroundSupportError as error:
            attempts.append(dict(outward_offset_m=float(offset),supported=False,error=str(error),
                details=error.details,query=query))
    raise GroundSupportError("stationary pedestrian has no level supported standing envelope within 0.6 m of curb",
        dict(stage='before_recording',placement_attempts=attempts))


def spawn_grounded_obstacle(world,task,reference,actors,*,road_reference,config=None):
    """Spawn/position before cameras, then physically settle an in-road vehicle."""
    import carla
    cfg=config or GroundingConfig();specs=task["unmarked_behavior"]["obstacles"]
    if not specs:return None,None,None
    spec=_fixture_spec(task,cfg);kind=spec["kind"]
    if kind not in ("parked_vehicle","pedestrian") or spec["motion"]!="stationary":raise ValueError("unsupported fixture")
    station=float(spec["route_s_m"]);_,yaw,*_=reference.sample(station)
    right=np.array([-math.sin(float(yaw)),math.cos(float(yaw)),0.]);edge=np.asarray(road_reference._right(station),float)
    blueprint=sorted(world.get_blueprint_library().filter(spec["blueprint_filter"]),key=lambda bp:bp.id)[0]
    if blueprint.has_attribute("is_invincible"):blueprint.set_attribute("is_invincible","true")
    initial=edge-right*3.+np.array([0.,0.,1.])
    transform=carla.Transform(carla.Location(*map(float,initial)),carla.Rotation(yaw=math.degrees(float(yaw))))
    actor=world.spawn_actor(blueprint,transform);actors.append(actor)
    actor.set_simulate_physics(False);world.tick()
    actual=actor.get_transform();origin=np.array([actual.location.x,actual.location.y,actual.location.z])
    vertices=np.array([[v.x,v.y,v.z] for v in actor.bounding_box.get_world_vertices(actual)])
    if kind=="parked_vehicle":
        delta=-cfg.road_clearance_m-float(np.max((vertices-edge)@right))
        target=origin+right*delta
        for extra in np.arange(0.,2.01,.05):
            moved=vertices+(target-origin)-right*extra
            if body_road_clearance(moved,road_reference)>=cfg.road_clearance_m:
                target-=right*extra;break
        else:raise RuntimeError("measured parked-car body cannot fit wholly inside road")
    else:
        # Pedestrian remains beside the curb, with its inward face at the edge.
        target=origin+right*(-float(np.min((vertices-edge)@right)))
    moved=vertices+target-origin
    bottom=moved[np.argsort(moved[:,2])[:4]]
    placement_search=None
    if kind=="pedestrian":
        offset,rays,placement_query,heights,placement_search=_pedestrian_standing_placement(world,actor,bottom,right,carla,cfg)
        target+=right*offset;bottom=bottom+right*offset
    else:
        rays,placement_query=_ground_queries(world,actor,bottom,carla,stage='initial_placement')
        try:heights=[_ground_point(r,{"Roads"},road_reference=road_reference,
            maximum_road_surface_residual_m=min(cfg.maximum_support_gap_m,cfg.maximum_support_penetration_m))[2] for r in rays]
        except GroundSupportError as error:
            raise GroundSupportError('initial fixture ground placement failed',dict(stage='before_physical_settling',
                actor_id=int(actor.id),actor_origin_world_m=origin.tolist(),planned_origin_world_m=target.tolist(),
                planned_body_bottom_world_xyz=bottom.tolist(),placement_rays=rays,cause=error.details)) from error
    target[2]+=max(heights)-float(bottom[:,2].min())+(cfg.initial_drop_m if kind=="parked_vehicle" else 0.)
    actor.set_transform(carla.Transform(carla.Location(*map(float,target)),transform.rotation));world.tick()
    rows=[_state(actor,world)]
    if kind=="parked_vehicle":
        actor.set_simulate_physics(True)
        actor.apply_control(carla.VehicleControl(throttle=0.,brake=1.,hand_brake=True,steer=0.))
    for tick in range(cfg.maximum_settle_ticks):
        world.tick();rows.append(_state(actor,world))
        if tick+1>=cfg.minimum_settle_ticks and _stable(rows,cfg):break
    else:raise RuntimeError("stationary fixture did not settle within bounded physics budget")
    final=rows[-1];support=[];support_points=[]
    if kind=="parked_vehicle":
        matrix=np.asarray(final["transform_world"])
        for i,wheel in enumerate(actor.get_physics_control().wheels):
            position=np.array([wheel.position.x,wheel.position.y,wheel.position.z]);radius=float(wheel.radius)
            point=wheel_bottom(position/100.,radius/100.,matrix[:3,1])
            support.append(dict(wheel_index=i,position_world_cm=position.tolist(),radius_cm=radius))
            support_points.append(point)
    else:
        vertices=np.asarray(final["bounding_box"]["vertices_world_xyz"])
        for point in vertices[np.argsort(vertices[:,2])[:4]]:
            support.append(dict(bottom_world_m=point.tolist()));support_points.append(point)
    support_rays,support_query=_ground_queries(world,actor,support_points,carla,stage='settled_support')
    for item,ray in zip(support,support_rays):item['ground_ray']=ray
    document=dict(schema=SCHEMA,actor_id=int(actor.id),kind=kind,task_sha256=digest(task),road_geometry_sha256=road_digest(road_reference),
        config=asdict(cfg),settle_states=rows,support=support,physics_enabled_during_settling=kind=="parked_vehicle",
        support_frame_id=final["frame_id"],support_timestamp_s=final["timestamp_s"],
        physics_disabled_after_settling=kind!="parked_vehicle",wheel_api_source=WHEEL_API_SOURCE,
        fixture_scope="physical_parked_vehicle_wholly_in_road" if kind=="parked_vehicle" else "stationary_pedestrian_grounded_envelope",
        placement_ground_rays=rays,collision_query_policy=dict(version='own_fixture_exclusion_without_world_tick_v1',
            queries=[placement_query,support_query]),initial_placement_before_sensors=True,
        pose_or_velocity_replacement_during_recording=False)
    if placement_search is not None:document['standing_placement_search']=placement_search
    document["qualification"]=verify_grounding(document,road_reference,task=task)
    spec.update(placement=document["fixture_scope"],actual_road_clearance_m=document["qualification"]["road_clearance_m"])
    return actor,spec,document
