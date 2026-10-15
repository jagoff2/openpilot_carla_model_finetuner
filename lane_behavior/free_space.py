"""Causal rendered road-surface permission for controlled static parked-car scenes.

This proves only raster-resolved ground cells in an observed road strip. It is
not an oncoming-traffic gap model. Every allowed cell must be visible in a saved
RGB/depth/semantic frame and in that frame's exact packed model crop. The
independent complete actor inventory restricts permission to an ego vehicle and
stationary parked vehicle; actor metadata never substitutes for visible ground.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from lane_dataset.geometry import EvidenceFrame, ObservableReference
from lane_dataset.config import sha256_file
from lane_dataset.visibility import decode_carla_depth_bgra
from .evidence import (FrameStamp, Observation, MODALITIES, SEMANTIC_SCHEMA, _digest,
    _lookup, array_sha256, decode_carla_instance_bgra, pack_model_rgb, visible_surface_points)
from .target import ObservedRoad


VERSION='observed_static_opposing_strip_surface_v1'


def load_saved_observation(payload,*,payload_sha256,stamp,modality_stamps,
                           depth_fixture,depth_fixture_sha256,scene_actor_ids):
    """Read exact captured sensor bytes, checking decoded RGB/depth on replay."""
    payload,depth_fixture=Path(payload),Path(depth_fixture)
    if sha256_file(payload)!=payload_sha256 or sha256_file(depth_fixture)!=depth_fixture_sha256:
        raise ValueError('free-space payload or depth fixture byte hash changed')
    fixture=json.loads(depth_fixture.read_text(encoding='utf-8'))[stamp.camera]
    with np.load(payload,allow_pickle=False) as stored:
        arrays={name:stored[name].copy() for name in ('rgb','depth_m','semantic_ids','instance_ids','instance_bgra',
            'packed','intrinsics','camera_from_world','model_to_source')}
        if (not np.array_equal(arrays['rgb'],stored['rgb_bgra'][...,[2,1,0]])
                or not np.array_equal(arrays['depth_m'],decode_carla_depth_bgra(stored['depth_bgra']).astype(np.float32))):
            raise ValueError('saved RGB/depth arrays disagree with raw camera bytes')
    observation=Observation(stamp=stamp,modality_stamps={k:FrameStamp(**v) if isinstance(v,dict) else v
        for k,v in modality_stamps.items()},**arrays,depth_mode=fixture['depth_mode'],
        depth_fixture_sha256=depth_fixture_sha256,depth_qualified=fixture['qualified'],scene_actor_ids=tuple(scene_actor_ids))
    _input(observation)
    return observation


@dataclass(frozen=True)
class FreeSpaceConfig:
    station_cell_m: float = 1.
    lateral_cell_m: float = .25
    maximum_ground_residual_m: float = .05
    maximum_cell_plane_residual_m: float = .025
    minimum_source_pixels: int = 4
    minimum_crop_pixels: int = 2
    maximum_age_s: float = 5.
    maximum_depth_m: float = 80.
    minimum_static_history_s: float = .5
    maximum_static_displacement_m: float = .03
    maximum_static_speed_mps: float = .05
    maximum_static_yaw_rate_rps: float = .01
    maximum_static_heading_change_rad: float = .01

    def __post_init__(self):
        if (any(not np.isfinite(v) or v<=0 for v in asdict(self).values())
                or self.station_cell_m>1. or self.lateral_cell_m>.25
                or self.maximum_ground_residual_m>.05 or self.maximum_age_s>5.
                or self.maximum_cell_plane_residual_m>.025 or self.minimum_static_history_s<.5
                or self.maximum_static_displacement_m>.03 or self.maximum_static_speed_mps>.05
                or self.maximum_static_yaw_rate_rps>.01 or self.maximum_static_heading_change_rad>.01
                or type(self.minimum_source_pixels) is not int or type(self.minimum_crop_pixels) is not int
                or self.minimum_source_pixels<4 or self.minimum_crop_pixels<2):
            raise ValueError('invalid bounded free-surface evidence settings')


def _reference_sha(road):
    from .passing import reference_fingerprint
    if not isinstance(road,ObservedRoad) or not isinstance(road.reference,ObservableReference):
        raise TypeError('free surface requires independently observed paired road edges')
    if road.left_role!='road_edge' or road.right_role!='road_edge':
        raise ValueError('whole-road edge roles required')
    return reference_fingerprint(road.reference)


def _inside(points,polygon):
    edges=np.roll(polygon,-1,axis=0)-polygon
    d=points[:,None,:]-polygon[None,:,:]
    cross=edges[None,:,0]*d[:,:,1]-edges[None,:,1]*d[:,:,0]
    return np.all(cross>=-1e-8,axis=1)|np.all(cross<=1e-8,axis=1)


def _raster(polygon,width,height,offset):
    lo=np.maximum(0,np.ceil(polygon.min(0)-offset).astype(int))
    hi=np.minimum([width-1,height-1],np.floor(polygon.max(0)-offset).astype(int))
    if np.any(hi<lo):return np.empty((0,2),int)
    x,y=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1))
    points=np.c_[x.ravel(),y.ravel()]
    return points[_inside(points+offset,polygon)]


def _input(observation):
    if (set(observation.modality_stamps)!=set(MODALITIES)
            or any(value!=observation.stamp for value in observation.modality_stamps.values())):
        raise ValueError('free surface modalities must share exact frame time and camera')
    if (observation.semantic_schema!=SEMANTIC_SCHEMA or not observation.depth_qualified
            or observation.depth_mode not in ('forward','radial')
            or len(observation.depth_fixture_sha256)!=64
            or any(c not in '0123456789abcdef' for c in observation.depth_fixture_sha256)):
        raise ValueError('qualified calibrated depth fixture and semantic schema required')
    rgb,depth=np.asarray(observation.rgb),np.asarray(observation.depth_m)
    if depth.shape!=rgb.shape[:2] or depth.dtype.kind!='f':raise ValueError('unaligned raw free-surface arrays')
    semantic,instance=decode_carla_instance_bgra(observation.instance_bgra)
    if (not np.array_equal(semantic,observation.semantic_ids) or not np.array_equal(instance,observation.instance_ids)
            or semantic.shape!=depth.shape):raise ValueError('free surface raw instance/semantic replay mismatch')
    if not np.array_equal(pack_model_rgb(rgb,observation.model_to_source),observation.packed):
        raise ValueError('free surface actual packed crop replay mismatch')
    # Reuse camera validation without inventing any depth samples.
    visible_surface_points(observation,np.empty((0,2),int))
    return dict(stamp=asdict(observation.stamp),modality_stamps={k:asdict(v) for k,v in observation.modality_stamps.items()},
        array_sha256={name:array_sha256(getattr(observation,name)) for name in ('rgb','depth_m','semantic_ids',
            'instance_ids','instance_bgra','packed','intrinsics','camera_from_world','model_to_source')},
        depth_fixture_sha256=observation.depth_fixture_sha256,depth_mode=observation.depth_mode,
        pixel_center_offset=observation.pixel_center_offset,scene_actor_ids=list(observation.scene_actor_ids))


def _cell(road,lo,hi,near,far):
    q=np.array([lo,hi]);left=road.reference._left(q);right=road.reference._right(q)
    width=np.linalg.norm((right-left)[:,:2],axis=1)
    if np.any(far>=width*.5):raise ValueError('requested free strip reaches the physical opposing curb')
    across=(right-left)/width[:,None];center=(left+right)*.5
    return np.array([center[0]-across[0]*near,center[1]-across[1]*near,
                     center[1]-across[1]*far,center[0]-across[0]*far])


def _measure_cell(observation,corners,cfg,lookup):
    transform=np.asarray(observation.camera_from_world);optical=corners@transform[:3,:3].T+transform[:3,3]
    if np.any(optical[:,2]<=0):return dict(supported=False,reason='cell_behind_camera')
    uv=optical@np.asarray(observation.intrinsics).T;uv=uv[:,:2]/uv[:,2:3]
    shape=observation.depth_m.shape;offset=observation.pixel_center_offset
    if np.any(uv<[offset,offset]) or np.any(uv>[shape[1]-1+offset,shape[0]-1+offset]):
        return dict(supported=False,reason='whole_cell_outside_source')
    model=np.c_[uv-offset,np.ones(4)]@np.linalg.inv(observation.model_to_source).T
    model=model[:,:2]/model[:,2:3]
    if not np.isfinite(model).all() or np.any(model<0) or np.any(model>[511,255]):
        return dict(supported=False,reason='whole_cell_outside_actual_crop')
    source=_raster(uv,shape[1],shape[0],offset);crop=_raster(model,512,256,0.)
    ix,iy,inside=lookup
    if len(source)<cfg.minimum_source_pixels or len(crop)<cfg.minimum_crop_pixels:
        return dict(supported=False,reason='cell_unresolved_at_source_or_model_resolution',source_pixels=len(source),crop_pixels=len(crop))
    if not np.all(inside[crop[:,1],crop[:,0]]):return dict(supported=False,reason='clamped_model_crop')
    selected=np.unique(np.r_[source,np.c_[ix[crop[:,1],crop[:,0]],iy[crop[:,1],crop[:,0]]]],axis=0)
    x,y=selected.T
    if not np.all(np.isin(observation.semantic_ids[y,x],[1,24])):
        return dict(supported=False,reason='cell_occluded_or_nonroad',source_pixels=len(source),crop_pixels=len(crop))
    values=observation.depth_m[y,x]
    if np.any(~np.isfinite(values)|(values<=0)|(values>cfg.maximum_depth_m)):
        return dict(supported=False,reason='cell_depth_unresolved')
    plane=np.cross(corners[1]-corners[0],corners[3]-corners[0]);plane/=np.linalg.norm(plane)
    if np.max(np.abs((corners-corners[0])@plane))>cfg.maximum_cell_plane_residual_m:
        return dict(supported=False,reason='observed_road_cell_not_planar')
    surface=visible_surface_points(observation,selected)
    residual=np.abs((surface-corners[0])@plane)
    if np.max(residual)>cfg.maximum_ground_residual_m:
        return dict(supported=False,reason='cell_depth_not_observed_ground',maximum_ground_residual_m=float(np.max(residual)))
    return dict(supported=True,source_pixels=len(source),crop_pixels=len(crop),
        source_pixels_xy=selected.tolist(),maximum_ground_residual_m=float(np.max(residual)),
        source_surface_xyz_sha256=array_sha256(surface),corners_world_xyz=corners.tolist())


def measure_free_surface(observation,road,*,opposing_width_m=1.,station_interval_m=None,config=None):
    """Measure cells from one actual crop; grants no actor-inventory permission."""
    cfg=config or FreeSpaceConfig();binding=_input(observation);reference_sha=_reference_sha(road)
    if not np.isfinite(opposing_width_m) or not 0<opposing_width_m<=2.:
        raise ValueError('bounded explicit opposing strip width required')
    lo,hi=map(float,station_interval_m if station_interval_m is not None else (road.reference.s[0],road.reference.s[-1]))
    if not np.isfinite([lo,hi]).all() or not road.reference.s[0]<=lo<hi<=road.reference.s[-1]:
        raise ValueError('free strip cannot extrapolate observed road support')
    station=np.linspace(lo,hi,int(np.ceil((hi-lo)/cfg.station_cell_m))+1)
    lateral=np.linspace(0.,opposing_width_m,int(np.ceil(opposing_width_m/cfg.lateral_cell_m))+1)
    lookup=_lookup(observation.model_to_source,observation.depth_m.shape);cells=[]
    for i,(a,b) in enumerate(zip(station[:-1],station[1:])):
        for j,(c,d) in enumerate(zip(lateral[:-1],lateral[1:])):
            result=_measure_cell(observation,_cell(road,a,b,c,d),cfg,lookup)
            cells.append(dict(station_index=i,lateral_index=j,interval_m=[float(a),float(b)],
                opposing_interval_m=[float(c),float(d)],**result))
    result=dict(version=VERSION,reference_sha256=reference_sha,input=binding,config=asdict(cfg),
        opposing_width_m=opposing_width_m,station_edges_m=station.tolist(),lateral_edges_m=lateral.tolist(),
        cells=cells,scope='rendered_surface_cells_only_no_traffic_permission',training_ready=False)
    result['proof_sha256']=_digest(result)
    return result


def _controlled(observations,scenes,decision,obstacle_ids,cfg):
    if len(observations)!=len(scenes) or not observations:raise ValueError('aligned complete scene inventory missing')
    if len(obstacle_ids)!=1:raise ValueError('bounded passing permission requires one static parked vehicle')
    actor_id=int(obstacle_ids[0]);positions=[];headings=[];times=[];identities=[];ego_id=None
    for observation,scene in zip(observations,scenes):
        if (scene.get('schema')!='controlled_native_static_scene_v1'
                or scene.get('scope')!='simulator_scene_QA_not_visual_free_space'
                or scene.get('frame_id')!=observation.stamp.frame_id
                or abs(scene.get('timestamp_s',float('inf'))-observation.stamp.timestamp_s)>1e-8):
            raise ValueError('controlled scene and camera do not share exact frame/time')
        actors=scene['actors'];ids=[a['actor_id'] for a in actors]
        if (len(set(ids))!=len(ids) or set(ids)!=set(observation.scene_actor_ids)
                or any(type(v) is not int or not 0<v<=65535 for v in ids)):
            raise ValueError('controlled inventory differs from complete scene actor identities')
        if scene['static_obstacle_actor_ids']!=[actor_id]:raise ValueError('controlled static obstacle identity changed')
        if ego_id is None:ego_id=scene['ego_actor_id']
        if ego_id!=scene['ego_actor_id'] or ego_id==actor_id:raise ValueError('controlled ego identity changed')
        traffic={a['actor_id']:a for a in actors if a['type_id'].startswith(('vehicle.','walker.'))}
        if any(not a['type_id'].startswith(('vehicle.','walker.','sensor.','traffic.'))
                and a['type_id']!='spectator' for a in actors):
            raise ValueError('controlled scene contains an unsupported actor class')
        if set(traffic)!={ego_id,actor_id} or not traffic[actor_id]['type_id'].startswith('vehicle.'):
            raise ValueError('controlled passing requires exactly ego and parked vehicle, no other traffic')
        state=traffic[actor_id]['state']
        if state.get('actor_id')!=actor_id or state.get('source')!='simulator_actor_state_qa':
            raise ValueError('parked actor inventory state binding changed')
        position=np.asarray(state['position_world_m'],float);velocity=np.asarray(state['velocity_world_mps'],float)
        if (position.shape!=(3,) or velocity.shape!=(3,) or not np.isfinite(position).all()
                or not np.isfinite(velocity).all() or not np.isfinite([state['yaw_rate_rps'],state['yaw_rad']]).all()
                or np.linalg.norm(velocity)>cfg.maximum_static_speed_mps
                or abs(state['yaw_rate_rps'])>cfg.maximum_static_yaw_rate_rps):
            raise ValueError('parked actor is moving or has invalid measured state')
        positions.append(position);headings.append(state['yaw_rad']);times.append(observation.stamp.timestamp_s)
        identities.append(_digest(scene))
    if len(set(times))<3 or max(times)-min(times)<cfg.minimum_static_history_s-1e-6:
        raise ValueError('controlled static history is insufficient')
    if np.max(np.linalg.norm(np.asarray(positions)-positions[0],axis=1))>cfg.maximum_static_displacement_m:
        raise ValueError('parked actor moved during free-space history')
    if np.max(np.abs((np.asarray(headings)-headings[0]+np.pi)%(2*np.pi)-np.pi))>cfg.maximum_static_heading_change_rad:
        raise ValueError('parked actor rotated during free-space history')
    return _digest(dict(scope='complete_controlled_scene_QA_only',scene_sha256=identities,ego_actor_id=ego_id,
        obstacle_actor_ids=[actor_id],decision=asdict(decision)))


def build_free_space_proof(observations,road,*,decision,controlled_scenes,obstacle_ids,
                          opposing_width_m=1.,station_interval_m=None,config=None):
    cfg=config or FreeSpaceConfig();observations=list(observations);scenes=list(controlled_scenes)
    reference_sha=_reference_sha(road)
    if abs(road.reference.decision_time_s-decision.timestamp_s)>1e-8:
        raise ValueError('free surface road belongs to another decision time')
    if not any(o.stamp.frame_id==decision.frame_id and abs(o.stamp.timestamp_s-decision.timestamp_s)<1e-8 for o in observations):
        raise ValueError('current decision observation required')
    seen=set()
    for observation in observations:
        stamp=observation.stamp;key=(stamp.frame_id,stamp.camera)
        if key in seen:raise ValueError('duplicate free-surface camera frame')
        seen.add(key);age=decision.timestamp_s-stamp.timestamp_s
        if age< -1e-8 or age>cfg.maximum_age_s+1e-8 or stamp.frame_id>decision.frame_id:
            raise ValueError('future or stale free-surface evidence')
    for evidence in road.reference.evidence_for_arc(road.reference.s[0],road.reference.s[-1]):
        if (evidence.frame_id>decision.frame_id or decision.timestamp_s-evidence.timestamp_s>cfg.maximum_age_s+1e-8
                or evidence.timestamp_s>decision.timestamp_s+1e-8):raise ValueError('road support is not causal within five seconds')
    controlled_sha=_controlled(observations,scenes,decision,obstacle_ids,cfg)
    measured=[measure_free_surface(o,road,opposing_width_m=opposing_width_m,station_interval_m=station_interval_m,config=cfg)
              for o in observations]
    chosen={}
    for frame in measured:
        for cell in frame['cells']:
            if cell['supported']:chosen[(cell['station_index'],cell['lateral_index'])]=dict(
                frame_proof_sha256=frame['proof_sha256'],stamp=frame['input']['stamp'],cell=cell)
    station=measured[0]['station_edges_m'];lateral=measured[0]['lateral_edges_m'];intervals=[];used=[]
    for i in range(len(station)-1):
        if not all((i,j) in chosen for j in range(len(lateral)-1)):continue
        interval=[station[i],station[i+1]]
        if intervals and abs(intervals[-1][1]-interval[0])<1e-8:intervals[-1][1]=interval[1]
        else:intervals.append(interval)
        used.extend(chosen[(i,j)] for j in range(len(lateral)-1))
    evidence=sorted({(u['stamp']['frame_id'],u['stamp']['timestamp_s'],u['stamp']['camera']) for u in used})
    result=dict(version=VERSION,decision=asdict(decision),reference_sha256=reference_sha,config=asdict(cfg),
        controlled_scene_sha256=controlled_sha,controlled_scenes=scenes,obstacle_ids=list(obstacle_ids),
        opposing_width_m=opposing_width_m,station_interval_m=station_interval_m,
        allowed_intervals_m=intervals,supported=bool(intervals),reasons=[] if intervals else ['no_complete_rendered_free_strip_cells'],
        evidence_frames=[dict(frame_id=f,timestamp_s=t,camera=c) for f,t,c in evidence],
        frame_proofs=measured,used_cells=used,scope='controlled_static_parked_vehicle_only_not_traffic_gap_prediction',
        dynamic_traffic_permitted=False,training_ready=False)
    result['proof_sha256']=_digest(result)
    return result


def verify_free_space_proof(proof,observations,road,**kwargs):
    replay=build_free_space_proof(observations,road,**kwargs)
    if _digest(proof)!=_digest(replay):raise ValueError('free-space proof differs from immutable raw replay')
    return replay


def observed_passing_space(proof,road):
    """Call after raw proof replay; the digest also catches in-memory mutation."""
    from .passing import ObservedPassingSpace
    document=dict(proof);claimed=document.pop('proof_sha256',None)
    if _digest(document)!=claimed or proof['reference_sha256']!=_reference_sha(road):
        raise ValueError('free-space proof or observed road binding changed')
    if not proof['supported']:raise ValueError('no observed free-surface interval qualified')
    return ObservedPassingSpace(reference_sha256=proof['reference_sha256'],
        allowed_intervals_m=tuple(tuple(v) for v in proof['allowed_intervals_m']),opposing_width_m=proof['opposing_width_m'],
        obstacle_ids=tuple(proof['obstacle_ids']),controlled_scene_sha256=proof['controlled_scene_sha256'],
        rendered_free_space_sha256=claimed,evidence_frames=tuple(EvidenceFrame(**v) for v in proof['evidence_frames']))
