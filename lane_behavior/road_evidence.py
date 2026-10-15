"""Offline replay of cropped, causal, rendered whole-road edge evidence.

Map samples define QA cross-sections and stable correspondence only. A knot is
retained only after the existing strict RGB/semantic/depth measurement succeeds
and its transition survives the actual packed-camera crop. Independent splines
use these measured knots exclusively, with no hidden map derivatives or edge
extrapolation. This is road evidence, not maneuver or training admission.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from lane_dataset.config import sha256_file, write_json
from lane_dataset.geometry import EvidenceFrame, LaneReference, ObservableReference
from lane_dataset.preprocessing import image_to_nv12, split_nv12
from lane_dataset.rendered_boundaries import edge_center, measure_camera
from lane_dataset.visibility import decode_carla_depth_bgra, project_world_points
from .evidence import FrameStamp, _lookup, array_sha256, pack_model_rgb
from .target import ObservedRoad

VERSION = "causal_strict_rendered_road_edges_and_native_crop_v1"
SOURCE_SPACING_M = .5


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class FrameRoadEvidence:
    """Produced by raw-payload replay; persist audit and replay before reuse."""
    stamp: FrameStamp
    reference_sha256: str
    knots: tuple[dict, ...]
    audit: dict
    proof_sha256: str


@dataclass(frozen=True)
class RoadAssembly:
    road: ObservedRoad | None
    reasons: tuple[str, ...]
    audit: dict


def _crop_edge_check(data, gray, reference, sample, offsets, depth_mode):
    """Replay transition on pixels actually selected by the stock model warp.

The <=1 cm/source-pixel test remains the original strict measurement. This
additional test proves crop availability and reuses contrast/depth/5 cm limits;
it does not replace source measurement precision with model pixel resolution.
"""
    q=sample['route_s']
    normal=reference._right(q)-reference._left(q)
    normal/=np.linalg.norm(normal)
    point=np.asarray(sample['point_world'])
    world=point+offsets[:,None]*normal
    pixels,optical=project_world_points(world,data['camera_from_world'],data['intrinsics'])
    source=np.c_[pixels-.5,np.ones(len(pixels))]
    projected=source@np.linalg.inv(data['model_to_source']).T
    uv=projected[:,:2]/projected[:,2:3]
    finite=np.isfinite(uv).all(1)&(optical[:,2]>0)
    xy=np.rint(np.where(finite[:,None],uv,-100.)).astype(int)
    inside=finite&(xy[:,0]>=0)&(xy[:,0]<512)&(xy[:,1]>=0)&(xy[:,1]<256)
    ix,iy,valid=_lookup(data['model_to_source'],gray.shape)
    indices=np.flatnonzero(inside)
    indices=indices[valid[xy[indices,1],xy[indices,0]]]
    sx=ix[xy[indices,1],xy[indices,0]];sy=iy[xy[indices,1],xy[indices,0]]
    tags=data['semantic_camera_ids'][sy,sx]
    road=np.zeros(len(offsets),bool);exterior=road.copy();luma=np.zeros(len(offsets))
    road[indices]=np.isin(tags,[1,24]);exterior[indices]=np.isin(tags,[2,10,25]);luma[indices]=gray[sy,sx]
    plane=np.cross(reference._center(q,1),normal);plane/=np.linalg.norm(plane)
    plane=data['camera_from_world'][:3,:3]@plane
    distance=float(plane@optical[len(offsets)//2])
    rays=np.c_[sx+.5,sy+.5,np.ones(len(sx))]@np.linalg.inv(data['intrinsics']).T
    denominator=rays@plane
    forward=np.divide(distance,denominator,out=np.full(len(sx),np.inf),where=np.abs(denominator)>1e-8)
    expected=forward if depth_mode=='forward' else forward*np.linalg.norm(rays,axis=1)
    residual=np.full(len(offsets),np.nan);residual[indices]=data['depth_m'][sy,sx]-expected
    result=edge_center(offsets,road,exterior,luma,residual,sample['side'])
    if result['measured'] and abs(result['offset_m'])>.05+1e-9:
        return dict(measured=False,reason='crop_road_edge_disagreement_above_five_cm')
    return result


def measure_road_frame(*, payload, payload_sha256, stamp, modality_stamps,
                       semantic_sensor_stamp, depth_fixture, depth_fixture_sha256,
                       reference, current_s):
    """Replay one saved frame without simulator calls or accepted-data writes.

``reference`` must be the physical whole-road edges, not the virtual right lane.
``current_s`` is snapped down to a fixed 0.5 m lattice solely for QA query
identity. Returned XYZ includes the measured edge offset, not the declared edge.
"""
    payload=Path(payload);depth_fixture=Path(depth_fixture)
    if sha256_file(payload)!=payload_sha256:
        raise ValueError('raw road payload hash mismatch')
    if sha256_file(depth_fixture)!=depth_fixture_sha256:
        raise ValueError('road depth fixture hash mismatch')
    for key in ('rgb','depth','packed'):
        actual=modality_stamps.get(key)
        if (FrameStamp(**actual) if isinstance(actual,dict) else actual)!=stamp:
            raise ValueError('unaligned road modality stamps')
    semantic=FrameStamp(**semantic_sensor_stamp) if isinstance(semantic_sensor_stamp,dict) else semantic_sensor_stamp
    if semantic!=stamp:
        raise ValueError('unaligned road semantic sensor stamp')
    fixture=json.loads(depth_fixture.read_text(encoding='utf-8'))[stamp.camera]
    if fixture.get('qualified') is not True or fixture.get('depth_mode') not in ('forward','radial'):
        raise ValueError('road depth fixture is not qualified')
    if not np.isfinite(current_s):
        raise ValueError('invalid road query origin')
    with np.load(payload,allow_pickle=False) as saved:
        data={key:saved[key] for key in ('rgb','rgb_bgra','depth_m','depth_bgra','semantic_camera_ids',
            'packed','intrinsics','camera_from_world','model_to_source')}
    shape=data['rgb'].shape[:2]
    if (data['rgb'].dtype!=np.uint8 or data['rgb'].shape!=(*shape,3)
            or data['depth_m'].shape!=shape or data['semantic_camera_ids'].shape!=shape
            or not np.array_equal(data['rgb'],data['rgb_bgra'][...,[2,1,0]])
            or not np.array_equal(data['depth_m'],decode_carla_depth_bgra(data['depth_bgra']).astype(np.float32))):
        raise ValueError('road raw modalities do not replay')
    if not np.array_equal(pack_model_rgb(data['rgb'],data['model_to_source']),data['packed']):
        raise ValueError('road packed model frame does not replay')
    gray=split_nv12(*image_to_nv12(data['rgb'],'RGB'))[0]
    origin=float(reference.s[0])
    snapped=origin+np.floor((current_s-origin)/SOURCE_SPACING_M)*SOURCE_SPACING_M
    measured=measure_camera(reference=reference,current_s=snapped,tags=data['semantic_camera_ids'],
        depth=data['depth_m'],camera_from_world=data['camera_from_world'],intrinsics=data['intrinsics'],
        depth_mode=fixture['depth_mode'],road_line_tag=24,luma=gray,road_tag=1,exterior_tags=[2,10,25],
        pixel_center_offset=.5)
    binding=_digest(dict(left=array_sha256(reference.left_xyz),right=array_sha256(reference.right_xyz),
        source_arc=array_sha256(reference.s),spacing_m=SOURCE_SPACING_M))
    rejected=Counter();knots=[];strict_counts=Counter()
    for sample in measured['samples']:
        if not sample['measured']:
            rejected[sample['reason']]+=1;continue
        strict_counts[sample['side']]+=1
        if sample.get('detector')!='road_edge':
            rejected['paint_is_not_unmarked_road_edge']+=1;continue
        if abs(sample['offset_m'])>.05+1e-9:
            rejected['rendered_boundary_disagreement_above_five_cm']+=1;continue
        crop=_crop_edge_check(data,gray,reference,sample,np.array(measured['offsets_m']),fixture['depth_mode'])
        if not crop['measured']:
            rejected['model_crop:'+crop['reason']]+=1;continue
        q=sample['route_s'];identity=int(round((q-origin)/SOURCE_SPACING_M))
        if abs(origin+identity*SOURCE_SPACING_M-q)>1e-7:
            raise ValueError('road query identity is not on fixed source lattice')
        normal=reference._right(q)-reference._left(q);normal/=np.linalg.norm(normal)
        xyz=np.asarray(sample['point_world'])+normal*sample['offset_m']
        knots.append(dict(side=sample['side'],source_index=identity,source_route_s=q,
            observed_xyz=xyz.tolist(),strict_measurement=sample,crop_measurement=crop))
    audit=dict(version=VERSION,stamp=asdict(stamp),payload=str(payload.resolve()),payload_sha256=payload_sha256,
        depth_fixture=str(depth_fixture.resolve()),depth_fixture_sha256=depth_fixture_sha256,
        reference_sha256=binding,source_identity_origin_m=origin,source_spacing_m=SOURCE_SPACING_M,
        snapped_query_origin_m=snapped,strict_measured_by_side=dict(strict_counts),
        retained_by_side=dict(Counter(k['side'] for k in knots)),rejections=dict(rejected),
        packed_sha256=array_sha256(data['packed']),knots=knots)
    return FrameRoadEvidence(stamp,binding,tuple(knots),audit,_digest(audit))


def assemble_observed_road(frames, *, decision, position_world, maximum_age_s=5., maximum_gap_m=8.):
    """Retain only common support of independently fitted causal edge runs.

Use freshly replayed ``measure_road_frame`` results; persisted audits alone are
not an admission API. Missing edges mask this target, not unrelated recordings.
"""
    frames=tuple(frames)
    if not 0<maximum_age_s<=5 or not 0<maximum_gap_m<=8:
        raise ValueError('road evidence limits cannot exceed five seconds/eight metres')
    position=np.asarray(position_world,float)
    if position.shape not in ((2,),(3,)) or not np.isfinite(position).all():
        raise ValueError('invalid road decision position')
    audit=dict(version=VERSION,decision=asdict(decision),maximum_age_s=maximum_age_s,
        maximum_gap_m=maximum_gap_m,frame_proofs=[],excluded_frames=[],training_ready=False)
    causal=[]
    for frame in frames:
        if (_digest(frame.audit)!=frame.proof_sha256 or frame.audit['knots']!=list(frame.knots)
                or frame.audit['stamp']!=asdict(frame.stamp) or frame.audit['reference_sha256']!=frame.reference_sha256):
            raise ValueError('road frame proof was changed after replay')
        age=decision.timestamp_s-frame.stamp.timestamp_s
        if age < -1e-8 or frame.stamp.frame_id>decision.frame_id or age>maximum_age_s+1e-8:
            audit['excluded_frames'].append(dict(stamp=asdict(frame.stamp),reason='future_or_stale'))
        else:
            causal.append(frame);audit['frame_proofs'].append(frame.proof_sha256)
    if len({f.reference_sha256 for f in causal})>1:
        raise ValueError('road evidence mixes source identities or geometry references')
    if not any(f.stamp.frame_id==decision.frame_id and abs(f.stamp.timestamp_s-decision.timestamp_s)<1e-8 for f in causal):
        return RoadAssembly(None,('current_decision_frame_not_bound',),audit)
    by_side={'left':{},'right':{}}
    for frame in sorted(causal,key=lambda f:(f.stamp.timestamp_s,f.stamp.camera)):
        evidence=EvidenceFrame(frame.stamp.frame_id,frame.stamp.timestamp_s,frame.stamp.camera)
        for knot in frame.knots:
            by_side[knot['side']][knot['source_index']]=(np.array(knot['observed_xyz']),evidence,frame.proof_sha256)
    audit['distinct_observed_knots']={side:len(points) for side,points in by_side.items()}
    runs={}
    for side,points in by_side.items():
        ids=np.array(sorted(points),int)
        xyz=np.array([points[i][0] for i in ids]).reshape(-1,3)
        gaps=np.linalg.norm(np.diff(xyz,axis=0),axis=1)
        runs[side]=[r for r in np.split(ids,np.flatnonzero(gaps>maximum_gap_m)+1) if len(r)>=4]
    audit['qualified_run_sizes']={side:[len(r) for r in values] for side,values in runs.items()}
    if not all(runs.values()):
        return RoadAssembly(None,('insufficient_independent_boundary_observations',),audit)
    candidates=[]
    for left_ids in runs['left']:
        for right_ids in runs['right']:
            lo=max(left_ids[0],right_ids[0]);hi=min(left_ids[-1],right_ids[-1])
            grid=np.union1d(left_ids,right_ids);grid=grid[(grid>=lo)&(grid<=hi)]
            if len(grid)<4:continue
            left=CubicSpline(left_ids,[by_side['left'][i][0] for i in left_ids],axis=0)(grid)
            right=CubicSpline(right_ids,[by_side['right'][i][0] for i in right_ids],axis=0)(grid)
            center=(left+right)*.5
            score=float(np.min(np.linalg.norm(center[:,:2]-position[:2],axis=1)))
            candidates.append((score,left_ids,right_ids,grid,left,right))
    if not candidates:
        return RoadAssembly(None,('observed_boundaries_have_no_common_corridor',),audit)
    _,left_ids,right_ids,grid,left,right=min(candidates,key=lambda row:row[0])
    refs=tuple(dict.fromkeys(by_side[side][i][1] for side,ids in (('left',left_ids),('right',right_ids)) for i in ids))
    center=(left+right)*.5;arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(center,axis=0),axis=1))]
    reference=ObservableReference(left,right,arc,[refs]*len(grid),decision.timestamp_s)
    reference.source_indices=grid;reference.source_left_indices=left_ids;reference.source_right_indices=right_ids
    reference.maximum_marking_gap_m=maximum_gap_m;reference.support_method=VERSION
    audit.update(source_indices=grid.tolist(),source_left_indices=left_ids.tolist(),source_right_indices=right_ids.tolist(),
        observed_arc_m=arc.tolist(),left_xyz=left.tolist(),right_xyz=right.tolist(),
        used_frame_proofs=sorted(set(by_side[side][i][2] for side,ids in (('left',left_ids),('right',right_ids)) for i in ids)))
    return RoadAssembly(ObservedRoad(reference),(),audit)


def replay_probe(source, plan, output):
    """Demonstrate support or a precise refusal from immutable native payloads."""
    source=Path(source);plan=Path(plan)
    report=json.loads((source/'probe-report.json').read_text(encoding='utf-8'))
    task=json.loads(plan.read_text(encoding='utf-8'))['tasks'][0]
    route=task['route'];physical=route['unmarked_road']
    reference=LaneReference(physical['road_left_edge_xyz'],physical['road_right_edge_xyz'],route['arc_m'])
    fixture=source/'depth-fixture.json'
    if sha256_file(fixture)!=report['depth_fixture_sha256']:
        raise ValueError('probe depth fixture changed')
    frames=[]
    for record in report['frame_records']:
        current_s=reference.project(record['ego']['position_world_m'])[0]
        for camera,meta in record['cameras'].items():
            frames.append(measure_road_frame(payload=source/meta['payload'],payload_sha256=meta['payload_sha256'],
                stamp=FrameStamp(record['frame_id'],record['timestamp_s'],camera),modality_stamps=meta['modality_stamps'],
                semantic_sensor_stamp=meta['separate_semantic_sensor_stamp'],depth_fixture=fixture,
                depth_fixture_sha256=report['depth_fixture_sha256'],reference=reference,current_s=current_s))
    last=report['frame_records'][-1]
    result=assemble_observed_road(frames,decision=FrameStamp(last['frame_id'],last['timestamp_s'],'narrow'),
        position_world=last['ego']['position_world_m'])
    document=dict(source_probe=str(source),source_probe_sha256=sha256_file(source/'probe-report.json'),
        plan=str(plan),plan_sha256=sha256_file(plan),supported=result.road is not None,reasons=list(result.reasons),
        frames=[dict(proof_sha256=f.proof_sha256,**f.audit) for f in frames],assembly=result.audit,
        training_ready=False)
    write_json(output,document)
    return document


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=replay_probe(args.source,args.plan,args.output)
    print(json.dumps(dict(supported=result['supported'],reasons=result['reasons'],
        distinct_observed_knots=result['assembly'].get('distinct_observed_knots'),output=str(args.output)),indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
