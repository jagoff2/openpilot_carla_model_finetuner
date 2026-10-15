"""Offline native curb diagnosis, explicitly not a production acceptance gate.

The semantic Road/Ground seam can lie on a flat paved gutter. A visible curb
face is a different physical surface. This diagnostic fits its qualified-depth
point cloud and compares RGB on that face with adjacent pavement, including
the exact stock model crop. Collision-surveyed toe agreement remains required
before any geometry from this experiment may support labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from lane_dataset.config import sha256_file,write_json
from lane_dataset.geometry import LaneReference
from lane_dataset.preprocessing import image_to_nv12,split_nv12
from lane_dataset.visibility import project_world_points,decode_carla_depth_bgra
from .evidence import FrameStamp,_lookup,pack_model_rgb,array_sha256
from .road_evidence import FrameRoadEvidence,_digest
from .curb_surface import detect_curb_step

EVIDENCE_VERSION='independent_survey_curb_face_and_lip_rgb_depth_native_crop_v2'


def _cropped_surface_points(data):
    x,y,inside=_lookup(data['model_to_source'],data['depth_m'].shape)
    xy=np.unique(np.c_[x[inside],y[inside]],axis=0)
    rays=np.c_[xy+.5,np.ones(len(xy))]@np.linalg.inv(data['intrinsics']).T
    optical=rays*data['depth_m'][xy[:,1],xy[:,0]][:,None]
    inverse=np.linalg.inv(data['camera_from_world'])
    world=optical@inverse[:3,:3].T+inverse[:3,3]
    gray=split_nv12(*image_to_nv12(data['rgb'],'RGB'))[0]
    return xy,world,data['semantic_camera_ids'][xy[:,1],xy[:,0]],gray[xy[:,1],xy[:,0]]


def _cropped_pixel_colors(data,xy):
    """Bind each sampled source pixel to actual native packed Y/U/V values."""
    sx,sy,inside=_lookup(data['model_to_source'],data['depth_m'].shape)
    width=data['depth_m'].shape[1]
    valid=np.flatnonzero(inside.ravel())
    keys,first=np.unique((sy*width+sx).ravel()[valid],return_index=True)
    wanted=xy[:,1]*width+xy[:,0]
    indices=np.searchsorted(keys,wanted)
    if np.any(indices>=len(keys)) or not np.array_equal(keys[indices],wanted):
        raise ValueError('curb color pixels are absent from actual model crop')
    my,mx=np.unravel_index(valid[first[indices]],sx.shape)
    packed=data['packed']
    colors=np.stack((packed[(mx%2)*2+my%2,my//2,mx//2],
        packed[4,my//2,mx//2],packed[5,my//2,mx//2]),axis=1)
    return data['rgb'][xy[:,1],xy[:,0]],colors,np.c_[mx,my]


def _upper_lip_color_proof(*,xy,tags,lateral,along,height,sign,toe_offset,colors):
    """Observe the raised lip/verge edge when sunlit face and road match.

    Both the curb top and adjacent level verge must exist in qualified depth
    and in the exact model crop. RGB and packed YUV contrasts must each reach
    12 in both longitudinal halves, with consistent color direction. Poles,
    vehicles, vegetation, and arbitrary static objects cannot supply the verge.
    This photometric alternative never substitutes for the fitted curb face.
    """
    rgb,packed,model_xy=colors
    outward=sign*(lateral-toe_offset)
    lip=(abs(along)<.8)&(outward>.015)&(outward<.20)&(height>.14)&(height<.175)&np.isin(tags,[2,25])
    verge=(abs(along)<.8)&(outward>.25)&(outward<.8)&(height>.10)&(height<.20)&np.isin(tags,[2,10,25])
    result=dict(method='observed_upper_lip_to_level_verge_source_RGB_and_native_packed_YUV',
        passed=False,contrast_threshold=12.,lip_samples=int(lip.sum()),verge_samples=int(verge.sum()))
    if min(lip.sum(),verge.sum())<6:
        return dict(result,reason='upper_lip_or_level_verge_unavailable')
    if min(np.ptp(along[lip]),np.ptp(along[verge]))<.5 or np.ptp(outward[lip])<.08:
        return dict(result,reason='upper_lip_or_verge_extent_unresolved')
    matrix=np.c_[outward[lip],along[lip],np.ones(lip.sum())]
    if np.linalg.matrix_rank(matrix)<3:
        return dict(result,reason='upper_lip_plane_rank_deficient')
    fit=np.linalg.lstsq(matrix,height[lip],rcond=None)[0]
    residual=float(abs(height[lip]-matrix@fit).max())
    if residual>.005 or abs(fit[0])>.2 or abs(fit[1])>.1:
        return dict(result,reason='upper_lip_depth_plane_unresolved',maximum_lip_plane_residual_m=residual)
    if abs(float(np.median(height[lip])-np.median(height[verge])))>.03:
        return dict(result,reason='adjacent_verge_not_level_with_observed_lip')
    halves=[];directions=[]
    for half in (-1.,1.):
        a,b=lip&(along*half>=0),verge&(along*half>=0)
        if min(a.sum(),b.sum())<3:
            return dict(result,reason='upper_lip_edge_not_supported_in_both_longitudinal_halves')
        source_delta=np.median(rgb[a].astype(float),axis=0)-np.median(rgb[b].astype(float),axis=0)
        packed_delta=np.median(packed[a].astype(float),axis=0)-np.median(packed[b].astype(float),axis=0)
        contrasts=[float(np.linalg.norm(source_delta)),float(np.linalg.norm(packed_delta))]
        halves.append(dict(half=int(half),lip_samples=int(a.sum()),verge_samples=int(b.sum()),
            source_rgb_contrast=contrasts[0],packed_yuv_contrast=contrasts[1]))
        directions.append(packed_delta)
    minimum=min(min(h['source_rgb_contrast'],h['packed_yuv_contrast']) for h in halves)
    cosine=float(directions[0]@directions[1]/max(np.linalg.norm(directions[0])*np.linalg.norm(directions[1]),1e-12))
    passed=minimum>=12. and cosine>=.5
    return dict(result,passed=passed,reason=None if passed else 'upper_lip_color_edge_unresolved',
        minimum_rgb_and_packed_contrast=minimum,halves=halves,color_direction_cosine=cosine,
        maximum_lip_plane_residual_m=residual,lip_source_pixels=xy[lip].tolist(),
        verge_source_pixels=xy[verge].tolist(),lip_model_pixels=model_xy[lip].tolist(),
        verge_model_pixels=model_xy[verge].tolist())


def inspect_curb_surface(data,reference,station,side,*,model_crop_only=False,_cropped_points=None,_cropped_colors=None):
    """Return a measured surface hypothesis, never training-ready evidence.

    Local axes retain road grade; the reference defines only a search box and
    correspondence station. The face location is fitted from actual metric
    depth returns. Its small internal fit residual is NOT an absolute-accuracy
    certificate and does not replace the independent collision toe survey.
    """
    if side not in ('left','right'):raise ValueError('invalid curb side')
    point=getattr(reference,'_'+side)(station)
    across=reference._right(station)-reference._left(station);across/=np.linalg.norm(across)
    tangent=reference._center(station,1);tangent/=np.linalg.norm(tangent)
    up=np.cross(tangent,across);up/=np.linalg.norm(up)
    if up[2]<0:up=-up
    sign=-1. if side=='left' else 1.
    shape=data['depth_m'].shape
    if model_crop_only:
        xy,world,tags,luma=_cropped_points if _cropped_points is not None else _cropped_surface_points(data)
    else:
        box=np.array([point+across*d+tangent*s+up*h for d in (-.9,.9) for s in (-.6,.6) for h in (-.01,.26)])
        pixels,optical=project_world_points(box,data['camera_from_world'],data['intrinsics'])
        if np.any(optical[:,2]<=0):return dict(surface_hypothesis=False,reason='search_box_not_in_front')
        lo=np.maximum(np.floor(pixels.min(0)-3).astype(int),0)
        hi=np.minimum(np.ceil(pixels.max(0)+3).astype(int),[shape[1]-1,shape[0]-1])
        if np.any(lo>=hi):return dict(surface_hypothesis=False,reason='search_box_outside_source')
        y,x=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1];xy=np.c_[x.ravel(),y.ravel()]
    if not model_crop_only:
        rays=np.c_[xy+.5,np.ones(len(xy))]@np.linalg.inv(data['intrinsics']).T
        optical=rays*data['depth_m'][xy[:,1],xy[:,0]][:,None]
        inverse=np.linalg.inv(data['camera_from_world'])
        world=optical@inverse[:3,:3].T+inverse[:3,3]
        tags=data['semantic_camera_ids'][xy[:,1],xy[:,0]]
        gray=split_nv12(*image_to_nv12(data['rgb'],'RGB'))[0]
        luma=gray[xy[:,1],xy[:,0]]
    delta=world-point;lateral=delta@across;along=delta@tangent;height=delta@up
    # A fixed 0.8 m patch undersamples far but visible curb faces after the
    # native warp. A bounded 2 m patch admits more actual face returns while
    # retaining the same plane residual and independent toe accuracy gates.
    along_limit=1. if model_crop_only else .4
    rgb_along_limit=.8 if model_crop_only else .3
    select=(abs(along)<along_limit)&(abs(lateral)<.85)&np.isin(tags,[2,25])&(height>.03)&(height<.125)
    count=int(select.sum())
    if count<12:return dict(surface_hypothesis=False,reason='insufficient_visible_face_samples',face_samples=count)
    matrix=np.c_[height[select],along[select],np.ones(count)]
    if np.linalg.matrix_rank(matrix)<3:
        return dict(surface_hypothesis=False,reason='curb_face_surface_rank_deficient',face_samples=count)
    fit=np.linalg.lstsq(matrix,lateral[select],rcond=None)[0]
    error=lateral[select]-matrix@fit
    face=(abs(along)<rgb_along_limit)&(abs(lateral-fit[2])<.06)&(height>.04)&(height<.12)&np.isin(tags,[2,25])
    pavement=(abs(along)<rgb_along_limit)&(sign*(lateral-fit[2])<-.08)&(sign*(lateral-fit[2])>-.25)&(abs(height)<.01)&np.isin(tags,[1,24,25])
    if min(face.sum(),pavement.sum())<3:
        return dict(surface_hypothesis=False,reason='face_or_adjacent_pavement_rgb_unavailable',face_samples=count,
            rgb_face_samples=int(face.sum()),rgb_pavement_samples=int(pavement.sum()))
    face_median=float(np.median(luma[face]));road_median=float(np.median(luma[pavement]))
    face_mad=float(np.median(abs(luma[face].astype(float)-face_median)))
    road_mad=float(np.median(abs(luma[pavement].astype(float)-road_median)))
    contrast=abs(face_median-road_median)
    method='observed_curb_face_to_pavement_luma'
    lip_proof=None
    if contrast<12 and model_crop_only:
        colors=_cropped_colors if _cropped_colors is not None else _cropped_pixel_colors(data,xy)
        lip_proof=_upper_lip_color_proof(xy=xy,tags=tags,lateral=lateral,along=along,height=height,
            sign=sign,toe_offset=fit[2],colors=colors)
        if lip_proof['passed']:
            contrast=lip_proof['minimum_rgb_and_packed_contrast'];method=lip_proof['method']
    toe=point+across*fit[2]
    pixels,_=project_world_points(np.array([toe,toe+up*.15]),data['camera_from_world'],data['intrinsics'])
    crop=np.c_[pixels-.5,np.ones(2)]@np.linalg.inv(data['model_to_source']).T;crop=crop[:,:2]/crop[:,2:3]
    return dict(surface_hypothesis=True,route_s=float(station),side=side,model_crop_only=model_crop_only,
        toe_xyz_hypothesis=toe.tolist(),toe_offset_from_reference_m=float(fit[2]),
        lateral_per_height=float(fit[0]),lateral_per_longitudinal=float(fit[1]),
        face_samples=count,face_source_pixels=xy[select].tolist(),
        maximum_face_longitudinal_search_extent_m=2.*along_limit,
        face_height_span_m=float(np.ptp(height[select])),face_along_span_m=float(np.ptp(along[select])),
        maximum_surface_fit_residual_m=float(abs(error).max()),p95_surface_fit_residual_m=float(np.quantile(abs(error),.95)),
        face_luma_median=face_median,pavement_luma_median=road_median,
        face_luma_mad=face_mad,pavement_luma_mad=road_mad,face_to_pavement_luma_contrast=abs(face_median-road_median),
        rgb_contrast=contrast,rgb_evidence_method=method,upper_lip_color_proof=lip_proof,
        original_rgb_contrast_threshold_passed=abs(face_median-road_median)>=12,
        toe_and_15cm_top_model_pixels=crop.tolist(),independent_collision_agreement_verified=False,training_ready=False)


def surveyed_reference(path,expected_sha256,*,raw_survey_path=None):
    """Rebuild whole-road QA curves only from measured physical curb-toe knots.

    Alternating held-out measured stations test both independent boundary fits.
    This creates QA geometry, not a visibility claim. Missing pole-occluded
    stations remain absent as source measurements; only <=4m gaps interpolate.
    """
    path=Path(path)
    if sha256_file(path)!=expected_sha256:raise ValueError('curb survey hash mismatch')
    document=json.loads(path.read_text(encoding='utf-8'))
    # Retained captures carry a byte-identical raw survey under their own
    # inventory. Its absolute original location is provenance, not a runtime
    # dependency; the original document's content hash remains mandatory.
    raw_path=Path(document['source_survey'] if raw_survey_path is None else raw_survey_path)
    if sha256_file(raw_path)!=document['source_survey_sha256']:
        raise ValueError('raw collision survey hash mismatch')
    raw=json.loads(raw_path.read_text(encoding='utf-8'))
    raw_rows={(r['route_s'],r['side']):r for r in raw['rows']}
    measured={side:[] for side in ('left','right')}
    uncertainty=[]
    for row in document['rows']:
        if not row.get('passed'):continue
        source=raw_rows[row['route_s'],row['side']]
        samples=source['samples']
        result=detect_curb_step([v['offset_m'] for v in samples],[v['xyz'][2] for v in samples])
        if not result['passed']:raise ValueError('passed collision toe does not replay from raw rays')
        toe=np.asarray(source['nominal_outer_edge_xyz'])+np.asarray(source['outward_unit'])*result['toe_offset_m']
        toe[2]=result['toe_height_m']
        if not np.allclose(toe,row['toe_xyz'],atol=1e-9,rtol=0):raise ValueError('collision toe coordinates changed')
        measured[row['side']].append((float(row['route_s']),toe))
        uncertainty.append(result['toe_uncertainty_m'])
    curves={};audit={}
    for side,values in measured.items():
        values.sort(key=lambda v:v[0]);s=np.array([v[0] for v in values]);xyz=np.array([v[1] for v in values])
        if len(s)<8 or np.any(np.diff(s)<=0) or np.max(np.diff(s))>4.+1e-6:
            raise ValueError('independent curb survey support has missing or oversized spatial gaps')
        errors=[]
        for fold in (0,1):
            fit=np.array([i for i in range(len(s)) if i%2==fold or i in (0,len(s)-1)])
            withheld=np.array([i for i in range(len(s)) if i not in fit])
            errors.extend(np.linalg.norm(CubicSpline(s[fit],xyz[fit],axis=0)(s[withheld])-xyz[withheld],axis=1).tolist())
        maximum=max(errors)
        if maximum+max(uncertainty)>.05:
            raise ValueError('held-out physical curb interpolation and toe uncertainty exceed five cm')
        curves[side]=(s,CubicSpline(s,xyz,axis=0))
        audit[side]=dict(measured_source_stations=s.tolist(),measured_toe_xyz=xyz.tolist(),
            maximum_withheld_error_m=maximum,maximum_measured_gap_m=float(np.diff(s).max()))
    lo=max(curves['left'][0][0],curves['right'][0][0]);hi=min(curves['left'][0][-1],curves['right'][0][-1])
    grid=np.union1d(curves['left'][0],curves['right'][0]);grid=grid[(grid>=lo)&(grid<=hi)]
    reference=LaneReference(curves['left'][1](grid),curves['right'][1](grid),grid)
    audit.update(survey=str(path.resolve()),survey_sha256=expected_sha256,raw_survey_path=str(raw_path.resolve()),raw_survey_sha256=document['source_survey_sha256'],
        maximum_toe_uncertainty_m=max(uncertainty),reference_role='independent_collision_QA_not_observed_geometry')
    audit['reference_arrays_sha256']=_digest(dict(left=array_sha256(reference.left_xyz),right=array_sha256(reference.right_xyz),arc=array_sha256(reference.s)))
    return reference,audit


def measure_curb_road_frame(*,payload,payload_sha256,stamp,modality_stamps,semantic_sensor_stamp,
                           depth_fixture,depth_fixture_sha256,survey,survey_sha256,current_s,
                           surveyed_geometry=None):
    """Return cropped rendered curb knots compatible with road assembly.

    Requires raw BGRA/depth replay, exact native packed RGB, qualified metric
    depth and independently replayed collision toes. Source-pixel pitch alone
    is not a curb-position measurement: the observed 3D face fit is checked
    directly against collision toe geometry within5cm, with <=5mm residual.
    """
    payload,depth_fixture=Path(payload),Path(depth_fixture)
    if sha256_file(payload)!=payload_sha256:raise ValueError('raw curb payload hash mismatch')
    if sha256_file(depth_fixture)!=depth_fixture_sha256:raise ValueError('curb depth fixture hash mismatch')
    for name in ('rgb','depth','packed'):
        value=modality_stamps.get(name)
        if (FrameStamp(**value) if isinstance(value,dict) else value)!=stamp:raise ValueError('unaligned curb modality stamp')
    semantic=FrameStamp(**semantic_sensor_stamp) if isinstance(semantic_sensor_stamp,dict) else semantic_sensor_stamp
    if semantic!=stamp:raise ValueError('unaligned curb semantic stamp')
    fixture=json.loads(depth_fixture.read_text(encoding='utf-8'))[stamp.camera]
    if fixture.get('qualified') is not True or fixture.get('depth_mode')!='forward':
        raise ValueError('qualified forward depth required for curb surface proof')
    reference,physical=surveyed_geometry or surveyed_reference(survey,survey_sha256)
    if physical['survey_sha256']!=survey_sha256 or sha256_file(survey)!=survey_sha256:
        raise ValueError('curb physical geometry binding mismatch')
    actual_reference=_digest(dict(left=array_sha256(reference.left_xyz),right=array_sha256(reference.right_xyz),arc=array_sha256(reference.s)))
    if (actual_reference!=physical['reference_arrays_sha256']
            or sha256_file(physical['raw_survey_path'])!=physical['raw_survey_sha256']):
        raise ValueError('cached curb geometry or raw collision evidence changed')
    with np.load(payload,allow_pickle=False) as saved:data={key:saved[key] for key in saved.files}
    if (not np.array_equal(data['rgb'],data['rgb_bgra'][...,[2,1,0]])
            or not np.array_equal(data['depth_m'],decode_carla_depth_bgra(data['depth_bgra']).astype(np.float32))
            or not np.array_equal(data['packed'],pack_model_rgb(data['rgb'],data['model_to_source']))):
        raise ValueError('curb raw modalities or packed frame do not replay')
    points=_cropped_surface_points(data);colors=_cropped_pixel_colors(data,points[0]);knots=[];rejections=Counter();measurements=[]
    start=max(reference.s[0],np.floor(current_s*2)/2+5.)
    for q in np.arange(start,min(reference.s[-1],current_s+60.),.5):
        for side in ('left','right'):
            result=inspect_curb_surface(data,reference,float(q),side,model_crop_only=True,
                _cropped_points=points,_cropped_colors=colors)
            if not result.get('surface_hypothesis'):
                rejections[result['reason']]+=1;continue
            reasons=[]
            if result['rgb_contrast']<12:reasons.append('curb_face_rgb_contrast')
            if result['maximum_surface_fit_residual_m']>.005:reasons.append('curb_face_depth_surface_disagreement')
            if result['face_height_span_m']<.05 or result['face_along_span_m']<.25:reasons.append('curb_face_surface_extent_unresolved')
            error_budget=(abs(result['toe_offset_from_reference_m'])+physical[side]['maximum_withheld_error_m']
                +physical['maximum_toe_uncertainty_m']+result['maximum_surface_fit_residual_m'])
            result['conservative_toe_error_budget_m']=error_budget
            if error_budget>.05:reasons.append('curb_toe_collision_error_budget_above_five_cm')
            result['independent_collision_agreement_verified']=not reasons
            result['collision_toe_xyz']=getattr(reference,'_'+side)(q).tolist()
            result['reasons']=reasons;measurements.append(result)
            if reasons:
                rejections.update(reasons);continue
            knots.append(dict(side=side,source_index=int(round(q*2)),source_route_s=float(q),
                observed_xyz=result['toe_xyz_hypothesis'],curb_surface_proof=result))
    binding=_digest(dict(survey_sha256=survey_sha256,left=array_sha256(reference.left_xyz),
        right=array_sha256(reference.right_xyz),source_arc=array_sha256(reference.s)))
    audit=dict(version=EVIDENCE_VERSION,stamp=asdict(stamp),payload=str(payload.resolve()),payload_sha256=payload_sha256,
        depth_fixture=str(depth_fixture.resolve()),depth_fixture_sha256=depth_fixture_sha256,reference_sha256=binding,
        independent_physical_survey=physical,source_spacing_m=.5,source_identity_origin_m=0.,knots=knots,
        measurements=measurements,retained_by_side=dict(Counter(k['side'] for k in knots)),rejections=dict(rejections),
        packed_sha256=array_sha256(data['packed']),scope='observed_curb_knots_only_not_behavior_admission')
    return FrameRoadEvidence(stamp,binding,tuple(knots),audit,_digest(audit))


def diagnose_saved_probe(source,plan,output):
    source,plan=Path(source),Path(plan)
    report=json.loads((source/'probe-report.json').read_text(encoding='utf-8'))
    task=json.loads(plan.read_text(encoding='utf-8'))['tasks'][0]
    geometry=task['route']['unmarked_road']
    reference=LaneReference(geometry['road_left_edge_xyz'],geometry['road_right_edge_xyz'],task['route']['arc_m'])
    fixture=source/'depth-fixture.json'
    if sha256_file(fixture)!=report['depth_fixture_sha256']:raise ValueError('depth fixture hash mismatch')
    fixtures=json.loads(fixture.read_text(encoding='utf-8'));rows=[];bindings={}
    # First and last retained frames check persistence without pretending this
    # stationary recording is a moving multi-view qualification.
    for frame in (report['frame_records'][0],report['frame_records'][-1]):
        for camera,meta in frame['cameras'].items():
            payload=source/meta['payload']
            if sha256_file(payload)!=meta['payload_sha256']:raise ValueError('raw payload hash mismatch')
            bindings[payload.name]=meta['payload_sha256']
            if fixtures[camera].get('qualified') is not True or fixtures[camera].get('depth_mode')!='forward':
                raise ValueError('diagnostic requires qualified forward metric depth')
            with np.load(payload,allow_pickle=False) as saved:data={key:saved[key] for key in saved.files}
            if not np.array_equal(data['packed'],pack_model_rgb(data['rgb'],data['model_to_source'])):
                raise ValueError('packed RGB does not replay')
            for q in (86.,87.,88.,90.,92.,95.,98.,100.,102.,105.,110.):
                for side in ('left','right'):
                    for crop in (False,True):
                        value=dict(frame_id=frame['frame_id'],timestamp_s=frame['timestamp_s'],camera=camera,
                            route_s=q,side=side,model_crop_only=crop)
                        value.update(inspect_curb_surface(data,reference,q,side,model_crop_only=crop))
                        rows.append(value)
    document=dict(scope='offline_native_curb_material_seam_diagnosis_not_admission',
        source_probe=str(source),source_probe_sha256=sha256_file(source/'probe-report.json'),
        plan=str(plan),plan_sha256=sha256_file(plan),depth_fixture_sha256=sha256_file(fixture),
        payload_sha256=bindings,rows=rows,training_ready=False,
        finding='Road/Ground material seam is flat gutter; visible raised curb toe lies farther outward',
        required_next='independent collision toe survey followed by same-frame RGB/depth surface and actual crop replay',
        unresolved_false_positive_tests=['vertical roadside object versus continuous curb','terrain/berm shape','multiple views and shadowed curb faces'],
        no_precision_claim='surface fit residual is internal consistency, not independent absolute toe accuracy')
    write_json(output,document)
    return document


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=diagnose_saved_probe(args.source,args.plan,args.output)
    print(json.dumps(dict(output=str(args.output),rows=len(result['rows']),training_ready=False),indent=2))
