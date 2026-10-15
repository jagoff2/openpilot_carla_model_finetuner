"""Measure painted centers and visible road edges from aligned observations.

Simulator geometry defines QA cross-sections, never student inputs. Each result
retains the sampled semantic bits and depth residuals, including failed samples.
Measurements do not constitute a visual maneuver review.
"""
from __future__ import annotations

import numpy as np

from .visibility import project_world_points

MAXIMUM_CROSS_SECTION_METERS_PER_PIXEL=.01
MAXIMUM_LOOKAHEAD_M=60.


def check_spatial_resolution(result, meters_per_pixel):
    """Reject unresolved camera measurements independently of their offset."""
    if result['measured'] and (meters_per_pixel is None or not np.isfinite(meters_per_pixel)
                              or not 0<meters_per_pixel<=MAXIMUM_CROSS_SECTION_METERS_PER_PIXEL):
        return dict(result,measured=False,reason='cross_section_pixel_resolution_insufficient')
    return result


def bind_marking_widths(reference, route):
    """Bind declared paint widths for resolving complete stripes, not fitting offsets."""
    markings=route.get('boundary_markings')
    if markings is None:return
    widths=np.asarray([[row[side]['width_m'] for side in ('left','right')] for row in markings],float)
    if widths.shape!=(len(reference.s),2) or not np.isfinite(widths).all() or np.any(widths<0):
        raise ValueError('marking widths must align with the reference')
    reference.boundary_marking_widths=widths


def stripe_center(offsets, marking, depth_residual, *, maximum_depth_error=.15, expected_width_m=None):
    offsets=np.asarray(offsets,float)
    marking=np.asarray(marking,bool)
    residual=np.asarray(depth_residual,float)
    if (offsets.ndim!=1 or len(offsets)<3 or marking.shape!=offsets.shape or
        residual.shape!=offsets.shape or not np.isfinite(offsets).all() or
        np.any(np.diff(offsets)<=0)):
        raise ValueError("stripe samples must be aligned on an increasing physical cross-section")
    # Locate paint independently of its agreement with the expected boundary.
    indices=np.flatnonzero(marking)
    runs=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
    runs=[run for run in runs if len(run)>=2 and run[0]>0 and run[-1]<len(offsets)-1]
    if not runs:
        return dict(measured=False,reason="no_complete_paint_stripe")
    centers=[float((offsets[run[0]]+offsets[run[-1]])*.5) for run in runs]
    selected=int(np.argmin(np.abs(centers)))
    run=runs[selected]
    if not np.isfinite(residual[run]).all() or np.max(np.abs(residual[run]))>maximum_depth_error:
        return dict(measured=False,reason="paint_depth_disagreement")
    width=float(offsets[run[-1]]-offsets[run[0]])
    if expected_width_m is not None and (not np.isfinite(expected_width_m) or expected_width_m<=0):
        raise ValueError('declared paint width must be finite and positive')
    minimum_width=.02 if expected_width_m is None else max(.02,.8*expected_width_m)
    maximum_width=.3 if expected_width_m is None else min(.3,1.25*expected_width_m)
    # A worn fragment at one edge does not locate the center of the full stripe.
    # This resolution test depends on width alone, never agreement with the map.
    # Intersecting/merged paint can be much wider than the declared stripe.
    # Its midpoint does not locate that stripe. Apply the width test equally
    # to centered and displaced samples, before testing map agreement.
    if not minimum_width-1e-9<=width<=maximum_width+1e-9:
        return dict(measured=False,reason="paint_width_unresolved",width_m=width)
    return dict(measured=True,offset_m=centers[selected],width_m=width,
                maximum_depth_residual_m=float(np.max(np.abs(residual[run]))))


def edge_center(offsets, road, exterior, luma, depth_residual, side):
    offsets=np.asarray(offsets,float);road=np.asarray(road,bool);exterior=np.asarray(exterior,bool)
    luma=np.asarray(luma,float);residual=np.asarray(depth_residual,float)
    if (any(v.shape!=offsets.shape for v in (road,exterior,luma,residual)) or
        np.any(road&exterior) or side not in ('left','right')):
        raise ValueError('invalid road-edge cross-section')
    # Positive offsets point right. Road must be on the lane's interior side.
    transitions=np.flatnonzero((exterior[:-1]&road[1:]) if side=='left' else (road[:-1]&exterior[1:]))
    if not len(transitions):return dict(measured=False,reason='no_road_exterior_transition')
    centers=(offsets[transitions]+offsets[transitions+1])*.5
    index=int(transitions[np.argmin(np.abs(centers))])
    lo=max(0,index-19);hi=min(len(offsets),index+21)
    road_region=road[lo:hi];other_region=exterior[lo:hi]
    if min(road_region.sum(),other_region.sum())<3:
        return dict(measured=False,reason='unresolved_road_edge')
    contrast=abs(float(np.median(luma[lo:hi][road_region]))-float(np.median(luma[lo:hi][other_region])))
    if not np.isfinite(contrast) or contrast<12:
        return dict(measured=False,reason='road_edge_rgb_contrast')
    pair=residual[index:index+2]
    if not np.isfinite(pair).all() or np.max(np.abs(pair))>.15:
        return dict(measured=False,reason='road_edge_depth_disagreement')
    return dict(measured=True,offset_m=float((offsets[index]+offsets[index+1])*.5),
                detector='road_edge',contrast=contrast,maximum_depth_residual_m=float(np.max(np.abs(pair))))


def measure_camera(*, reference, current_s, tags, depth, camera_from_world,
                   intrinsics, depth_mode, road_line_tag, luma=None, road_tag=None, exterior_tags=(),pixel_center_offset=0.):
    if pixel_center_offset not in (0.,.5):raise ValueError('unsupported sensor pixel center convention')
    offsets=np.linspace(-.4,.4,161)
    # Paint beyond a junction can remain resolved in the narrow camera even
    # when the nearest 25 m has no markings. Distance alone is not a precision
    # test: every sample still needs <=1 cm/pixel, depth, and stripe-width checks.
    queries=np.arange(max(reference.s[0],current_s+5.),min(reference.s[-1],current_s+MAXIMUM_LOOKAHEAD_M),.5)
    rows=[]
    for side in ("left","right"):
        boundary=getattr(reference,"_"+side)(queries)
        normal=reference._right(queries)-reference._left(queries)
        normal/=np.linalg.norm(normal,axis=1)[:,None]
        for progress,point,direction in zip(queries,boundary,normal):
            world=point[None,:]+offsets[:,None]*direction[None,:]
            pixels,optical=project_world_points(world,camera_from_world,intrinsics)
            pixel_span=float(np.linalg.norm(pixels[81]-pixels[79]))
            meters_per_pixel=.01/pixel_span if np.isfinite(pixel_span) and pixel_span>0 else None
            finite=np.isfinite(pixels).all(axis=1)&(optical[:,2]>0)
            xy=np.rint(np.where(finite[:,None],pixels-pixel_center_offset,-100.)).astype(int)
            inside=finite&(xy[:,0]>=0)&(xy[:,0]<tags.shape[1])&(xy[:,1]>=0)&(xy[:,1]<tags.shape[0])
            marking=np.zeros(len(offsets),bool)
            residual=np.full(len(offsets),np.nan)
            expected=optical[:,2] if depth_mode=="forward" else np.linalg.norm(optical,axis=1)
            if pixel_center_offset:
                tangent=reference._center(progress,1)
                plane_normal=np.cross(tangent,direction)
                plane_normal/=np.linalg.norm(plane_normal)
                plane_normal=np.asarray(camera_from_world)[:3,:3]@plane_normal
                plane_distance=float(plane_normal@optical[80])
                rays=np.column_stack((xy+pixel_center_offset,np.ones(len(xy))))@np.linalg.inv(intrinsics).T
                denominator=rays@plane_normal
                forward=np.divide(plane_distance,denominator,out=np.full(len(xy),np.inf),where=np.abs(denominator)>1e-8)
                expected=forward if depth_mode=='forward' else forward*np.linalg.norm(rays,axis=1)
            x,y=xy[inside,0],xy[inside,1]
            marking[inside]=tags[y,x]==road_line_tag
            residual[inside]=depth[y,x]-expected[inside]
            expected_width=None
            if hasattr(reference,'boundary_marking_widths'):
                index=int(np.argmin(np.abs(reference.s-progress)))
                declared=float(reference.boundary_marking_widths[index,0 if side=='left' else 1])
                if declared>0:expected_width=declared
            result=stripe_center(offsets,marking,residual,expected_width_m=expected_width)
            edge_proof={}
            if luma is not None and road_tag is not None:
                road=np.zeros(len(offsets),bool);exterior=road.copy();gray=np.zeros(len(offsets),np.uint8)
                road[inside]=np.isin(tags[y,x],[road_tag,road_line_tag])
                exterior[inside]=np.isin(tags[y,x],exterior_tags)
                gray[inside]=luma[y,x]
                edge_proof=dict(road_bits=np.packbits(road).tolist(),exterior_bits=np.packbits(exterior).tolist(),luma=gray.tolist())
                if not result['measured']:
                    result=edge_center(offsets,road,exterior,gray,residual,side)
            result=check_spatial_resolution(result,meters_per_pixel)
            if not result['measured']:
                # Negative measurements cannot support acceptance. Retain their
                # location/reason, but do not duplicate hundreds of unusable
                # depth/luma arrays per frame. Every positive measurement keeps
                # the complete replayable proof, including displaced stripes.
                rows.append(dict(side=side,route_s=float(progress),proof_omitted='unresolved_no_evidence',**result))
                continue
            rows.append(dict(side=side,route_s=float(progress),point_world=point.tolist(),
                expected_marking_width_m=expected_width,
                cross_section_meters_per_pixel=meters_per_pixel,
                marking_bits=np.packbits(marking).tolist(),
                depth_residual_m=[float(value) if np.isfinite(value) else None for value in residual],
                **edge_proof,**result))
    return dict(offsets_m=offsets.tolist(),samples=rows,pixel_center_offset=pixel_center_offset)


def summarize_frames(frames):
    from collections import deque
    errors=[];maximum=0.;counts=[];support_counts=[];history=deque();last_frame=None
    for frame in frames:
        sides={side:[] for side in ("left","right")}
        spatial={side:set() for side in ('left','right')}
        if last_frame is not None and frame['frame_id']<=last_frame:
            errors.append('unordered_rendered_boundary_frames')
        last_frame=frame['frame_id']
        for camera in frame["cameras"].values():
            offsets=camera["offsets_m"]
            for sample in camera["samples"]:
                if sample.get('proof_omitted')=='unresolved_no_evidence':
                    if sample.get('measured') is not False or not sample.get('reason'):
                        errors.append(f"invalid_omitted_measurement:{frame['frame_id']}")
                    continue
                marking=np.unpackbits(np.asarray(sample["marking_bits"],np.uint8))[:len(offsets)].astype(bool)
                residual=np.asarray([np.nan if v is None else v for v in sample["depth_residual_m"]])
                recomputed=stripe_center(offsets,marking,residual,expected_width_m=sample.get('expected_marking_width_m'))
                if not recomputed['measured'] and 'road_bits' in sample:
                    road=np.unpackbits(np.asarray(sample['road_bits'],np.uint8))[:len(offsets)].astype(bool)
                    other=np.unpackbits(np.asarray(sample['exterior_bits'],np.uint8))[:len(offsets)].astype(bool)
                    recomputed=edge_center(offsets,road,other,sample['luma'],residual,sample['side'])
                if 'cross_section_meters_per_pixel' in sample:
                    recomputed=check_spatial_resolution(recomputed,sample['cross_section_meters_per_pixel'])
                if recomputed!= {key:sample[key] for key in recomputed}:
                    errors.append(f"measurement_replay_mismatch:{frame['frame_id']}")
                if recomputed["measured"]:
                    sides[sample["side"]].append(abs(recomputed["offset_m"]))
                    spatial[sample['side']].add(int(round(sample['route_s']*2)))
        counts.append({key:len(value) for key,value in sides.items()})
        # Physical geometry is static. Repeated precise measurements can remain
        # valid when dashed paint temporarily leaves a camera crop. Keep the
        # same causal 4.8-second window as label evidence; never borrow future
        # images, fill raw visibility, or relax the 5 cm alignment threshold.
        history.append((frame['frame_id'],spatial))
        while history and history[0][0]<frame['frame_id']-96:history.popleft()
        supported={side:len(set().union(*(v[side] for _,v in history))) for side in sides}
        support_counts.append(supported)
        if any(value<3 for value in supported.values()):
            errors.append(f"insufficient_paired_rendered_boundaries:{frame['frame_id']}")
        maximum=max(maximum,max((value for samples in sides.values() for value in samples),default=0.))
    if not frames:
        errors.append("no_rendered_boundary_frames")
    if maximum>.05+1e-9:
        errors.append("rendered_boundary_disagreement_above_five_cm")
    return dict(passed=not errors,errors=sorted(set(errors)),frame_count=len(frames),
        maximum_disagreement_m=maximum,measurements=dict(samples_per_frame=counts,causal_supported_locations_per_frame=support_counts),
        method="aligned semantic paint centers or RGB-contrasted road/exterior transitions with metric depth checks; 5 mm cross-sections",
        maximum_evidence_age_s=4.8,
        scope="three distinct resolved locations per side in current/past 4.8 seconds; 5 cm maximum disagreement; raw current visibility remains separate")
