"""Independent physical curb-toe survey from collision rays, never RGB depth.

The pure detector fits road grade, a raised curb face and its upper plateau.
The toe is the face/road-plane intersection. A semantic material seam without
height relief cannot qualify. Surveyed geometry still needs independent rendered
evidence and does not itself confer label visibility or training admission.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import time

import numpy as np

from lane_dataset.simulator import world_owner

VERSION='independent_collision_road_grade_beveled_curb_toe_v2'


def detect_curb_step(offsets_m,heights_m,*,minimum_height_m=.08,maximum_height_m=.25,
                     maximum_residual_m=.005,maximum_toe_uncertainty_m=.02):
    """Fit an increasing outward cross-section; retain measured road grade.

    Offsets run from road interior to exterior; all samples must be actual
    collision hit heights. Requires >=0.15m road and top support, a resolved
    rising face, <=5mm surface fit residual and <=2cm toe bracket uncertainty.
    All thresholds are geometric and fixed before any rendered RGB comparison.
    """
    x,z=np.asarray(offsets_m,float),np.asarray(heights_m,float)
    if (x.ndim!=1 or z.shape!=x.shape or len(x)<12 or not np.isfinite([x,z]).all()
            or np.any(np.diff(x)<=0) or x[-1]-x[0]<.5):
        raise ValueError('curb survey requires finite increasing measured cross-section samples')
    if not .05<=minimum_height_m<maximum_height_m<=.35 or not 0<maximum_residual_m<=.005:
        raise ValueError('unsupported curb geometry limits')
    road_mask=x<=x[0]+.20
    grade=np.polyfit(x[road_mask],z[road_mask],1)
    ground=np.polyval(grade,x);h=z-ground
    road_error=float(np.max(abs(h[road_mask])))
    if road_error>maximum_residual_m:
        return dict(passed=False,reason='road_grade_not_resolved',road_residual_m=road_error)
    top_mask=x>=x[-1]-.20
    rise=float(np.median(h[top_mask]));top_error=float(np.max(abs(h[top_mask]-rise)))
    if not minimum_height_m<=rise<=maximum_height_m:
        return dict(passed=False,reason='no_bounded_raised_curb',height_m=rise)
    if top_error>maximum_residual_m:
        return dict(passed=False,reason='curb_top_not_parallel_to_road_grade',top_residual_m=top_error)
    # Native curb meshes have a steep face followed by a shallow top bevel.
    # Estimate the lower face independently, excluding that differently sloped
    # surface rather than weakening the residual threshold for the whole curb.
    face=(h>=.015)&(h<=rise*.80)
    if face.sum()<3:
        return dict(passed=False,reason='curb_face_unresolved',face_samples=int(face.sum()))
    # Isolated rocks, depressions or multiple steps cannot silently select a
    # convenient sub-face; every intermediate-height sample participates.
    slope,intercept=np.polyfit(x[face],h[face],1)
    if not 1.<=slope<=20.:
        return dict(passed=False,reason='curb_rise_slope_outside_resolved_face',slope=float(slope))
    toe=float(-intercept/slope);top=float((rise-intercept)/slope)
    if not x[0]+.15<=toe<top<=x[-1]-.15:
        return dict(passed=False,reason='curb_toe_or_top_not_bracketed')
    predicted=np.clip(slope*(x-toe),0.,rise)
    model='road_face_top'
    bevel_data={}
    if np.max(abs(h-predicted))>maximum_residual_m:
        bevel=(h>rise*.88)&(h<rise-.001)
        if bevel.sum()>=2 and np.ptp(h[bevel])>=.004:
            bevel_slope,bevel_intercept=np.polyfit(x[bevel],h[bevel],1)
            if .25<=bevel_slope<slope:
                join=float((bevel_intercept-intercept)/(slope-bevel_slope))
                bevel_top=float((rise-bevel_intercept)/bevel_slope)
                join_height=float(slope*join+intercept)
                if (toe<join<bevel_top<=x[-1]-.15 and bevel_top-join<=.05
                        and .70*rise<=join_height<=.98*rise):
                    predicted=np.where(x<=join,np.maximum(slope*(x-toe),0.),
                        np.minimum(bevel_slope*x+bevel_intercept,rise))
                    top=bevel_top;model='road_face_top_bevel_plateau'
                    bevel_data=dict(bevel_slope=float(bevel_slope),bevel_start_offset_m=join,
                        bevel_top_offset_m=bevel_top,bevel_samples=int(bevel.sum()))
    residual=abs(h-predicted)
    maximum=float(residual.max())
    if maximum>maximum_residual_m:
        return dict(passed=False,reason='curb_step_model_disagreement',maximum_residual_m=maximum)
    before=np.flatnonzero(x<=toe);after=np.flatnonzero(x>=toe)
    bracket=float(x[after[0]]-x[before[-1]])
    uncertainty=bracket+maximum/slope
    if uncertainty>maximum_toe_uncertainty_m:
        return dict(passed=False,reason='curb_toe_uncertainty',toe_uncertainty_m=float(uncertainty))
    return dict(passed=True,toe_offset_m=toe,toe_height_m=float(np.polyval(grade,toe)),
        top_offset_m=top,curb_height_m=rise,face_slope=float(slope),road_grade=float(grade[0]),
        road_intercept_m=float(grade[1]),maximum_residual_m=maximum,toe_bracket_m=bracket,
        toe_uncertainty_m=float(uncertainty),face_samples=int(face.sum()),version=VERSION,
        surface_model=model,**bevel_data)


@dataclass
class _SurveyOwnership:
    port:int
    root:Path
    pid:int
    active:bool=True


@contextmanager
def curb_survey_owner(port,root):
    """Acquire the exact live collector lock before connecting/loading a world.

    Caller performs reservation checks before this context and immediately
    inside it, just as capture does. No connection, map load or tick occurs here.
    Do not additionally nest world_owner; this context owns that same lock.
    """
    with world_owner(port,root):
        lease=_SurveyOwnership(int(port),Path(root).resolve(),os.getpid())
        try:yield lease
        finally:lease.active=False


def survey_owned_world(world,reference,*,ownership,stations=None,maximum_seconds=120.,
                       maximum_rays=8000,progress=None,carla_module=None,source_bindings=None):
    """Raycast both curb toes in an already reserved, owned, empty native world.

    ``reference`` is the whole-road outer-shoulder QA reference. Coarse search
    spans -0.9..+0.4m outward from it. Ten coarse rays, <=1cm bisection,
    then a short 5mm profile resolve the face. Default stations are 2m apart.
    Never loads, spawns, ticks, controls, or replaces a world. Budget exhaustion
    returns explicitly incomplete evidence; it never fills unsurveyed stations.
    """
    if not isinstance(ownership,_SurveyOwnership) or not ownership.active or ownership.pid!=os.getpid():
        raise ValueError('active curb_survey_owner lease required before any world call')
    if not 0<maximum_seconds<=600 or not 1<=maximum_rays<=100000:
        raise ValueError('bounded curb survey budget required')
    if carla_module is None:
        import carla as carla_module
    if any(actor.type_id.startswith(('vehicle.','walker.','sensor.')) for actor in world.get_actors()):
        raise ValueError('curb survey requires an empty owned world without dynamic occluders')
    q=np.asarray(stations if stations is not None else np.unique(np.r_[np.arange(reference.s[0],reference.s[-1],2.),reference.s[-1]]),float)
    if q.ndim!=1 or not len(q) or not np.isfinite(q).all() or np.any(np.diff(q)<=0) or q[0]<reference.s[0] or q[-1]>reference.s[-1]:
        raise ValueError('invalid curb survey stations')
    allowed={carla_module.CityObjectLabel.Roads,carla_module.CityObjectLabel.RoadLines,
        carla_module.CityObjectLabel.Ground,carla_module.CityObjectLabel.Sidewalks,carla_module.CityObjectLabel.Terrain}
    begin=time.monotonic();ray_count=0;rows=[];budget_exhausted=False
    class BudgetExceeded(Exception):pass
    for station in q:
        if budget_exhausted:break
        for side,sign in (('left',-1.),('right',1.)):
            point=getattr(reference,'_'+side)(float(station))
            outward=reference._right(station)-reference._left(station);outward[2]=0.
            outward=sign*outward/np.linalg.norm(outward)
            samples={}
            def hit(offset):
                nonlocal ray_count
                if ray_count>=maximum_rays or time.monotonic()-begin>=maximum_seconds:raise BudgetExceeded()
                nominal=point+float(offset)*outward
                a=carla_module.Location(x=float(nominal[0]),y=float(nominal[1]),z=float(point[2]+1.))
                b=carla_module.Location(x=float(nominal[0]),y=float(nominal[1]),z=float(point[2]-1.))
                hits=world.cast_ray(a,b);ray_count+=1
                # Preserve all hits and reject named occluders. Unlabelled NONE
                # geometry cannot be discarded without rendered evidence here.
                values=[h for h in hits if np.isfinite([h.location.x,h.location.y,h.location.z]).all()
                    and abs(h.location.z-a.z)>1e-4]
                if not values:raise ValueError('collision_ray_missed')
                first=max(values,key=lambda h:h.location.z)
                if first.label not in allowed:raise ValueError('collision_surface_not_road_or_curb:'+str(first.label))
                sample=dict(offset_m=float(offset),xyz=[float(first.location.x),float(first.location.y),float(first.location.z)],
                    label=str(first.label),all_hits=[dict(xyz=[float(h.location.x),float(h.location.y),float(h.location.z)],label=str(h.label)) for h in values])
                samples[round(float(offset),7)]=sample
                return sample['xyz'][2]
            try:
                coarse=np.array([-.9,-.8,-.7,-.5,-.3,-.1,.1,.2,.3,.4])
                heights=np.array([hit(x) for x in coarse])
                ground=np.polyfit(coarse[:3],heights[:3],1)
                rise=heights-np.polyval(ground,coarse)
                step_height=float(np.median(rise[-3:]))
                candidates=np.flatnonzero(rise>step_height*.5)
                if .08<=step_height<=.25 and len(candidates) and candidates[0]>0:
                    upper=int(candidates[0]);lo,hi=coarse[upper-1],coarse[upper]
                    while hi-lo>.006:
                        middle=round(float((lo+hi)*.5),7)
                        height=hit(middle)-np.polyval(ground,middle)
                        if height>step_height*.5:hi=middle
                        else:lo=middle
                    guess=(lo+hi)*.5
                    for offset in np.round(np.arange(max(coarse[0],guess-.035),min(coarse[-1],guess+.035)+.0001,.005),7):
                        if float(offset) not in samples:hit(offset)
                offsets=np.array(sorted(samples));z=np.array([samples[x]['xyz'][2] for x in offsets])
                result=detect_curb_step(offsets,z)
                if result['passed']:
                    toe=point+outward*result['toe_offset_m'];toe[2]=result['toe_height_m']
                    result['toe_xyz']=toe.tolist()
            except BudgetExceeded:
                budget_exhausted=True;result=dict(passed=False,reason='survey_budget_exhausted')
            except ValueError as error:
                result=dict(passed=False,reason=str(error))
            rows.append(dict(route_s=float(station),side=side,nominal_outer_edge_xyz=point.tolist(),
                outward_unit=outward.tolist(),samples=[samples[k] for k in sorted(samples)],**result))
            if progress:progress(len(rows),2*len(q),ray_count)
            if budget_exhausted:break
    return dict(version=VERSION,scope='independent_collision_curb_toe_geometry_not_rendered_admission',
        requested_stations=q.tolist(),rows=rows,ray_count=ray_count,elapsed_seconds=time.monotonic()-begin,
        complete=len(rows)==2*len(q) and not budget_exhausted,
        passed=len(rows)==2*len(q) and all(row['passed'] for row in rows),
        budget_exhausted=budget_exhausted,training_ready=False,port=ownership.port,
        source_bindings=dict(source_bindings or {}),station_sampling='explicit measured stations; no unsurveyed edge interpolation claim')
