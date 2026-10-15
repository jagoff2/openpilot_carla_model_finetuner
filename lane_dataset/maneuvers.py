"""Geometric maneuver proposals and spatial route intent for capture.

Proposals locate events for recording. Rendered phase review remains mandatory
before admission; a curvature calculation is not a completed visual review.
"""
from __future__ import annotations

import math
import numpy as np


def event_heading_change(route, start, end):
    """Measure signed and absolute heading change in the selected event only."""
    arc = np.asarray(route['arc_m'], float)
    center = np.asarray(route['center_xyz'], float)
    delta = np.diff(center[:, :2], axis=0)
    if (center.shape != (len(arc), 3) or len(arc) < 4 or
            not np.isfinite(center).all() or np.any(np.linalg.norm(delta, axis=1) <= 0) or
            not arc[0] <= start < end <= arc[-1]):
        raise ValueError('invalid maneuver heading geometry')
    stations = (arc[:-1]+arc[1:])*.5
    headings = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    query = np.r_[start, stations[(stations > start)&(stations < end)], end]
    values = np.interp(query, stations, headings)
    return float(values[-1]-values[0]), float(np.abs(np.diff(values)).sum())


def maneuver_event_errors(route, family, phases):
    """A curve elsewhere on the route cannot qualify a straight recorded event."""
    if family == 'straight_recovery':
        return []  # Its independent measured offset/recovery test is mandatory.
    if not phases or 'entry' not in phases or 'exit' not in phases:
        return ['maneuver_phase_ranges_missing']
    start, end = phases['entry'][0], phases['exit'][1]
    signed, total = event_heading_change(route, start, end)
    if family == 'intersection_turn':
        if not any(max(start, lo) < min(end, hi) for lo, hi in route.get('junction_ranges_m', [])):
            return ['intersection_event_outside_junction']
        if abs(signed) < math.radians(20.):
            return ['intersection_event_has_no_resolved_turn']
    elif total < math.radians(5.):
        return ['curved_event_has_no_resolved_heading_change']
    return []


def propose_maneuver(route, family):
    arc=np.asarray(route["arc_m"],float)
    curvature=np.asarray(route["curvature_1pm"],float)
    if (arc.ndim!=1 or len(arc)<4 or curvature.shape!=arc.shape or
        not np.isfinite(arc).all() or not np.isfinite(curvature).all() or
        np.any(np.diff(arc)<=0)):
        raise ValueError("maneuver proposal requires finite ordered route geometry")
    result=dict(annotation_status="proposal_requires_rendered_review")
    if family=="straight_recovery":
        result["annotation_blockers"]=["physical_recovery_event_requires_retained_prefix_and_measured_onset"]
        return result
    active=np.flatnonzero(np.abs(curvature)>=max(.004,.15*float(np.max(np.abs(curvature)))))
    if len(active)<4:
        result["annotation_blockers"]=["no_resolved_curved_maneuver"]
        return result
    start,end=float(arc[active[0]]),float(arc[active[-1]])
    if family=="intersection_turn":
        ranges=route.get("junction_ranges_m",[])
        if not ranges:
            result["annotation_blockers"]=["intersection_road_scope_missing"]
            return result
        # A route can cross a junction straight, then curve far beyond it.
        # Classifying from full-route curvature made numerical noise inside
        # that junction count as a left/right turn.
        turns=[(abs(event_heading_change(route,*pair)[0]),pair) for pair in ranges]
        turns=[value for value in turns if value[0]>=math.radians(20.)]
        if not turns:
            result['annotation_blockers']=['intersection_event_has_no_resolved_turn']
            return result
        _,(start,end)=max(turns,key=lambda value:value[0])
        result.update(desire_route_range_m=[max(float(arc[0]),start-10.),end],
                      intent_required_route_range_m=[max(float(arc[0]),start-5.),min(float(arc[-1]),start+1.)])
    inside=(arc>=start)&(arc<=end)
    a=arc[inside];k=np.abs(curvature[inside])
    mass=np.r_[0.,np.cumsum((k[:-1]+k[1:])*.5*np.diff(a))]
    if len(a)<4 or mass[-1]<=0:
        result["annotation_blockers"]=["turn_heading_change_unresolved"]
        return result
    q1,q3=np.interp([.25*mass[-1],.75*mass[-1]],mass,a)
    edges=[max(float(arc[0]),start-10.),start,float(q1),float(q3),end,min(float(arc[-1]),end+10.)]
    minimum_phase_width=2.*float(route.get("nominal_speed_mps",2.5))
    if any(right-left<minimum_phase_width for left,right in zip(edges,edges[1:])):
        result["annotation_blockers"]=["maneuver_phase_too_short_for_four_2hz_decisions"]
        return result
    result["maneuver_phase_ranges"]={phase:[left,right] for phase,left,right in
        zip(("approach","entry","apex","exit","recovery"),edges,edges[1:])}
    result["phase_basis"]="road curvature heading-change quartiles with measured junction interval where present"
    result["annotation_blockers"]=[]
    return result


def route_desire(task, index, route_s):
    neutral=[1.,0.,0.,0.,0.,0.,0.,0.]
    if not task or task.get("family")!="intersection_turn":
        return neutral
    bounds=task.get("desire_route_range_m")
    if bounds is None:
        active=task.get("desire_onset_index",80)<=index<=task.get("desire_end_index",180)
    else:
        if (len(bounds)!=2 or not all(math.isfinite(float(value)) for value in bounds)
            or bounds[1]<=bounds[0] or not math.isfinite(route_s)):
            raise ValueError("route intent requires finite increasing spatial bounds")
        active=bounds[0]<=route_s<=bounds[1]
    if not active:
        return neutral
    direction=task.get("direction")
    if direction not in ("left","right"):
        raise ValueError("intersection intent requires a left or right branch")
    value=[0.]*8
    value[1 if direction=="left" else 2]=1.
    return value
