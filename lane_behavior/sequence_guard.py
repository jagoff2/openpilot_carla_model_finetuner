"""Rebudget physical native routes and check actual teacher envelopes pre-capture.

These checks are simulator planning constraints. They do not establish recorded
timing, camera evidence, label admission, or driving qualification.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np

from .short_native import sequence_budget


VERSION = 'surveyed_short_native_sequence_budget_v1'
ON_ROAD_VERSION = 'surveyed_on_road_parked_sequence_budget_v2'
FRAME = 'surveyed_physical_whole_road_arc_m'


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def rebudget(source_task, *, length_m, start_s, obstacle_s):
    """Preserve source stations by explicit mapping, then recompute metric margins."""
    original = source_task['sequence_budget']
    expected = sequence_budget(source_task['route']['length_m'],start_s=source_task['capture_start_route_m'],
        obstacle_s=original['obstacle_s_m'],speed_mps=source_task['nominal_speed_mps'],
        ticks=source_task['capture_ticks'],dt=original['capture_dt_s'])
    if _digest(original) != _digest(expected) or source_task['minimum_ticks'] != original['capture_ticks']:
        raise ValueError('parent short sequence budget changed or does not match task')
    actors = source_task['unmarked_behavior']['obstacles']
    if len(actors)>1 or any(abs(a['route_s_m']-original['obstacle_s_m'])>1e-8 for a in actors):
        raise ValueError('short budget must bind its single obstacle placement')
    current = sequence_budget(length_m,start_s=start_s,obstacle_s=obstacle_s,
        speed_mps=source_task['nominal_speed_mps'],ticks=source_task['capture_ticks'],dt=original['capture_dt_s'])
    current.update(version=VERSION,station_coordinate_frame=FRAME,
        parent_budget_sha256=_digest(original),parent_budget=deepcopy(original),
        station_remapping=dict(start=dict(parent=original['start_s_m'],surveyed=start_s),
            obstacle=dict(parent=original['obstacle_s_m'],surveyed=obstacle_s)),
        physical_route_length_m=float(length_m))
    return current


def phase_ranges(budget, *, clear=False):
    if clear:
        return {'right_lane_follow':[budget['start_s_m']+budget['cruise_speed_mps']*budget['required_history_s'],
                                    budget['planned_recovery_end_m']]}
    first,last = budget['planned_event_envelope_m']
    return dict(approach=[budget['start_s_m'],first],avoidance_and_return=[first,last],
                nominal_recovery=[last,budget['planned_recovery_end_m']])


def validate_teacher_budget(task, target, *, start_s, ticks):
    """Raise before recording when actual bodies/maneuver exceed the fixed budget."""
    from .planned import PlannedTarget
    from .target import _project_vertices
    if not isinstance(target,PlannedTarget):raise TypeError('runtime budget requires actual PlannedTarget')
    target.verify_contract()
    if not target.feasible:raise ValueError('runtime sequence budget needs a feasible physical teacher')
    budget = task['sequence_budget']
    if budget.get('version') not in (VERSION,ON_ROAD_VERSION) or budget.get('station_coordinate_frame') != FRAME:
        raise ValueError('capture requires a rebudgeted surveyed physical route')
    extended=budget['version']==ON_ROAD_VERSION
    if extended:
        declared=task['unmarked_behavior']['obstacles']
        if (len(declared)!=1 or declared[0]['kind']!='parked_vehicle'
                or declared[0].get('placement_policy')!='fully_on_road_physically_settled'
                or task['unmarked_behavior'].get('passing_space_policy')!='observed_clear_surface_during_static_parked_avoidance_only'):
            raise ValueError('extended transition budget requires explicit grounded on-road parked avoidance')
    expected = sequence_budget(task['route']['length_m'],start_s=task['capture_start_route_m'],
        obstacle_s=budget['obstacle_s_m'],speed_mps=task['nominal_speed_mps'],ticks=task['capture_ticks'],
        dt=budget['capture_dt_s'],maximum_transition_m=14. if extended else 8.)
    if any(budget.get(k) != v for k,v in expected.items() if k!='station_coordinate_frame'):
        raise ValueError('surveyed sequence budget was relaxed or changed')
    if (type(ticks) is not int or ticks != budget['capture_ticks']
            or abs(float(start_s)-budget['start_s_m'])>1e-7
            or task['minimum_ticks']!=ticks):
        raise ValueError('runtime start/ticks do not match surveyed sequence budget')
    road = task['route']['unmarked_road']
    ref = target.reference
    if (not np.array_equal(ref.s,np.asarray(task['route']['arc_m']))
            or not np.array_equal(ref.left_xyz,np.asarray(road['road_left_edge_xyz']))
            or not np.array_equal(ref.right_xyz,np.asarray(road['road_right_edge_xyz']))):
        raise ValueError('teacher target uses a different physical road')
    if (abs(ref.s[0])>1e-8 or abs(ref.s[-1]-task['route']['length_m'])>1e-7
            or abs(budget['physical_route_length_m']-ref.s[-1])>1e-7):
        raise ValueError('physical route metric length changed')
    specs = task['unmarked_behavior']['obstacles']
    if len(specs)!=len(target.obstacles) or len(specs)>1:
        raise ValueError('runtime obstacle count differs from short scene budget')
    effective_half_length = float(np.abs(target.ego.corners()[:,0]).max())
    if effective_half_length>budget['maximum_ego_half_length_m']+1e-8:
        raise ValueError('actual ego footprint exceeds sequence half-length bound')
    if (start_s<target.s[0] or budget['required_suffix_end_m']>target.s[-1]+1e-8):
        raise ValueError('physical teacher body support does not cover required capture suffix')
    grid = np.linspace(ref.s[0],ref.s[-1],int(np.ceil((ref.s[-1]-ref.s[0])/.05))+1)
    actor_reports = []
    for spec,obstacle in zip(specs,target.obstacles):
        if (spec['kind']!=obstacle.kind or obstacle.motion!='static'
                or abs(spec['route_s_m']-budget['obstacle_s_m'])>1e-7):
            raise ValueError('actual obstacle differs from fixed sequence scene')
        stations,_ = _project_vertices(ref,grid,obstacle.polygon())
        half = float(np.abs(stations-budget['obstacle_s_m']).max())
        margin = target.config.obstacle_clearance(obstacle.kind)
        if half>budget['maximum_actor_half_length_m']+1e-8:
            raise ValueError('actual actor footprint or placement exceeds sequence bound')
        if margin is None or margin>budget['maximum_obstacle_clearance_m']+1e-8:
            raise ValueError('actual obstacle clearance exceeds sequence bound')
        actor_reports.append(dict(identity=obstacle.identity,kind=obstacle.kind,
            maximum_longitudinal_extent_from_placement_m=half,clearance_m=margin))
    if bool(specs)!=bool(target.excursions):
        raise ValueError('short obstacle case must have nonzero avoidance and recovery')
    first,last = budget['planned_event_envelope_m']
    excursion_reports = []
    for event in target.excursions:
        departure = event.plateau_start_s-event.start_s
        recovery = event.end_s-event.plateau_end_s
        if (departure>budget['maximum_transition_m']+1e-8 or recovery>budget['maximum_transition_m']+1e-8
                or event.start_s<first-1e-8 or event.end_s>last+1e-8):
            raise ValueError('actual teacher transition/event exceeds short sequence envelope')
        if event.end_s+budget['minimum_retained_nominal_recovery_m']>budget['planned_recovery_end_m']+1e-8:
            raise ValueError('actual teacher cannot retain five metres nominal recovery')
        excursion_reports.append(dict(start_s=event.start_s,end_s=event.end_s,
            departure_transition_m=departure,recovery_transition_m=recovery))
    return dict(version=VERSION,passed=True,scope='actual_static_bodies_and_privileged_teacher_budget_only',
        budget_sha256=_digest(budget),teacher_contract_sha256=target.contract_sha256,
        ego_effective_half_length_m=effective_half_length,actors=actor_reports,excursions=excursion_reports,
        actual_body_dimensions_qualified=True,actual_teacher_excursions_qualified=True,
        actual_timing_qualified=False,observed_label_admission=False,training_ready=False)
