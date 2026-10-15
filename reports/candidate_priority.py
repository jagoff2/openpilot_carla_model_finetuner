"""Schedule observable routes first without changing data admission or quotas.

Map markings are a scheduling hint, never proof of camera visibility. Unpainted
curbs and intersections remain eligible and still require their RGB/depth QA.
"""
from __future__ import annotations

PAINT = {'solid', 'broken', 'solidsolid', 'solidbroken', 'brokensolid',
         'brokenbroken', 'bottsdots'}
RANKING_VERSION = 1


def boundary_risk(task):
    route = task.get('route', {})
    arc = route.get('arc_m', [])
    markings = route.get('boundary_markings', [])
    phases = task.get('maneuver_phase_ranges', {})
    if len(arc) < 2 or len(markings) != len(arc) or not phases:
        return dict(tier=1, maximum_unpainted_gap_m=0., both_unpainted_m=0.,
                    painted_fraction=0., scope='unknown_boundary_metadata')
    # Cover the event's full input history and the actual driven suffix. The
    # rest of the reserved route may contain junctions the car never reaches.
    start = max(0., phases['approach'][0] - 5.*task.get('nominal_speed_mps', 3.5))
    stop = task.get('terminal_stop_route_m', phases['recovery'][1])
    gaps = [0., 0.]
    run_start = [None, None]
    total = painted = both = 0.
    for i, (a, b) in enumerate(zip(arc, arc[1:])):
        lo, hi = max(start, a), min(stop, b)
        if hi <= lo:
            continue
        total += hi-lo
        visible = []
        for side_index, side in enumerate(('left', 'right')):
            marking = markings[i][side]
            present = (str(marking.get('type', '')).lower() in PAINT
                       and marking.get('width_m', 0.) > 0.)
            visible.append(present)
            if present:
                run_start[side_index] = None
            else:
                if run_start[side_index] is None:
                    run_start[side_index] = lo
                gaps[side_index] = max(gaps[side_index], hi-run_start[side_index])
        if all(visible):
            painted += hi-lo
        elif not any(visible):
            both += hi-lo
    gap = max(gaps)
    return dict(tier=0 if gap <= 8. else 1 if both <= 8. else 2,
                maximum_unpainted_gap_m=gap, both_unpainted_m=both,
                painted_fraction=painted/max(total, 1e-9),
                scope='map_marking_priority_only_not_visibility_or_admission')
