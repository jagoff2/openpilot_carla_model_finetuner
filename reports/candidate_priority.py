"""Schedule observable routes first without changing data admission or quotas.

Map markings are a scheduling hint, never proof of camera visibility. Unpainted
curbs and intersections remain eligible and still require their RGB/depth QA.
"""
from __future__ import annotations

PAINT = {'solid', 'broken', 'solidsolid', 'solidbroken', 'brokensolid',
         'brokenbroken', 'bottsdots'}
RANKING_VERSION = 2


def preferred_split(counts, targets, available):
    """Fill the smallest accepted/in-flight fraction of its configured target."""
    eligible = [s for s in available if counts.get(s, 0) < targets[s]]
    return min(eligible, key=lambda s: (counts.get(s, 0) / targets[s],
        ('train', 'validation', 'test').index(s))) if eligible else None


def scheduled_candidates(path, root):
    """Assign new geometry by split deficit; never move accepted held-out data.

    Independent cursors let a temporarily overrepresented split wait without
    losing its candidate pool. Boundary/family ranking is retained within each
    split. Accepted manifests and physical overlap still govern admission.
    """
    import json
    from pathlib import Path
    from lane_dataset.config import Config
    from reports.candidate_cache import read_cache
    root = Path(root)
    targets = Config().targets
    streams = {s: iter(read_cache(path, test=(s == 'test'))) for s in targets}
    metadata = {}
    plan_signature = None
    assignments = {}
    try:
        while streams:
            status_path = root / 'reports/automatic/status.json'
            status = json.loads(status_path.read_text(encoding='utf-8-sig')) if status_path.exists() else {}
            counts = dict(status.get('accepted_labels', {}))
            family_counts = dict(status.get('accepted_by_family', {}))
            for worker in status.get('active_workers', []):
                split, family = worker.get('split'), worker.get('family')
                if split is None:
                    job = worker['job']
                    if job not in metadata:
                        job_path = root / 'reports/automatic' / (job + '.job.json')
                        if not job_path.exists():
                            continue
                        task = json.loads(job_path.read_text(encoding='utf-8-sig'))['task']
                        metadata[job] = task['split'], task['family']
                    split, family = metadata[job]
                counts[split] = counts.get(split, 0) + Config().labels_per_episode
                key = split + '/' + family
                family_counts[key] = family_counts.get(key, 0) + Config().labels_per_episode
            split = preferred_split(counts, targets, streams)
            if split is None:
                return
            try:
                candidate = next(streams[split])
            except StopIteration:
                del streams[split]
                continue
            if family_counts.get(split + '/' + candidate['family'], 0) >= targets[split] // 5:
                continue
            plan = root / 'reports/scenario-plan.automatic.json'
            signature = (plan.stat().st_mtime_ns, plan.stat().st_size) if plan.exists() else None
            if signature != plan_signature:
                assignments = {r['group_id']: r['split'] for r in
                    json.loads(plan.read_text(encoding='utf-8-sig')).get('assignments', [])} if plan.exists() else {}
                plan_signature = signature
            if assignments.get(candidate['group_id'], split) != split:
                continue
            yield candidate, split
    finally:
        for stream in streams.values():
            stream.close()


def boundary_risk(task):
    route = task.get('route', {})
    arc = route.get('arc_m', [])
    markings = route.get('boundary_markings', [])
    phases = task.get('maneuver_phase_ranges', {})
    if len(arc) < 2 or len(markings) != len(arc) or not phases:
        return dict(tier=1, maximum_unpainted_gap_m=0., both_unpainted_m=0.,
                    painted_fraction=0., scope='unknown_boundary_metadata')
    # Include the entire physical warmup, not just the labelled event. An
    # unmarked tight junction before an otherwise painted event can prevent
    # initialization or use up its route budget. Untraversed preview after the
    # terminal stop must not penalize a route.
    start = arc[0]
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
