"""Versioned retry and geometry failure memory for the unattended queue."""
import json
import time

from lane_dataset.capture_policy import CAPTURE_POLICY_VERSION


def failure_kind(result):
    errors=' '.join(map(str,result.get('errors',[])))
    for phrase,kind in (
        ('known geometry depth fixture','depth_fixture'),
        ('no physical road surface','road_surface'),
        ('rolling warmup failed','warmup'),
        ('mpc_optimization_failed','controller'),
        ('insufficient_observed_stopping_corridor','stopping_corridor'),
        ('physical rolling initialization exceeded','warmup_budget'),
        ('insufficient pre-maneuver history','warmup_budget')):
        if phrase in errors:return kind
    return None


def retry_after_policy_change(result):
    version=result.get('capture_policy_version',0) or 0
    # Completed events rejected by corrected admission checks must also retry.
    corrected_admission=any(version<fixed and phrase in str(error)
        for error in result.get('errors',[])
        for phrase,fixed in (('insufficient_phase_candidates',2),
                             ('rendered_boundary_disagreement_above_five_cm',3)))
    return (version<CAPTURE_POLICY_VERSION
        and result.get('status') in ('failed','rejected','worker_process_failed','geometry_failure_cached')
        and (corrected_admission or failure_kind(result) is not None or any('WinError 5' in str(e) or 'explicit stop file' in str(e)
            for e in result.get('errors',[]))))


def priority_retries(output):
    """Retry completed events with corrected admission before unproven routes."""
    from pathlib import Path
    output=Path(output)
    state_path=output/'state.json'
    if not state_path.exists():return
    state=json.loads(state_path.read_text())
    candidates=[]
    for job,result in state.get('completed',{}).items():
        report=result.get('labels',{})
        labels=report.get('label_count',0) if isinstance(report,dict) else 0
        if labels and retry_after_policy_change(result):candidates.append((-labels,job))
    for _,job in sorted(candidates):
        path=output/(job+'.job.json')
        if path.is_file():
            task=json.loads(path.read_text())['task']
            if task.get('route') and task.get('maneuver_phase_ranges'):yield task


def remember_failure(state,task,result):
    kind=failure_kind(result)
    if kind is None or result.get('capture_policy_version')!=CAPTURE_POLICY_VERSION:return
    # Depth geometry and road collision meshes are appearance-independent.
    # Visibility/control failures retain the opportunity to try other weather.
    appearance='all' if kind in ('depth_fixture','road_surface','warmup_budget') else task.get('appearance',0)
    key=json.dumps([task['group_id'],appearance,CAPTURE_POLICY_VERSION])
    state.setdefault('geometry_failures',{})[key]=dict(kind=kind,job=task['episode_id'],errors=result['errors'])


def cached_failure(state,task):
    failures=state.get('geometry_failures',{})
    for appearance in ('all',task.get('appearance',0)):
        key=json.dumps([task['group_id'],appearance,CAPTURE_POLICY_VERSION])
        if key in failures:return failures[key]
    return None


def archive_previous_result(path):
    if path.exists():path.rename(path.with_name(path.stem+'.before-policy-'+str(CAPTURE_POLICY_VERSION)+'-'+str(time.time_ns())+'.json'))
