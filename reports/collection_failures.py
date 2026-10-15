"""Versioned retry and geometry failure memory for the unattended queue."""
import json
import time

from lane_dataset.capture_policy import CAPTURE_POLICY_VERSION


def failure_kind(result):
    errors=' '.join(map(str,result.get('errors',[])))
    for phrase,kind in (
        ('lateral_jerk_above_limit','physical_jerk'),
        ('lateral_acceleration_above_limit','physical_acceleration'),
        ('rendered_boundary_disagreement_above_five_cm','rendered_boundary'),
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
    # Version 7 corrects paint-width metrology and route progress. It does not
    # make missing boundary pixels visible or repair failed depth fixtures.
    # Do not spend another full capture on every version-6 warmup failure.
    if version == 6:
        return (result.get('status') in ('failed','rejected','worker_process_failed','geometry_failure_cached')
            and any('rendered_boundary_disagreement_above_five_cm' in str(e)
                    or 'mpc_optimization_failed' in str(e) for e in result.get('errors',[])))
    # Completed events rejected by corrected admission checks must also retry.
    corrected_admission=any(version<fixed and phrase in str(error)
        for error in result.get('errors',[])
        for phrase,fixed in (('insufficient_phase_candidates',2),
                             ('rendered_boundary_disagreement_above_five_cm',3)))
    return (version<CAPTURE_POLICY_VERSION
        and result.get('status') in ('failed','rejected','worker_process_failed','geometry_failure_cached')
        and (corrected_admission or failure_kind(result) in ('depth_fixture','road_surface','warmup',
            'controller','stopping_corridor','warmup_budget') or any('WinError 5' in str(e) or 'explicit stop file' in str(e)
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


def retime_complete_event(task, speed, calibration):
    """Keep the spatial event; budget a full capture at a measured lower speed."""
    import copy, math
    if not calibration.qualified or not calibration.speed_min_mps<=speed<=calibration.speed_max_mps:
        raise ValueError('slower retry requires qualified measured calibration')
    if not 0<speed<task['nominal_speed_mps']:
        raise ValueError('retry speed must be lower than the failed capture')
    result=copy.deepcopy(task)
    end=max(hi for lo,hi in result['maneuver_phase_ranges'].values())
    result['nominal_speed_mps']=speed
    result['minimum_ticks']=max(600,int(math.ceil(end/(calibration.speed_min_mps*.05)))+210)
    result['retry_reason']='complete event exceeded measured lateral dynamics; slower calibrated recapture'
    result['retry_original_speed_mps']=task['nominal_speed_mps']
    return result


def slower_physical_retries(output):
    """One bounded, slower retry per complete geometry with failed dynamics."""
    from pathlib import Path
    from reports.automatic_corpus import measured_calibration
    from reports.candidate_priority import preferred_split
    from lane_dataset.config import Config
    output=Path(output);state_path=output/'state.json'
    if not state_path.exists():return
    state=json.loads(state_path.read_text());seen=set()
    status_path=output/'status.json'
    counts=json.loads(status_path.read_text()).get('accepted_labels',{}) if status_path.exists() else {}
    plan_path=output.parent/'scenario-plan.automatic.json'
    assignments={r['group_id']:r['split'] for r in json.loads(plan_path.read_text()).get('assignments',[])} if plan_path.exists() else {}
    for job,final in sorted(state.get('completed',{}).items()):
        path=output/(job+'.job.result.json');task_path=output/(job+'.job.json')
        if final.get('status')=='accepted' or not path.is_file() or not task_path.is_file():continue
        raw=json.loads(path.read_text());errors=raw.get('errors',[])
        labels=raw.get('labels') or {}
        if not isinstance(labels,dict) or labels.get('label_count')!=20 or not errors:continue
        physical={'lateral_jerk_above_limit','lateral_acceleration_above_limit'}
        corrected={'rendered_boundary_agreement_missing_or_failed','rendered_boundary_disagreement_above_five_cm'}
        if not set(errors)&physical or set(errors)-(physical|corrected):continue
        task=json.loads(task_path.read_text())['task']
        # Do not repeatedly lower speed, retry failed geometry forever, or
        # override an existing validation/test assignment.
        if task.get('retry_reason') or task.get('nominal_speed_mps',0)<=2.5:continue
        key=task['group_id']
        if key in seen:continue
        split=assignments.get(key) or preferred_split(counts,Config().targets,
            ('test',) if task['town']=='Town05' else ('train','validation'))
        if split is None:continue
        retry=retime_complete_event(task,2.5,measured_calibration(2.5))
        retry.update(split=split,episode_id=key+'-'+split+'-slow25-0',appearance=0)
        seen.add(key)
        yield retry


def remember_failure(state,task,result):
    kind=failure_kind(result)
    version=result.get('capture_policy_version')
    unchanged=(version==6 and CAPTURE_POLICY_VERSION==7 and kind in (
        'physical_jerk','physical_acceleration','depth_fixture','road_surface','warmup','stopping_corridor','warmup_budget'))
    if kind is None or (version!=CAPTURE_POLICY_VERSION and not unchanged):return
    # Depth geometry and road collision meshes are appearance-independent.
    # Visibility/control failures retain the opportunity to try other weather.
    appearance='all' if kind in ('depth_fixture','road_surface','warmup_budget') else task.get('appearance',0)
    key=json.dumps([task['group_id'],appearance,CAPTURE_POLICY_VERSION,task.get('nominal_speed_mps')])
    state.setdefault('geometry_failures',{})[key]=dict(kind=kind,job=task['episode_id'],errors=result['errors'])
    # A third split/weather copy cannot repair repeatable physical failures.
    # Keep the first alternative appearance, then stop after both have failed.
    repeated=state.setdefault('repeated_geometry_failures',{})
    repeat_key=json.dumps([task['group_id'],kind,CAPTURE_POLICY_VERSION,task.get('nominal_speed_mps')])
    records=repeated.setdefault(repeat_key,{})
    records[str(task.get('appearance',0))]=task['episode_id']
    if len(records)>=2:
        state['geometry_failures'][json.dumps([task['group_id'],'all',CAPTURE_POLICY_VERSION,task.get('nominal_speed_mps')])]=dict(
            kind=kind,job=task['episode_id'],errors=result['errors'],confirmed_appearances=records)
    # Route IDs differ when their distant event/tail differs. Remember an
    # identical physical initialization corridor across those route variants.
    if kind=='warmup':
        signature=warmup_signature(task)
        if signature:
            entry=state.setdefault('warmup_failures',{}).setdefault(signature,dict(appearances={},jobs=[]))
            entry['appearances'][str(task.get('appearance',0))]=task['episode_id']
            if task['episode_id'] not in entry['jobs']:entry['jobs'].append(task['episode_id'])
            entry['errors']=result['errors']


def cached_failure(state,task):
    failures=state.get('geometry_failures',{})
    for appearance in ('all',task.get('appearance',0)):
        key=json.dumps([task['group_id'],appearance,CAPTURE_POLICY_VERSION,task.get('nominal_speed_mps')])
        if key in failures:return failures[key]
    if state.get('warmup_failures'):
        entry=state['warmup_failures'].get(warmup_signature(task))
        if entry and len(entry['appearances'])>=2:
            return dict(kind='repeated_physical_warmup_corridor',job=entry['jobs'][0],
                errors=entry['errors'],confirmed_jobs=entry['jobs'])
    return None


def warmup_signature(task):
    """Identity of both boundaries over the full maximum warmup travel budget."""
    import hashlib
    import numpy as np
    route=task.get('route',{})
    arc=np.asarray(route.get('arc_m',[]),float)
    speed=task.get('nominal_speed_mps')
    if len(arc)<2 or speed is None:return None
    distance=20.*(float(speed)+.2)
    if arc[-1]-arc[0]<distance:return None
    grid=np.linspace(arc[0],arc[0]+distance,int(distance)+2)
    boundaries=[]
    for side in ('left_xyz','right_xyz'):
        xyz=np.asarray(route.get(side,[]),float)
        if xyz.shape!=(len(arc),3) or not np.isfinite(xyz).all():return None
        boundaries.append(np.round(np.column_stack([np.interp(grid,arc,xyz[:,i]) for i in range(3)]),2).tolist())
    identity=[CAPTURE_POLICY_VERSION,task.get('town'),speed,task.get('physical_map_initialization'),boundaries]
    return hashlib.sha256(json.dumps(identity,separators=(',',':')).encode()).hexdigest()


def restore_failure_memory(output,state):
    """Restore measured outcomes once at startup; skips are not capture jobs."""
    from pathlib import Path
    output=Path(output)
    captures=state.setdefault('capture_jobs',{})
    for job,final in state.get('completed',{}).items():
        path=output/(job+'.job.result.json')
        if not path.is_file():continue
        result=json.loads(path.read_text(encoding='utf-8-sig'))
        captures[job]=dict(status=final['status'],capture_policy_version=result.get('capture_policy_version'))
        task_path=output/(job+'.job.json')
        if final['status']=='accepted' or not task_path.is_file():continue
        remember_failure(state,json.loads(task_path.read_text(encoding='utf-8-sig'))['task'],
            {**result,'errors':final.get('errors',result.get('errors',[]))})


def capture_statistics(state):
    from collections import Counter
    captures=state.get('capture_jobs',{})
    counts=Counter(row['status'] for row in captures.values())
    current=Counter(row['status'] for row in captures.values()
        if row.get('capture_policy_version')==CAPTURE_POLICY_VERSION)
    def report(values):
        total=sum(values.values());accepted=values['accepted']
        return dict(completed_jobs=total,accepted_jobs=accepted,rejected_jobs=total-accepted,
            acceptance_rate=accepted/total if total else None,by_outcome=dict(values))
    return dict(**report(counts),considered_candidates=len(state.get('completed',{})),
        skipped_before_capture=len(set(state.get('completed',{}))-set(captures)),
        current_policy=dict(version=CAPTURE_POLICY_VERSION,**report(current)))


def archive_previous_result(path):
    if path.exists():path.rename(path.with_name(path.stem+'.before-policy-'+str(CAPTURE_POLICY_VERSION)+'-'+str(time.time_ns())+'.json'))
