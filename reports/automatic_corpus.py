"""Resumable capture, measured admission, publication and corpus export.

Runs as a parent process in the visible console. An episode failure is a
recorded job result, never a reason to abandon the rest of the corpus queue.
No diagnostic frame is counted as an accepted training label.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,os,shutil,subprocess,sys,time,traceback
from collections import Counter
from functools import lru_cache
from pathlib import Path
import numpy as np
from lane_dataset.config import Config,PROJECT,DEFAULT_ROOT,sha256_file,write_json
from lane_dataset.settings import path_setting
from lane_dataset.simulator import world_owner
from lane_dataset.capture_policy import CAPTURE_POLICY_VERSION, speed_bounds

TRAINING_PYTHON=path_setting('training_python','LANE_TRAINING_PYTHON',sys.executable)
DEFAULT_PLAN=DEFAULT_ROOT/'planning/reports/scenario-tasks.json'
CALIBRATION_ROOT=DEFAULT_ROOT


def configure_root(root):
    global CALIBRATION_ROOT
    root=Path(root).resolve()
    if root!=CALIBRATION_ROOT:measured_calibration.cache_clear()
    CALIBRATION_ROOT=root


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def binding(path):return dict(path=str(Path(path).resolve()),sha256=sha256_file(path))
def log(event,**fields):print(json.dumps(dict(event=event,**fields),default=str),flush=True)

@lru_cache(maxsize=8)
def native_map(town):
    import carla
    source=Path(Config().carla_root)/f'CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr'
    return carla.Map(town,source.read_text())

@lru_cache(maxsize=4)
def measured_calibration(speed):
    from lane_dataset.capture import _calibration
    return _calibration(CALIBRATION_ROOT,speed)[0]

def event_direction(route,phases,family):
    if family=='straight_recovery':return 'straight'
    arc=np.asarray(route['arc_m']);curvature=np.asarray(route['curvature_1pm'])
    lo=phases['entry'][0];hi=phases['apex'][0] if family!='intersection_turn' else phases['exit'][1]
    selected=curvature[(arc>=lo)&(arc<=hi)]
    signed=float(np.sum(selected))
    if not len(selected) or abs(signed)<1e-7:raise ValueError('maneuver_direction_unresolved')
    return 'right' if signed>0 else 'left'


def physical_check(shard):
    from lane_dataset.quality import qualify_episode
    from lane_dataset.geometry import LaneReference
    from lane_dataset.admission import rendered_boundary_status,validate_capture_envelope
    m=read(shard/'manifest.json');g=read(shard/'geometry.json')
    with np.load(shard/'telemetry.npz',allow_pickle=False) as archive:t={k:archive[k] for k in archive.files}
    boundary=rendered_boundary_status(shard,m,t);envelope=validate_capture_envelope(shard,m,t)
    yaw=t['rotation_world'][:,2];a=t['acceleration_world'];d=m['vehicle_dimensions']
    physical=qualify_episode(timestamps_s=t['timestamp'],positions_xyz=t['position_world'],yaw_rad=yaw,
        speeds_mps=t['speed'],lateral_acceleration_mps2=-np.sin(yaw)*a[:,0]+np.cos(yaw)*a[:,1],
        steering_fractions=t['steer'],reference=LaneReference(g['left_xyz'],g['right_xyz'],g['arc_m']),
        vehicle_length_m=d['length'],vehicle_width_m=d['width'],vehicle_height_m=d['height'],
        vehicle_bbox_center_xyz=d['bbox_center'],rotation_world=t['rotation_world'],velocity_world=t['velocity_world'],
        collisions=len(m['collision_events']),rendered_boundary_agreement=boundary['passed'],
        completed_maneuver=t['route_s'][-1]>=m['maneuver_phase_ranges']['recovery'][1],
        intended_progress_m=m['intended_progress_m'],recovery=m.get('recovery',False),recovery_onset_s=m.get('recovery_onset_s',0.)).to_dict()
    return m,t,physical,boundary,envelope


def prepare_capture_task(task):
    """Check each queued task again in the worker before allocating simulator data."""
    from lane_dataset.inventory import sample_route,route_document
    from lane_dataset.boundary_reference import lane_facing_marking_reference
    task=copy.deepcopy(task);speed=task['nominal_speed_mps']
    task['survey_road_surface']=True
    if any(hi-lo<2.*speed-1e-6 for lo,hi in task['maneuver_phase_ranges'].values()):
        raise ValueError('phase_cannot_contain_four_distinct_half_second_labels_at_calibrated_speed')
    needed=task['warmup_distance_budget_m']+task['minimum_ticks']*.05*speed_bounds(measured_calibration(speed))[1]+task['planned_preview_m']+2.
    if task['route']['length_m']>=needed:return task
    start=task['start'];roadmap=native_map(task['town'])
    waypoint=roadmap.get_waypoint_xodr(start['road_id'],start['lane_id'],start['s'])
    points=sample_route(waypoint,needed+2.,.5,task.get('branch',0))
    _,route=route_document(points,task['town'])
    if route['length_m']<needed:raise ValueError('route_cannot_contain_complete_recording_and_preview_budget')
    old=task['route'];length=min(len(old['center_xyz']),len(route['center_xyz']))
    # Compare the same physical lane centers before applying marking offsets.
    old_center=(np.asarray(old['left_xyz'])+np.asarray(old['right_xyz']))*.5
    if np.max(np.linalg.norm(old_center[:length,:2]-np.asarray(route['center_xyz'])[:length,:2],axis=1))>.15:
        raise ValueError('extended_recording_budget_changed_selected_route')
    marks=[{side:dict(type=str(getattr(wp,side+'_lane_marking').type),width_m=float(getattr(wp,side+'_lane_marking').width)) for side in ('left','right')} for wp in points]
    route=lane_facing_marking_reference(route,marks)
    route['section']['family']=task['family'];route['section']['direction']=task['direction']
    group='auto-'+hashlib.sha256((task['town']+repr(np.round(route['center_xyz'],3).tolist())).encode()).hexdigest()[:20]
    task.update(route=route,group_id=group,episode_id=group+'-'+str(task['appearance']))
    return task


def retain_rejection_diagnostics(root,result):
    """Bound disk use by retaining reports, geometry, telemetry and audit PNGs."""
    if not result.get('shard'):return
    shard=Path(result['shard']).resolve();root=Path(root).resolve()
    if shard.parent!=root/'quarantine' or not (shard/'reservation.json').is_file():return
    reservation=read(shard/'reservation.json')
    if reservation.get('id')!=shard.name:raise ValueError('rejection cleanup reservation identity mismatch')
    manifest=read(shard/'manifest.json')
    if manifest.get('status')=='accepted' or manifest.get('accepted_label_count',0):raise ValueError('cannot compact accepted data')
    # A failed episode can contain useful, fully authenticated earlier frames.
    # Preserve payloads for ordinary admission/recovery; disk reservations still
    # account for them. Only empty preflight failures can be compacted here.
    if manifest.get('partial_recording_frames',0) or manifest.get('archives') or manifest.get('recoverable_payloads_preserved'):
        manifest['recoverable_payloads_preserved']=True
        manifest['automatic_rejection_errors']=result.get('errors',[])
        write_json(shard/'manifest.json',manifest)
        return
    deleted=[]
    for name in ('packed_narrow.npz','packed_wide.npz','evidence.jsonl.gz','rendered-boundary-samples.json.gz','candidates.json.gz'):
        path=shard/name
        if path.is_file():
            deleted.append(dict(path=name,bytes=path.stat().st_size,sha256=sha256_file(path)))
            path.unlink()
    manifest.update(retained_for_diagnostics_only=True,automatic_rejection_errors=result.get('errors',[]),discarded_rejected_payloads=deleted)
    manifest['files']={k:v for k,v in manifest.get('files',{}).items() if (shard/k).is_file()}
    write_json(shard/'manifest.json',manifest)


def ensure_server(root,port):
    """Reuse the expected CARLA server; recover its exact port after a crash."""
    import carla,psutil
    def ready():
        client=carla.Client('127.0.0.1',port);client.set_timeout(5.)
        version=client.get_server_version()
        if version!='0.9.16':raise ValueError('unexpected_simulator_version:'+version)
        return True
    for attempt in range(3):
        try:
            if ready():return
        except RuntimeError:
            log('simulator_reconnect',port=port,attempt=attempt+1)
    executable=Path(Config().carla_root)/'CarlaUE4.exe'
    if not executable.is_file():raise FileNotFoundError(executable)
    # Only the configured CARLA installation and exact requested port are owned.
    for process in psutil.process_iter(['pid','name','exe','cmdline']):
        try:
            info=process.info;command=info['cmdline'] or [];binary=Path(info['exe'] or '').resolve()
            if (Path(Config().carla_root).resolve() in binary.parents
                    and f'-carla-rpc-port={port}' in command and 'CarlaUE4' in info['name']):
                process.terminate()
                try:process.wait(timeout=10)
                except psutil.TimeoutExpired:process.kill();process.wait(timeout=10)
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    gpu=(port-2400)//100;path=Path(root)/'reports/automatic'/f'server-{port}.log'
    command=[str(executable),'-RenderOffScreen','-nosound','-quality-level=Epic',f'-graphicsadapter={gpu}',
        f'-carla-rpc-port={port}','-NoTextureStreaming','-ExecCmds=r.DynamicRes.OperationMode=0,r.ScreenPercentage=100',f'-abslog={path}']
    subprocess.Popen(command,cwd=executable.parent,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    deadline=time.monotonic()+120
    while time.monotonic()<deadline:
        try:
            if ready():log('simulator_recovered',port=port);return
        except RuntimeError:time.sleep(2)
    raise RuntimeError('simulator_restart_did_not_become_ready')


def execute_job(root,job_path,port,reprocess=False):
    configure_root(root)
    from lane_dataset.capture import capture_development,_finite_json
    from lane_dataset.admission import build_labels
    job=read(job_path);task=job['task'];result_path=job_path.with_suffix('.result.json')
    result=dict(job=job['id'],status='failed',task=task['episode_id'],capture_policy_version=CAPTURE_POLICY_VERSION)
    try:
        if task.get('terminal_stop_route_m') is not None:
            task=prepare_capture_task(task)
            skip=candidate_screen(root,task)
            if skip:raise ValueError(skip)
            write_json(job_path.with_suffix('.capture-task.json'),task)
        result['task']=task['episode_id']
        if reprocess:
            shard=Path(read(result_path)['shard'])
            if shard.resolve().parent!=root.resolve()/'quarantine':raise ValueError('reprocessing requires an owned quarantined capture')
            captured=dict(path=str(shard),manifest=read(shard/'manifest.json'))
            log('reprocessing_preserved_capture',episode=task['episode_id'],shard=shard)
        else:
            if task.get('route'):ensure_server(root,port)
            log('capture_start',episode=task['episode_id'],town=task['town'],port=port,ticks=task['minimum_ticks'])
            captured=capture_development(root,port,(port-2400)//100,task['minimum_ticks'],task=task)
        shard=Path(captured['path']);result['shard']=str(shard)
        if captured['manifest'].get('error') and (shard/'partial-telemetry.npz').exists():
            from lane_dataset.partial_capture import recover_capture_prefix
            if recover_capture_prefix(shard):
                captured['manifest']=read(shard/'manifest.json')
                log('recovered_complete_event_prefix',shard=shard)
        if captured['manifest'].get('error'):
            result['errors']=[captured['manifest']['error']['message']]
        else:
            from reports.recovery_annotation import annotate_recorded_recovery
            annotate_recorded_recovery(shard)
            log('label_build',episode=task['episode_id'],shard=shard)
            labels=build_labels(shard,dataset_root=root)
            from reports.choose_training_candidates import improve_candidate_coverage
            if improve_candidate_coverage(shard,labels):
                labels=build_labels(shard,dataset_root=root)
            m,t,physical,boundary,envelope=physical_check(shard)
            result.update(labels=labels,physical=physical,boundary=boundary,envelope=envelope)
            errors=list(labels['build_errors'])+list(physical['reasons'])+boundary['errors']+envelope['errors']
            if not errors:
                clip=subprocess.run([str(TRAINING_PYTHON),'-u','-m','lane_dataset.review_video',str(shard)],cwd=PROJECT,check=False)
                if clip.returncode:errors.append('review_clip_export_failed')
            result.update(status='measured_candidate_passed' if not errors else 'rejected',errors=sorted(set(errors)))
    except Exception as error:
        result.update(errors=[str(error)],traceback=traceback.format_exc())
    result=_finite_json(result)
    write_json(result_path,result)
    if result['status']!='measured_candidate_passed':retain_rejection_diagnostics(root,result)
    log('job_complete',**{k:v for k,v in result.items() if k not in ('labels','boundary','envelope','physical','traceback')})
    return 0  # Rejection is a persisted result. The parent advances its queue.


def normalize_task(task):
    """Add physical warmup before the event; end movement after recovery.

    The future suffix is retained at a physical stop, not by driving into an
    unrelated next junction. No missing maneuver phases are invented.
    """
    import carla
    from lane_dataset.inventory import sample_route,route_document
    from lane_dataset.boundary_reference import lane_facing_marking_reference
    from lane_dataset.geometry import LaneReference
    from lane_dataset.maneuvers import propose_maneuver
    speed=float(task.get('collection_speed_mps',task.get('nominal_speed_mps',3.5)))
    calibration=measured_calibration(speed)
    warmup={1.5:22.5,2.5:35.,3.5:45.}.get(speed,15.*speed)
    extension=warmup+5.*speed+2.
    roadmap=native_map(task['town']);start=roadmap.get_waypoint_xodr(task['start']['road_id'],task['start']['lane_id'],task['start']['s'])
    previous=sorted(start.previous(extension),key=lambda w:(w.road_id,w.lane_id,w.s))
    if not previous:raise ValueError('candidate_has_no_physical_warmup_approach')
    points=sample_route(previous[task.get('branch',0)%len(previous)],float(task['route']['length_m'])+100.,.5,task.get('branch',0))
    section,route=route_document(points,task['town'])
    marks=[{side:dict(type=str(getattr(wp,side+'_lane_marking').type),width_m=float(getattr(wp,side+'_lane_marking').width)) for side in ('left','right')} for wp in points]
    route=lane_facing_marking_reference(route,marks)
    reference=LaneReference(route['left_xyz'],route['right_xyz'],route['arc_m'])
    shift,offset=reference.project(np.asarray(task['route']['center_xyz'])[0,:2])
    if abs(offset)>.1 or not extension-10.<shift<extension+10.:raise ValueError('warmup_extension_does_not_rejoin_selected_candidate')
    recovery=task['family']=='straight_recovery'
    proposal=propose_maneuver({**task['route'],'nominal_speed_mps':speed},task['family'])
    if recovery:
        proposal=dict(maneuver_phase_ranges={name:[warmup+speed*(5+2*i),warmup+speed*(7+2*i)] for i,name in enumerate(('approach','entry','apex','exit','recovery'))},
            annotation_status='automatic_physical_recovery_event',annotation_blockers=[],
            phase_basis='Five two-second stages following measured release of the held initial offset; exact positions bound after capture')
    if 'maneuver_phase_ranges' in proposal and not recovery:
        proposal['maneuver_phase_ranges']={k:[v+shift for v in bounds] for k,bounds in proposal['maneuver_phase_ranges'].items()}
        for name in ('intent_required_route_range_m','desire_route_range_m'):
            if name in proposal:proposal[name]=[v+shift for v in proposal[name]]
    elif not recovery:
        proposal=propose_maneuver({**route,'nominal_speed_mps':speed},task['family'])
    if not proposal.get('maneuver_phase_ranges'):raise ValueError('complete_maneuver_unavailable:'+','.join(proposal.get('annotation_blockers',[])))
    phases=proposal['maneuver_phase_ranges']
    if phases['approach'][0]<warmup+5.*speed-1e-6:raise ValueError('maneuver_does_not_leave_retained_five_second_prefix')
    if any(hi-lo<2.*speed-1e-6 for lo,hi in phases.values()):
        raise ValueError('phase_cannot_contain_four_distinct_half_second_labels_at_calibrated_speed')
    stop=phases['recovery'][1]+10.
    ticks=600 if recovery else int(math.ceil(phases['recovery'][1]/(calibration.speed_min_mps*.05)))+210
    # Reserve the full conservative speed-bound envelope checked by capture,
    # even though movement will finish at the declared physical stop.
    required_length=max(stop+30.,warmup+ticks*.05*speed_bounds(calibration)[1]+25.)+2.
    if route['length_m']<required_length:
        points=sample_route(previous[task.get('branch',0)%len(previous)],required_length+2.,.5,task.get('branch',0))
        section,route=route_document(points,task['town'])
        marks=[{side:dict(type=str(getattr(wp,side+'_lane_marking').type),width_m=float(getattr(wp,side+'_lane_marking').width)) for side in ('left','right')} for wp in points]
        route=lane_facing_marking_reference(route,marks)
        reference=LaneReference(route['left_xyz'],route['right_xyz'],route['arc_m'])
        if route['length_m']<required_length:raise ValueError('route_cannot_contain_complete_recording_and_preview_budget')
    digest=hashlib.sha256((task['town']+repr(np.round(reference.center_xyz,3).tolist())).encode()).hexdigest()[:20]
    group='auto-'+digest
    result={**task,**proposal,'group_id':group,'episode_id':group+'-'+str(task['appearance']),
        'route':route,'start':route['start'],'nominal_speed_mps':speed,'minimum_ticks':ticks,
        'physical_map_initialization':True,'warmup_distance_budget_m':warmup,'planned_preview_m':25.,
        'terminal_stop_route_m':stop,'maximum_control_rejection_frames':20,'semantic_qa':True,'survey_road_surface':True}
    result['direction']=event_direction(route,phases,task['family'])
    result['route']['section']['family']=task['family'];result['route']['section']['direction']=result['direction']
    if recovery:
        result.update(recovery=True,initial_recovery_offset_m=.25 if task['appearance']%2==0 else -.25,minimum_ticks=600)
    return result


@lru_cache(maxsize=512)
def accepted_summary(path,modified_ns,size):
    manifest=read(path)
    fields=('split','family','direction','geometry_group_id','episode_id','accepted_label_count')
    return {**{name:manifest[name] for name in fields},
        'history_crop_loss_count':sum(r['visibility_category']=='history_crop_loss' for r in manifest['label_records'])}


def accepted_manifests(root):
    return [accepted_summary(str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in sorted((root/'accepted').glob('*/manifest.json'))]


def publish_candidate(root,result,task,test_report):
    """Freeze an append-only split version, bind measured evidence, then replay admission."""
    from lane_dataset.splits import GeometrySection,RoadInterval,GeometryGroup,SplitAssignment,audit_split_overlap,group_geometry_sections
    from lane_dataset.storage import Store
    from reports.compact_payloads import compact_capture
    candidate=read(Path(result['shard'])/'manifest.json')
    # Keep the requested aggregate quota feasible without imposing an extra
    # per-family or per-episode crop-loss requirement.
    crop=sum(r['visibility_category']=='history_crop_loss' for r in candidate['label_records'])
    related=[m for m in accepted_manifests(root) if m['split']==candidate['split']]
    target=Config().targets[candidate['split']]
    remaining=target-20*(len(related)+1)
    required=math.ceil(.25*target)-crop-sum(m['history_crop_loss_count'] for m in related)
    if required>remaining:
        raise ValueError(f'aggregate_crop_loss_quota_would_be_unreachable:need_{required}_in_{remaining}_remaining_labels')
    compact_capture(Path(result['shard']))
    shard=Path(result['shard']);m=read(shard/'manifest.json');g=read(shard/'geometry.json')
    previous_path=root/'reports/scenario-plan.automatic.json'
    previous=read(previous_path) if previous_path.exists() else dict(groups=[],assignments=[],splits_frozen=True)
    plan=copy.deepcopy(previous);group_id=m['geometry_group_id']
    section=g['section'];section={**section,'family':m['family'],'direction':m['direction']}
    if not any(v['group_id']==group_id for v in plan['groups']):
        plan['groups'].append(dict(group_id=group_id,sections=[section]))
        plan['assignments'].append(dict(group_id=group_id,split=m['split'],family=m['family'],appearances=2 if m['split']=='train' else 1,seed=m['seed']))
    groups=[GeometryGroup(v['group_id'],tuple(GeometrySection(**{**s,'road_intervals':tuple(RoadInterval(**i) for i in s['road_intervals'])}) for s in v['sections'])) for v in plan['groups']]
    assignments=[SplitAssignment(**v) for v in plan['assignments']]
    conflicts=audit_split_overlap(groups,assignments)
    # Independent groups cannot be manufactured from overlapping same-split routes either.
    owner={s.section_id:v.group_id for v in groups for s in v.sections}
    for merged in group_geometry_sections([s for v in groups for s in v.sections]):
        if len({owner[s.section_id] for s in merged.sections})>1:conflicts.append('candidate_overlaps_existing_independent_geometry')
    if conflicts:raise ValueError(';'.join(conflicts))
    version=str(time.time_ns());directory=root/'qualification-versions'/version;directory.mkdir(parents=True)
    split_path=directory/'splits.json';write_json(split_path,plan)
    evidence={}
    measurements={
        'controller':dict(passed=result['physical']['accepted'],metrics=result['physical']['metrics']),
        'rendered_markings':dict(passed=result['boundary']['passed'],measurements=result['boundary']['measurements'],maximum_disagreement_m=result['boundary']['maximum_disagreement_m']),
        'depth':dict(passed=m['depth_qualified'],measurements=read(shard/'depth-fixture.json')),
        'worker_benchmark':dict(passed=not m.get('error'),measurements=m['measured_throughput'],scope='Measured complete single-worker capture, including actual warmup and terminal suffix; no two-worker speedup claim')}
    for key,value in measurements.items():
        path=directory/(key+'.json');write_json(path,value);evidence[key]=binding(path)
    frozen_tests=directory/'tests.xml';shutil.copy2(test_report,frozen_tests)
    for key in ('runtime_action','hidden_road_perturbation','split_overlap','preprocessing'):evidence[key]=binding(frozen_tests)
    qualification=dict(schema_version=1,config_sha256=Config().digest(),
        sources={str(p):sha256_file(p) for p in sorted((PROJECT/'lane_dataset').glob('*.py'))},
        evidence=evidence,split_binding=binding(split_path),sealed_path=str((directory/'qualification.json').relative_to(root)))
    write_json(directory/'qualification.json',qualification)
    config_path=root/'config.json';config=read(config_path)
    if config['sha256']!=Config().digest():raise ValueError('prepared_configuration_changed')
    write_json(config_path,{**config,'frozen':True})
    review_path=directory/'automated-review.json'
    frozen_clip=directory/'review-stock-crops.mp4';shutil.copy2(shard/'review-stock-crops.mp4',frozen_clip)
    review=dict(review_mode='automated_measured_v1',human_reviewed=False,reviewer='automatic measured physical/pixel/phase validation',decision='pass',
        geometry_sha256=sha256_file(shard/'geometry.json'),clip=binding(frozen_clip),
        frame_count=m['measured_throughput']['ticks'],label_indices=[r['label_index'] for r in m['label_records']],
        phases=['approach','entry','apex','exit','recovery'],rendered_boundary_measurement=result['boundary'],
        **{name:m.get(name) for name in ('maneuver_phase_ranges','intent_required_route_range_m','desire_route_range_m')})
    write_json(review_path,review);m['automatic_review_file']=str(review_path.relative_to(root));write_json(shard/'manifest.json',m)
    old_qualification=read(root/'qualification.json') if (root/'qualification.json').exists() else None
    write_json(root/'qualification.json',qualification)
    try:destination=Store(root).publish_qualified(shard)
    except Exception:
        if old_qualification is not None:write_json(root/'qualification.json',old_qualification)
        else:(root/'qualification.json').unlink(missing_ok=True)
        raise
    write_json(previous_path,plan);write_json(root/'reports/scenario-plan.json',plan)
    return destination


def snapshot(root,state):
    manifests=accepted_manifests(root);counts=Counter(m['split'] for m in manifests)
    family=Counter((m['split'],m['family']) for m in manifests)
    document=dict(updated=time.time(),pid=os.getpid(),status=state.get('status','running'),
        accepted_labels={s:20*counts[s] for s in Config().targets},target_labels=Config().targets,
        accepted_total=20*len(manifests),target_total=5000,active=state.get('active'),
        completed_jobs=len(state['completed']),rejected_jobs=sum(v.get('status')!='accepted' for v in state['completed'].values()),
        accepted_by_family={s+'/'+f:20*n for (s,f),n in family.items()},last_result=state.get('last_result'))
    document['history_crop_loss_labels']={s:sum(m['history_crop_loss_count'] for m in manifests if m['split']==s) for s in Config().targets}
    document['active_workers']=list(state.get('workers',{}).values())
    document['configured_workers']=state.get('configured_workers',[])
    write_json(root/'reports/automatic/status.json',document)
    write_json(root/'training-index.partial.json',dict(schema_version=1,complete=False,accepted_labels=document['accepted_total'],
        episodes=[str(p.relative_to(root)) for p in sorted((root/'accepted').glob('*/manifest.json'))],config_sha256=Config().digest()))
    return document


def prepare_output(root):
    from lane_dataset.cli import prepare
    from lane_dataset.capture import _calibration
    prepare(root)
    for speed in (1.5,2.5,3.5):_calibration(root,speed)


def wait_worker(root,state,state_path,job_id,job_path,result_path,reprocess=False):
    """Reattach the exact surviving worker instead of opening a second tick owner."""
    import psutil
    active=state.get('active') or {};process=None;child=None
    if active.get('job')==job_id and active.get('pid'):
        try:
            candidate=psutil.Process(active['pid'])
            command=candidate.cmdline()
            if 'reports.automatic_corpus' in command and str(job_path) in command:process=candidate
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    if process is None:
        # Publication of a recovered earlier episode can replace the active
        # status record while a later capture survives the parent restart.
        matches=[]
        for candidate in psutil.process_iter(['pid','ppid','cmdline']):
            command=candidate.info.get('cmdline') or []
            if ('reports.automatic_corpus' in command and '--job' in command
                    and str(job_path) in command and str(root) in command):matches.append(candidate)
        identities={candidate.pid for candidate in matches}
        roots=[candidate for candidate in matches if candidate.info.get('ppid') not in identities]
        if len(roots)>1:raise RuntimeError('multiple existing capture workers match the same job')
        if roots:process=roots[0]
    path=root/'reports/automatic'/(job_id+'.log')
    with path.open('a',encoding='utf-8',buffering=1) as stream:
        if process is None:
            command=[sys.executable,'-u','-m','reports.automatic_corpus','--root',str(root),'--job',str(job_path),'--port','2400']
            if reprocess:command.append('--reprocess')
            child=subprocess.Popen(command,cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            process=psutil.Process(child.pid)
        state['active']=dict(job=job_id,pid=process.pid,log=str(path),stage='capture_and_label_qualification')
        write_json(state_path,state)
        with path.open(encoding='utf-8',errors='replace') as reader:
            while True:
                for line in reader:print(line,end='',flush=True)
                snapshot(root,state)
                if (child is not None and child.poll() is not None) or not process.is_running():break
                time.sleep(2)
        if not result_path.exists():
            from lane_dataset.storage import Store
            effective=job_path.with_suffix('.capture-task.json')
            episode=(read(effective) if effective.exists() else read(job_path)['task'])['episode_id']
            saved=None
            for row in reversed(Store(root).rows()):
                if row['episode']!=episode or row['status']!='quarantined':continue
                shard=Path(row['path']);manifest=shard/'manifest.json'
                if not manifest.exists():continue
                captured=read(manifest)
                if not captured.get('error') and all((shard/name).is_file() for name in ('packed_narrow.npz','packed_wide.npz','evidence.jsonl.gz','telemetry.npz','candidates.json.gz')):
                    saved=dict(job=job_id,status='completed_capture_needs_postprocessing',shard=str(shard));break
            write_json(result_path,saved or dict(status='worker_process_failed',errors=[f'worker_exit:{child.returncode if child else "orphan_exited"}']))


def candidate_screen(root,task):
    """Reject quota and physical overlap conflicts before spending simulator time."""
    from lane_dataset.splits import GeometrySection,RoadInterval,group_geometry_sections
    from lane_dataset.maneuvers import maneuver_event_errors
    if task.get('route'):
        semantic_errors=maneuver_event_errors(task['route'],task['family'],task.get('maneuver_phase_ranges'))
        if semantic_errors:return semantic_errors[0]
    manifests=accepted_manifests(root);related=[m for m in manifests if m['split']==task['split'] and m['family']==task['family']]
    per_family=Config().targets[task['split']]//100
    if len(related)>=per_family:return 'quota_already_filled'
    groups={m['geometry_group_id'] for m in related};existing=task['group_id'] in groups
    max_groups=per_family//(2 if task['split']=='train' else 1)
    if not existing and len(groups)>=max_groups:return 'independent_group_quota_already_reserved'
    if task['family']!='straight_recovery':
        if task['direction'] not in ('left','right'):return 'turn_direction_unresolved'
        same={m['geometry_group_id'] for m in related if m['direction']==task['direction']}
        if not existing and len(same)>=math.ceil(max_groups/2):return 'turn_direction_quota_already_filled'
    if (task['town']=='Town05')!=(task['split']=='test'):return 'town05_test_isolation'
    plan=root/'reports/scenario-plan.automatic.json'
    if plan.exists():
        document=read(plan)
        if task['group_id'] in {g['group_id'] for g in document['groups']}:
            assignments=[a for a in document['assignments'] if a['group_id']==task['group_id']]
            if len(assignments)!=1:return 'existing_geometry_group_assignment_invalid'
            if assignments[0]['split']!=task['split']:
                return 'geometry_group_already_assigned_to_'+assignments[0]['split']
            return None
        def section(value):return GeometrySection(**{**value,'road_intervals':tuple(RoadInterval(**i) for i in value['road_intervals'])})
        proposed=section(task['route']['section'])
        for group in document['groups']:
            for old in group['sections']:
                if len(group_geometry_sections([proposed,section(old)]))<2:return 'overlap_with_accepted_independent_group'
    return None


def needs_recovery_policy_retry(result):
    """Retry an old worker once when its retained telemetry proves the new tail."""
    if not any(str(error).startswith('insufficient_phase_candidates:recovery:') for error in result.get('errors',[])):return False
    shard=Path(result.get('shard',''))
    if not (shard/'manifest.json').exists() or (shard/'recovery-annotation.json').exists():return False
    from reports.recovery_annotation import recovery_annotation
    manifest=read(shard/'manifest.json');geometry=read(shard/'geometry.json')
    with np.load(shard/'telemetry.npz',allow_pickle=False) as values:telemetry={k:values[k] for k in values.files}
    return recovery_annotation(manifest,geometry,telemetry) is not None


def candidate_queue(tasks,root=None):
    if root is not None:
        from reports.collection_failures import priority_retries
        for task in priority_retries(Path(root)/'reports/automatic'):
            yield task,True
    pilot_path=Path(root or DEFAULT_ROOT)/'planning/town06-flat-turn-plan.json'
    pilot=read(pilot_path)[0]
    pilot.update(episode_id='auto-town06-road40-terminal-0',group_id='auto-town06-road40-terminal',split='train',terminal_stop_route_m=170.,semantic_qa=True)
    yield pilot,True
    yield {**pilot,'appearance':1,'episode_id':'auto-town06-road40-terminal-1'},True
    pool=None if root is None else Path(root)/'reports/automatic/screened-pool.json'
    if pool is not None and (pool.exists() or pool.with_suffix('.sqlite').exists()):
        from reports.candidate_cache import build_cache,read_cache
        cache=pool.with_suffix('.sqlite')
        if not cache.exists():cache=build_cache(pool)
        for candidate in read_cache(cache):
            try:direction=event_direction(candidate['route'],candidate['maneuver_phase_ranges'],candidate['family'])
            except ValueError:continue
            for split in (('test',) if candidate['town']=='Town05' else ('validation','train')):
                for appearance in range(2 if split=='train' else 1):
                    task=copy.deepcopy(candidate)
                    task['direction']=direction
                    task['route']['section']['direction']=task['direction']
                    task.update(split=split,appearance=appearance,seed=candidate['seed']+appearance,
                        episode_id=candidate['group_id']+'-'+split+'-'+str(appearance))
                    yield task,True
    for task in tasks:yield task,False
    # Replacement candidates are discovered automatically when the initial plan
    # lacks usable routes. Every expanded envelope is screened against accepted
    # geometry before capture, regardless of lane direction or appearance.
    import carla
    from lane_dataset.inventory import sample_route,route_document
    from lane_dataset.splits import APPROVED_TOWNS
    for spacing in (25.,10.,5.):
        for town in APPROVED_TOWNS:
            path=Path(Config().carla_root)/f'CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr'
            roadmap=carla.Map(town,path.read_text())
            for start in sorted(roadmap.generate_waypoints(spacing),key=lambda w:(w.road_id,w.lane_id,w.s)):
                for branch in range(3):
                    for length in (100.,140.):
                        points=sample_route(start,length,.5,branch)
                        if len(points)<int(length/.5)*.95:continue
                        section,route=route_document(points,town)
                        group='search-'+section.section_id
                        seed=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)
                        for split in (('test',) if town=='Town05' else ('validation','train')):
                            for appearance in range(2 if split=='train' else 1):
                                yield dict(episode_id=f'{group}-{split}-{appearance}',group_id=group,
                                    town=town,start=route['start'],route=route,branch=branch,split=split,
                                    family=section.family,direction=section.direction,appearance=appearance,
                                    procedural=False,seed=seed+appearance),False


def _run_legacy_serial(root,plan_path,test_report):
    from lane_dataset.storage import Store
    from lane_dataset.cli import validate
    output=root/'reports/automatic';output.mkdir(parents=True,exist_ok=True)
    with world_owner('automatic-corpus-supervisor',root):
        prepare_output(root)
        Store(root).recover();state_path=output/'state.json'
        state=read(state_path) if state_path.exists() else dict(completed={},status='running',plan=str(plan_path))
        tasks=read(plan_path)
        state['status']='running'
        for original,prepared in candidate_queue(tasks,root):
            job_id=original['episode_id']
            if job_id in state['completed']:continue
            try:
                task=copy.deepcopy(original) if prepared else normalize_task(original)
                capture_task=output/(job_id+'.job.capture-task.json')
                actual=read(capture_task) if capture_task.exists() else task
                published=root/'accepted'/actual['episode_id']
                if published.exists():
                    state['completed'][job_id]=dict(status='accepted',path=str(published),labels=20,resumed_after_publication=True)
                    continue
                skip=candidate_screen(root,task)
                if skip:
                    state['completed'][job_id]=dict(status=skip);continue
                job_path=output/(job_id+'.job.json');write_json(job_path,dict(id=job_id,task=task))
                result_path=job_path.with_suffix('.result.json')
                if not result_path.exists():
                    wait_worker(root,state,state_path,job_id,job_path,result_path)
                result=read(result_path)
                if result['status']=='completed_capture_needs_postprocessing':
                    log('resume_completed_capture_postprocessing',job=job_id,shard=result['shard'])
                    wait_worker(root,state,state_path,job_id,job_path,result_path,reprocess=True)
                    result=read(result_path)
                retry_marker=result_path.with_name(result_path.stem+'.before-recovery-policy.json')
                if needs_recovery_policy_retry(result) and not retry_marker.exists():
                    shutil.copy2(result_path,retry_marker);result_path.unlink()
                    log('retry_with_measured_recovery_annotation',job=job_id)
                    wait_worker(root,state,state_path,job_id,job_path,result_path)
                    result=read(result_path)
                if result['status']=='measured_candidate_passed':
                    state['active']=dict(job=job_id,stage='independent_admission_and_publication');snapshot(root,state)
                    destination=publish_candidate(root,result,task,test_report)
                    result=dict(status='accepted',path=str(destination),labels=20)
                state['completed'][job_id]=result;state['last_result']={**result,'job':job_id}
                log('corpus_progress',**snapshot(root,state))
                if sum(m['accepted_label_count'] for m in accepted_manifests(root))==5000:
                    validation=validate(root)
                    if validation['status']=='release_ready':
                        index=read(root/'training-index.partial.json');index['complete']=True
                        write_json(root/'training-index.json',index);state['status']='complete';break
                    state['last_result']=validation
            except Exception as error:
                if 'result' in locals() and result.get('job')==job_id and result.get('shard'):
                    retained=Path(result['shard'])
                    if retained.parent==root/'quarantine' and retained.exists():
                        retain_rejection_diagnostics(root,{**result,'errors':[str(error)]})
                state['completed'][job_id]=dict(status='rejected',errors=[str(error)],traceback=traceback.format_exc())
                state['last_result']=dict(job=job_id,errors=[str(error)])
                log('candidate_rejected',job=job_id,error=str(error))
            finally:
                state['active']=None;write_json(state_path,state);snapshot(root,state)
        if state['status']!='complete':state['status']='candidate_pool_exhausted_with_reported_deficits'
        write_json(state_path,state);log('corpus_status',**snapshot(root,state))


def run(root,plan_path,test_report):
    """All supervisor entry points use both approved collection GPUs."""
    from reports.parallel_corpus import run as run_parallel
    configure_root(root)
    return run_parallel(root,plan_path,test_report)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--plan',type=Path);parser.add_argument('--job',type=Path)
    parser.add_argument('--port',type=int,default=2400);parser.add_argument('--test-report',type=Path)
    parser.add_argument('--reprocess',action='store_true')
    args=parser.parse_args()
    args.root=args.root.resolve()
    if args.job:return execute_job(args.root,args.job,args.port,args.reprocess)
    run(args.root,args.plan or args.root/'planning/reports/scenario-tasks.json',args.test_report or args.root/'reports/tests.xml')

if __name__=='__main__':raise SystemExit(main())
