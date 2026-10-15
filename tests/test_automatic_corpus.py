import json
import numpy as np
from lane_dataset.automatic_review import review_errors
from lane_dataset.config import write_json
from reports.automatic_corpus import execute_job


def test_review_holds_remove_training_index_entries_but_preserve_artifacts(tmp_path,monkeypatch):
    from reports import automatic_corpus as runner
    summaries=[]
    for episode in ('clean','held'):
        path=tmp_path/'accepted'/episode/'manifest.json'
        path.parent.mkdir(parents=True);path.write_text('{}')
        summaries.append(dict(episode_id=episode,split='train',family='straight_recovery',history_crop_loss_count=0))
    write_json(tmp_path/'reports/review-holds.json',dict(schema_version=1,
        episodes={'held':dict(status='held',reasons=['operator review'])}))
    monkeypatch.setattr(runner,'accepted_manifests',lambda root:summaries)
    result=runner.snapshot(tmp_path,dict(completed={},status='paused'))
    assert result['accepted_total']==20
    assert result['accepted_artifact_labels']==40 and result['review_held_labels']==20
    index=json.loads((tmp_path/'training-index.partial.json').read_text())
    assert len(index['episodes'])==1 and 'clean' in index['episodes'][0]
    assert (tmp_path/'accepted/held/manifest.json').read_text()=='{}'


def test_failed_capture_is_checkpointed_without_abandoning_parent_queue(tmp_path,monkeypatch):
    from lane_dataset import capture
    monkeypatch.setattr(capture,'capture_development',lambda *a,**k:dict(path=str(tmp_path/'quarantine'),manifest={'error':{'message':'lost evidence'}}))
    job=tmp_path/'one.job.json'
    write_json(job,dict(id='one',task=dict(episode_id='one',town='Town06',minimum_ticks=600)))
    assert execute_job(tmp_path,job,2400)==0
    result=json.loads(job.with_suffix('.result.json').read_text())
    assert result['status']=='failed' and result['errors']==['lost evidence']
    assert result.get('labels') is None


def test_machine_review_requires_actual_complete_phases_and_declares_no_human_review():
    names=['approach','entry','apex','exit','recovery']
    phases={name:[10*i,10*i+10] for i,name in enumerate(names)}
    records=[dict(phase=name,label_index=10*i+j) for i,name in enumerate(names) for j in (1,3,5,7)]
    review=dict(review_mode='automated_measured_v1',human_reviewed=False,frame_count=51,label_indices=[r['label_index'] for r in records])
    assert not review_errors(review,dict(maneuver_phase_ranges=phases),dict(timestamp=np.arange(51),route_s=np.arange(51)),records)
    changed={**review,'human_reviewed':True}
    assert 'automatic_review_mode_invalid' in review_errors(changed,dict(maneuver_phase_ranges=phases),dict(timestamp=np.arange(51),route_s=np.arange(51)),records)
    assert 'automatic_review_complete_maneuver_not_recorded' in review_errors(review,dict(maneuver_phase_ranges=phases),dict(timestamp=np.arange(51),route_s=np.arange(51)-1),records)


def test_resume_reattaches_exact_child_without_launching_second_tick_owner(tmp_path,monkeypatch):
    from reports import automatic_corpus as runner
    import psutil
    from types import SimpleNamespace
    output=tmp_path/'reports/automatic';output.mkdir(parents=True)
    job=output/'one.job.json';result=output/'one.job.result.json'
    write_json(result,dict(status='failed',errors=['recorded rejection']))
    state=dict(active=dict(job='one',pid=123),completed={})
    process=SimpleNamespace(pid=123,cmdline=lambda:['python','-m','reports.automatic_corpus','--job',str(job)],is_running=lambda:False)
    monkeypatch.setattr(psutil,'Process',lambda pid:process)
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *a,**k:(_ for _ in ()).throw(AssertionError('duplicate worker')))
    runner.wait_worker(tmp_path,state,output/'state.json','one',job,result)
    assert state['active']['pid']==123


def test_rejection_compaction_preserves_diagnostics_and_refuses_accepted(tmp_path):
    from reports.automatic_corpus import retain_rejection_diagnostics
    import pytest
    shard=tmp_path/'quarantine'/'one-a1';shard.mkdir(parents=True)
    write_json(shard/'reservation.json',dict(id=shard.name))
    write_json(shard/'manifest.json',dict(status='development',accepted_label_count=0,files={}))
    (shard/'packed_narrow.npz').write_bytes(b'rejected archive')
    (shard/'partial-telemetry.npz').write_bytes(b'diagnostic telemetry')
    retain_rejection_diagnostics(tmp_path,dict(shard=str(shard),errors=['control rejection']))
    assert not (shard/'packed_narrow.npz').exists()
    assert (shard/'partial-telemetry.npz').read_bytes()==b'diagnostic telemetry'
    write_json(shard/'manifest.json',dict(status='accepted',accepted_label_count=20))
    with pytest.raises(ValueError,match='accepted'):retain_rejection_diagnostics(tmp_path,dict(shard=str(shard)))


def test_interrupted_capture_keeps_all_earlier_training_payloads(tmp_path):
    from reports.automatic_corpus import retain_rejection_diagnostics
    shard=tmp_path/'quarantine'/'one-a1';shard.mkdir(parents=True)
    write_json(shard/'reservation.json',dict(id=shard.name))
    write_json(shard/'manifest.json',dict(status='development',accepted_label_count=0,partial_recording_frames=320))
    for name in ('packed_narrow.npz','packed_wide.npz','evidence.jsonl.gz','candidates.json.gz'):
        (shard/name).write_bytes(b'recoverable evidence')
    retain_rejection_diagnostics(tmp_path,dict(shard=str(shard),errors=['late capture failure']))
    assert (shard/'packed_narrow.npz').read_bytes()==b'recoverable evidence'
    assert json.loads((shard/'manifest.json').read_text())['recoverable_payloads_preserved']


def test_resume_discovers_surviving_worker_after_status_was_replaced(tmp_path,monkeypatch):
    from reports import automatic_corpus as runner
    import psutil
    from types import SimpleNamespace
    output=tmp_path/'reports/automatic';output.mkdir(parents=True)
    job=output/'one.job.json';result=output/'one.job.result.json'
    write_json(result,dict(status='failed',errors=['recorded rejection']))
    state=dict(active=None,completed={})
    command=['python','-m','reports.automatic_corpus','--root',str(tmp_path),'--job',str(job)]
    wrapper=SimpleNamespace(pid=123,info=dict(pid=123,ppid=1,cmdline=command),is_running=lambda:False)
    actual=SimpleNamespace(pid=124,info=dict(pid=124,ppid=123,cmdline=command),is_running=lambda:False)
    monkeypatch.setattr(psutil,'process_iter',lambda *a:[wrapper,actual])
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *a,**k:(_ for _ in ()).throw(AssertionError('duplicate worker')))
    runner.wait_worker(tmp_path,state,output/'state.json','one',job,result)
    assert state['active']['pid']==123


def test_finished_capture_is_reprocessed_after_worker_exits_before_result(tmp_path,monkeypatch):
    from reports import automatic_corpus as runner
    from lane_dataset.storage import Store
    import psutil
    from types import SimpleNamespace
    output=tmp_path/'reports/automatic';output.mkdir(parents=True)
    job=output/'one.job.json';result=output/'one.job.result.json'
    write_json(job,dict(task=dict(episode_id='one')))
    shard=tmp_path/'quarantine/one-a1';shard.mkdir(parents=True)
    write_json(shard/'manifest.json',dict(error=None,episode_id='one'))
    for name in ('packed_narrow.npz','packed_wide.npz','evidence.jsonl.gz','telemetry.npz','candidates.json.gz'):(shard/name).write_bytes(b'retained')
    state=dict(active=dict(job='one',pid=123),completed={})
    process=SimpleNamespace(pid=123,cmdline=lambda:['python','-m','reports.automatic_corpus','--job',str(job)],is_running=lambda:False)
    monkeypatch.setattr(psutil,'Process',lambda pid:process)
    monkeypatch.setattr(Store,'rows',lambda self:[dict(episode='one',status='quarantined',path=str(shard))])
    runner.wait_worker(tmp_path,state,output/'state.json','one',job,result)
    saved=json.loads(result.read_text())
    assert saved['status']=='completed_capture_needs_postprocessing'
    assert saved['shard']==str(shard)
