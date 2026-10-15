from pathlib import Path
from types import SimpleNamespace

import pytest

from lane_dataset.config import write_json
from reports import parallel_corpus as parallel


def task(name, group=None):
    return dict(episode_id=name, group_id=group or name, split='train', family='constant_radius')


def test_all_supervisor_entry_points_use_both_gpu_scheduler(monkeypatch):
    calls = []
    monkeypatch.setattr(parallel, 'run', lambda *args: calls.append(args))
    parallel.corpus.run('root', 'plan', 'tests')
    assert calls == [('root', 'plan', 'tests')]


@pytest.mark.parametrize('resume', [False, True])
def test_two_ports_run_concurrently_and_failure_keeps_both_slot_queue_moving(tmp_path, monkeypatch, resume):
    from lane_dataset.storage import Store
    output = tmp_path/'reports/automatic'
    output.mkdir(parents=True)
    plan = tmp_path/'plan.json'
    write_json(plan, [])
    events = []
    jobs = [task('a'), task('b'), task('c')]
    old = output/'a.job.json'
    write_json(old, dict(id='a', task=jobs[0]))
    class FakeWorker:
        def __init__(self, root, job_path, port, process=None, **kwargs):
            self.job = parallel.corpus.read(job_path)
            self.id = self.job['id']; self.port = port; self.stage = 'capture'
            self.polls = 0
            events.append(('attach' if process else 'start', self.id, port))
        def poll(self):
            self.polls += 1
            return self.polls >= 2
        def summary(self):
            return dict(job=self.id, port=self.port, gpu=(self.port-2400)//100)
        def close(self): pass
    def finish(root, state, worker, tests):
        events.append(('finish', worker.id, worker.port))
        state['completed'][worker.id] = dict(status='failed')
    def snapshot(root, state):
        return dict(accepted_total=0, active_workers=list(state.get('workers', {}).values()))
    monkeypatch.setattr(parallel, 'Worker', FakeWorker)
    monkeypatch.setattr(parallel, 'finish', finish)
    monkeypatch.setattr(parallel, 'discover_workers', lambda root: [(object(), old, 2400)] if resume else [])
    monkeypatch.setattr(parallel.corpus, 'prepare_output', lambda root: None)
    monkeypatch.setattr(Store, 'recover', lambda self: None)
    monkeypatch.setattr(parallel.corpus, 'candidate_queue', lambda *args: iter((t, True) for t in jobs))
    monkeypatch.setattr(parallel.corpus, 'candidate_screen', lambda *args: None)
    monkeypatch.setattr(parallel.corpus, 'snapshot', snapshot)
    monkeypatch.setattr(parallel.time, 'sleep', lambda seconds: None)
    parallel.run(tmp_path, plan, tmp_path/'tests.xml')
    assert events[:2] == [('attach' if resume else 'start', 'a', 2400), ('start', 'b', 2500)]
    assert len([e for e in events if e[1] == 'a' and e[0] in ('start', 'attach')]) == 1
    assert ('start', 'c', 2400) in events
    state = parallel.corpus.read(output/'state.json')
    assert set(state['completed']) == {'a', 'b', 'c'}
    assert state['configured_workers'] == [dict(gpu=0, port=2400), dict(gpu=1, port=2500)]


def test_discovery_collapses_venv_wrapper_and_keeps_distinct_gpu_jobs(tmp_path, monkeypatch):
    def process(pid, parent, job, port):
        command = ['python', '-m', 'reports.automatic_corpus', '--root', str(tmp_path),
            '--job', str(tmp_path/job), '--port', str(port)]
        return SimpleNamespace(pid=pid, info=dict(pid=pid, ppid=parent, cmdline=command))
    processes = [process(1, 0, 'a.job.json', 2400), process(2, 1, 'a.job.json', 2400),
        process(3, 0, 'b.job.json', 2500), process(4, 3, 'b.job.json', 2500)]
    monkeypatch.setattr(parallel.psutil, 'process_iter', lambda *args: processes)
    found = parallel.discover_workers(tmp_path)
    assert [(p.pid, port) for p, _, port in found] == [(1, 2400), (3, 2500)]
    processes.append(process(5, 0, 'duplicate.job.json', 2500))
    with pytest.raises(RuntimeError, match='same simulator port'):
        parallel.discover_workers(tmp_path)


def test_same_geometry_split_variants_are_deferred_while_another_worker_owns_group():
    workers = {2400: SimpleNamespace(job=dict(task=task('validation', 'geometry-a')))}
    assert parallel.busy_group(task('train', 'geometry-a'), workers)
    assert not parallel.busy_group(task('train', 'geometry-b'), workers)


def test_reattached_worker_reports_current_warmup_without_replaying_old_failure(tmp_path,capsys):
    import json
    output=tmp_path/'reports/automatic';output.mkdir(parents=True)
    job=output/'one.job.json';write_json(job,{'id':'one','task':task('one')})
    rows=[{'event':'capture_start'},{'tick':1500,'ticks':1506},
          {'event':'job_complete','status':'failed'}, {'event':'capture_start'},
          {'phase':'rolling_initialization','frame':20,'speed':1.2,'ready':False}]
    (output/'one.log').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    worker=parallel.Worker(tmp_path,job,2400,process=SimpleNamespace(pid=123,is_running=lambda:True))
    assert not worker.poll()
    assert worker.latest['frame']==20 and 'tick' not in worker.latest
    assert worker.stage=='rolling_initialization'
    assert 'failed' not in capsys.readouterr().out
    worker.close()


def test_publication_rechecks_quota_after_other_worker_admission(tmp_path, monkeypatch):
    job = tmp_path/'one.job.json'
    worker = SimpleNamespace(id='one', port=2500, job_path=job, job=dict(task=task('one')),
        result_path=job.with_suffix('.result.json'), result=lambda: dict(status='measured_candidate_passed'))
    monkeypatch.setattr(parallel.corpus, 'needs_recovery_policy_retry', lambda result: False)
    monkeypatch.setattr(parallel.corpus, 'candidate_screen', lambda root, task: 'quota_already_filled')
    monkeypatch.setattr(parallel.corpus, 'publish_candidate', lambda *args: pytest.fail('must not overfill quota'))
    state = dict(completed={})
    assert parallel.finish(tmp_path, state, worker, tmp_path/'tests.xml') is None
    assert state['completed']['one']['errors'] == ['quota_already_filled']


def test_already_accepted_validation_geometry_is_not_recaptured_for_training(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel.corpus, 'accepted_manifests', lambda root: [])
    write_json(tmp_path/'reports/scenario-plan.automatic.json', dict(
        groups=[dict(group_id='geometry-a')], assignments=[dict(group_id='geometry-a', split='validation')]))
    candidate = dict(episode_id='train-a', group_id='geometry-a', split='train',
        family='constant_radius', direction='left', town='Town07')
    assert parallel.corpus.candidate_screen(tmp_path, candidate) == 'geometry_group_already_assigned_to_validation'
    assert parallel.corpus.candidate_screen(tmp_path, {**candidate, 'split':'validation'}) is None
