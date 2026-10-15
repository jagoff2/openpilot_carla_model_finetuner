from contextlib import contextmanager
from dataclasses import asdict
from types import SimpleNamespace
import json

import psutil
import pytest

from lane_dataset.config import write_json
from lane_behavior import world_access as access
from reports import parallel_corpus as parallel


@pytest.fixture
def fake_processes(monkeypatch):
    clock = [1000.]
    processes = {101:(100., psutil.STATUS_RUNNING), 202:(200., psutil.STATUS_RUNNING),
        303:(300., psutil.STATUS_RUNNING)}
    def process(pid):
        if pid not in processes:
            raise psutil.NoSuchProcess(pid)
        created, status = processes[pid]
        return SimpleNamespace(create_time=lambda:created, is_running=lambda:True, status=lambda:status)
    monkeypatch.setattr(access.os, 'getpid', lambda:101)
    monkeypatch.setattr(access.time, 'time', lambda:clock[0])
    monkeypatch.setattr(access.psutil, 'Process', process)
    monkeypatch.setattr(access.psutil, 'pid_exists', lambda pid:pid in processes)
    monkeypatch.setattr(access.psutil, 'process_iter', lambda *args:[])
    return clock, processes


def test_active_lease_blocks_dispatch_without_touching_other_port(tmp_path, fake_processes):
    lease = access.acquire_reservation(tmp_path, 2400, duration_s=120.)
    assert lease.owner_pid == 101 and lease.owner_create_time == 100.
    assert access.reservation_status(tmp_path, 2400).status == 'active'
    assert not access.reservation_status(tmp_path, 2500).blocked
    with pytest.raises(access.PortReservedError, match='reserved'):
        with access.collection_dispatch(tmp_path, 2400):
            pytest.fail('reserved port dispatched')
    with access.collection_dispatch(tmp_path, 2500):
        pass
    assert access.release_reservation(tmp_path, lease)
    assert not access.release_reservation(tmp_path, lease)


@pytest.mark.parametrize('cause,status', [('expired','expired'), ('dead','dead_owner'), ('reused','pid_reused')])
def test_stale_lease_never_reserves_a_reused_pid_or_dead_process(tmp_path, fake_processes, cause, status):
    clock, processes = fake_processes
    lease = access.acquire_reservation(tmp_path, 2400, duration_s=10.)
    if cause == 'expired':
        clock[0] += 11.
    elif cause == 'dead':
        del processes[101]
    else:
        processes[101] = (999., psutil.STATUS_RUNNING)
    current = access.reservation_status(tmp_path, 2400)
    assert not current.blocked and current.status == status
    replacement = access.acquire_reservation(tmp_path, 2400, owner_pid=202)
    with pytest.raises(access.PortReservedError):
        access.release_reservation(tmp_path, lease)
    assert access.reservation_status(tmp_path, 2400).reservation.token == replacement.token


def test_expiry_renewal_is_bounded_and_old_token_cannot_remove_another_owner(tmp_path, fake_processes):
    clock, _ = fake_processes
    lease = access.acquire_reservation(tmp_path, 2400, duration_s=10.)
    clock[0] += 5.
    renewed = access.renew_reservation(tmp_path, lease, duration_s=3600.)
    assert renewed.token == lease.token and renewed.expires_at-clock[0] == 3600.
    with pytest.raises(ValueError, match='3600'):
        access.renew_reservation(tmp_path, renewed, duration_s=3601.)
    with pytest.raises(access.PortReservedError):
        access.acquire_reservation(tmp_path, 2400, owner_pid=202)
    clock[0] = renewed.expires_at+1.
    with pytest.raises(access.PortReservedError, match='expired'):
        access.renew_reservation(tmp_path, renewed)


@pytest.mark.parametrize('content', ['{', '{}', '[1,2,3]'])
def test_malformed_lease_fails_closed_for_only_its_port(tmp_path, fake_processes, content):
    path = tmp_path/'reports/worker-reservations/port-2400.json'
    path.parent.mkdir(parents=True)
    path.write_text(content)
    status = access.reservation_status(tmp_path, 2400)
    assert status.blocked and status.status == 'malformed' and status.reason
    assert not access.reservation_status(tmp_path, 2500).blocked
    with pytest.raises(access.PortReservedError):
        access.acquire_reservation(tmp_path, 2400)


def test_reservation_mutex_makes_final_dispatch_and_acquisition_atomic(tmp_path, fake_processes):
    with access.collection_dispatch(tmp_path, 2400):
        with pytest.raises(access.PortReservedError, match='mutex'):
            access.acquire_reservation(tmp_path, 2400)
    assert access.acquire_reservation(tmp_path, 2400).port == 2400


def running_collector(root):
    write_json(root/'reports/automatic/status.json', dict(status='running', pid=303,
        configured_workers=[dict(port=2400), dict(port=2500)]))


def test_own_lease_requires_supported_supervisor_and_drained_workers(tmp_path, fake_processes, monkeypatch):
    running_collector(tmp_path)
    access.acquire_reservation(tmp_path, 2400)
    with pytest.raises(RuntimeError, match='reservation support'):
        access.require_reserved_or_idle(tmp_path, 2400)
    write_json(tmp_path/'reports/automatic/state.json', dict(worker_reservation_protocol=dict(
        protocol=access.PROTOCOL, pid=303, create_time=300.)))
    result = access.require_reserved_or_idle(tmp_path, 2400)
    assert result['access'] == 'reserved' and result['world_lock_and_empty_actor_check_required']
    command = ['python', '-m', 'reports.automatic_corpus', '--job', 'x', '--root', str(tmp_path), '--port', '2400']
    worker = SimpleNamespace(info=dict(pid=404, cmdline=command))
    monkeypatch.setattr(access.psutil, 'process_iter', lambda *args:[worker])
    with pytest.raises(RuntimeError, match='draining'):
        access.require_reserved_or_idle(tmp_path, 2400)
    worker.info['cmdline'] = command+['--reprocess']
    assert access.require_reserved_or_idle(tmp_path, 2400)['access'] == 'reserved'


def test_other_owner_cannot_be_overridden_even_with_paused_collector(tmp_path, fake_processes):
    access.acquire_reservation(tmp_path, 2400, owner_pid=202)
    with pytest.raises(RuntimeError, match='unavailable'):
        access.require_reserved_or_idle(tmp_path, 2400, paused_supervisor_pid=303)
    with pytest.raises(RuntimeError, match='no owner override'):
        access.require_reserved_or_idle(tmp_path, 2400, owner_pid=202)


def test_old_idle_and_explicit_paused_permission_remain_available(tmp_path, fake_processes):
    assert access.require_reserved_or_idle(tmp_path, 2400)['access'] == 'idle_or_explicitly_paused'
    running_collector(tmp_path)
    with pytest.raises(RuntimeError, match='running automatic collector'):
        access.require_reserved_or_idle(tmp_path, 2400)
    _, processes = fake_processes
    processes[303] = (300., psutil.STATUS_STOPPED)
    result = access.require_reserved_or_idle(tmp_path, 2400, paused_supervisor_pid=303)
    assert result['access'] == 'idle_or_explicitly_paused'


def test_collector_protocol_identity_cannot_be_reused_after_pid_reuse(tmp_path, fake_processes):
    running_collector(tmp_path)
    write_json(tmp_path/'reports/automatic/state.json', dict(worker_reservation_protocol=dict(
        protocol=access.PROTOCOL, pid=303, create_time=299.)))
    access.acquire_reservation(tmp_path, 2400)
    with pytest.raises(RuntimeError, match='matching reservation support'):
        access.require_reserved_or_idle(tmp_path, 2400)


def test_reserved_recapture_keeps_result_until_atomic_dispatch_is_possible(tmp_path, fake_processes, monkeypatch):
    result_path = tmp_path/'one.job.result.json'
    result = dict(status='failed', capture_policy_version=8)
    write_json(result_path, result)
    worker = SimpleNamespace(id='one', port=2400, job_path=tmp_path/'one.job.json',
        result_path=result_path, result=lambda:result, job=dict(task={}))
    monkeypatch.setattr(parallel.corpus, 'needs_recovery_policy_retry', lambda _:True)
    lease = access.acquire_reservation(tmp_path, 2400)
    with pytest.raises(access.PortReservedError):
        parallel.finish(tmp_path, dict(completed={}), worker, tmp_path/'tests.xml')
    assert json.loads(result_path.read_text()) == result
    assert not result_path.with_name(result_path.stem+'.before-recovery-policy.json').exists()
    access.release_reservation(tmp_path, lease)
    monkeypatch.setattr(parallel, 'Worker', lambda *a, **k:'replacement')
    assert parallel.finish(tmp_path, dict(completed={}), worker, tmp_path/'tests.xml') == 'replacement'


def test_supervisor_adopts_current_job_publishes_it_and_keeps_other_gpu_collecting(tmp_path, fake_processes, monkeypatch):
    from lane_dataset.storage import Store
    output = tmp_path/'reports/automatic'
    output.mkdir(parents=True)
    jobs = [dict(episode_id=n, group_id=n, split='train', family='constant_radius') for n in ('a','b','c')]
    old = output/'a.job.json'
    write_json(old, dict(id='a', task=jobs[0]))
    plan = tmp_path/'plan.json'
    write_json(plan, [])
    lease = access.acquire_reservation(tmp_path, 2400)
    events = []
    class FakeWorker:
        def __init__(self, root, job_path, port, process=None, **kwargs):
            self.job = parallel.corpus.read(job_path)
            self.id, self.port, self.stage, self.polls = self.job['id'], port, 'capture', 0
            events.append(('attach' if process else 'start', self.id, port))
        def poll(self):
            self.polls += 1
            return self.polls >= 2
        def summary(self):
            return dict(job=self.id, port=self.port)
        def close(self):
            pass
    def finish(root, state, worker, tests):
        state['completed'][worker.id] = dict(status='accepted')
        events.append(('publish', worker.id, worker.port))
    monkeypatch.setattr(parallel, 'Worker', FakeWorker)
    monkeypatch.setattr(parallel, 'finish', finish)
    monkeypatch.setattr(parallel, 'discover_workers', lambda _:[(object(), old, 2400)])
    monkeypatch.setattr(parallel.corpus, 'prepare_output', lambda _:None)
    monkeypatch.setattr(Store, 'recover', lambda _:None)
    monkeypatch.setattr(parallel.corpus, 'candidate_queue', lambda *a:iter((t,True) for t in jobs))
    monkeypatch.setattr(parallel.corpus, 'candidate_screen', lambda *a:None)
    monkeypatch.setattr(parallel.corpus, 'snapshot', lambda *a:dict(accepted_total=0))
    monkeypatch.setattr(parallel.time, 'sleep', lambda _:None)
    parallel.run(tmp_path, plan, tmp_path/'tests.xml')
    assert events[:2] == [('attach','a',2400),('start','b',2500)]
    assert ('publish','a',2400) in events and ('start','c',2500) in events
    assert not any(event[0] == 'start' and event[2] == 2400 for event in events)
    assert access.reservation_status(tmp_path, 2400).reservation.token == lease.token
    final = parallel.corpus.read(output/'state.json')
    assert set(final['completed']) == {'a','b','c'}
    assert final['worker_reservation_protocol']['protocol'] == access.PROTOCOL
