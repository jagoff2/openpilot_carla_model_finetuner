"""Two GPU capture workers, with one owner of admission and the corpus ledger.

The supervisor survives/rejoins existing capture processes during restarts.
Both GPU slots are reserved for collection; training is not scheduled here.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import psutil

from lane_dataset.config import Config, PROJECT, write_json
from lane_dataset.simulator import world_owner
from reports import automatic_corpus as corpus
from reports.collection_failures import (retry_after_policy_change, remember_failure,
    cached_failure, archive_previous_result)
from lane_dataset.capture_policy import CAPTURE_POLICY_VERSION


def discover_workers(root):
    """Collapse the Windows venv wrapper and interpreter to one actual job."""
    matches = []
    for process in psutil.process_iter(['pid', 'ppid', 'cmdline']):
        command = process.info.get('cmdline') or []
        if ('reports.automatic_corpus' not in command or '--job' not in command
                or '--root' not in command or '--port' not in command):
            continue
        try:
            if Path(command[command.index('--root')+1]).resolve() != root.resolve():
                continue
            job = Path(command[command.index('--job')+1]).resolve()
            port = int(command[command.index('--port')+1])
        except (ValueError, IndexError):
            continue
        matches.append((process, job, port))
    identities = {p.pid for p, _, _ in matches}
    roots = [(p, job, port) for p, job, port in matches if p.info['ppid'] not in identities]
    if len({port for _, _, port in roots}) != len(roots):
        raise RuntimeError('multiple collection jobs already own the same simulator port')
    if len({job for _, job, _ in roots}) != len(roots):
        raise RuntimeError('duplicate existing collection job')
    return roots


class Worker:
    def __init__(self, root, job_path, port, *, process=None, reprocess=False):
        self.root, self.job_path, self.port = root, job_path, port
        self.job = corpus.read(job_path)
        self.id = self.job['id']
        self.result_path = job_path.with_suffix('.result.json')
        self.log_path = root/'reports/automatic'/(self.id+'.log')
        self.child = None
        offset=0
        if process is None:
            offset=self.log_path.stat().st_size if self.log_path.exists() else 0
            command = [sys.executable, '-u', '-m', 'reports.automatic_corpus',
                '--root', str(root), '--job', str(job_path), '--port', str(port)]
            if reprocess:
                command.append('--reprocess')
            with self.log_path.open('a', encoding='utf-8', buffering=1) as log:
                self.child = subprocess.Popen(command, cwd=PROJECT, stdout=log,
                    stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            process = psutil.Process(self.child.pid)
        elif self.log_path.exists():
            # Reattach only the latest attempt. Earlier failed runs remain in
            # the log, but must not appear as current progress or new errors.
            with self.log_path.open('rb') as stream:
                while True:
                    start=stream.tell();line=stream.readline()
                    if not line:break
                    try:event=json.loads(line).get('event')
                    except (ValueError,AttributeError):continue
                    if event in ('capture_start','reprocessing_preserved_capture'):offset=start
        self.process = process
        self.log_path.touch(exist_ok=True)
        self.reader = self.log_path.open(encoding='utf-8', errors='replace')
        self.reader.seek(offset)
        self.latest = None
        self.stage = 'capture_and_label_qualification'
        corpus.log('collection_worker_attached' if self.child is None else 'collection_worker_started',
            job=self.id, gpu=(port-2400)//100, port=port, pid=process.pid)

    def poll(self):
        for line in self.reader:
            print(f'[GPU {(self.port-2400)//100}] {line}', end='', flush=True)
            try:
                value = json.loads(line)
                if value.get('event')=='capture_start':
                    self.latest=None;self.stage='capture_and_label_qualification'
                if value.get('phase')=='rolling_initialization':
                    self.latest={k:value[k] for k in ('phase','frame','speed','ready','steady_intervals','reasons') if k in value}
                    self.stage='rolling_initialization'
                if 'tick' in value:
                    self.latest = {k: value[k] for k in ('tick', 'ticks', 'route_s', 'speed', 'reasons') if k in value}
                    self.stage='capture_and_label_qualification'
                if value.get('event') == 'label_build':
                    self.stage = 'label_qualification'
            except (ValueError, TypeError):
                pass
        return self.child.poll() is not None if self.child else not self.process.is_running()

    def summary(self):
        return dict(job=self.id, pid=self.process.pid, gpu=(self.port-2400)//100,
            port=self.port, log=str(self.log_path), stage=self.stage, latest=self.latest)

    def result(self):
        if not self.result_path.exists():
            from lane_dataset.storage import Store
            task_path = self.job_path.with_suffix('.capture-task.json')
            episode = (corpus.read(task_path) if task_path.exists() else self.job['task'])['episode_id']
            saved = None
            for row in reversed(Store(self.root).rows()):
                if row['episode'] != episode or row['status'] != 'quarantined':
                    continue
                shard = Path(row['path'])
                if not (shard/'manifest.json').exists():
                    continue
                captured = corpus.read(shard/'manifest.json')
                telemetry='partial-telemetry.npz' if captured.get('error') else 'telemetry.npz'
                if all((shard/name).is_file() for name in (
                        'packed_narrow.npz', 'packed_wide.npz', 'evidence.jsonl.gz', telemetry, 'candidates.json.gz')):
                    saved = dict(job=self.id, status='completed_capture_needs_postprocessing', shard=str(shard))
                    break
            write_json(self.result_path, saved or dict(job=self.id, status='worker_process_failed',
                errors=[f'worker_exit:{self.child.returncode if self.child else "orphan_exited"}']))
        return corpus.read(self.result_path)

    def close(self):
        self.reader.close()


def busy_group(task, workers):
    # Do not race two split assignments for the same not-yet-admitted geometry.
    return any(w.job['task']['group_id'] == task['group_id'] for w in workers.values())


def save(root, state, workers):
    state['workers'] = {str(port): worker.summary() for port, worker in workers.items()}
    state['active'] = next(iter(state['workers'].values()), None)
    state['configured_workers'] = [dict(gpu=gpu, port=port) for gpu, port, _ in Config().workers]
    write_json(root/'reports/automatic/state.json', state)
    return corpus.snapshot(root, state)


def finish(root, state, worker, test_report):
    result = worker.result()
    retry_marker = worker.result_path.with_name(worker.result_path.stem+'.before-recovery-policy.json')
    if result['status'] == 'completed_capture_needs_postprocessing':
        return Worker(root, worker.job_path, worker.port, reprocess=True)
    if corpus.needs_recovery_policy_retry(result) and not retry_marker.exists():
        shutil.copy2(worker.result_path, retry_marker)
        worker.result_path.unlink()
        return Worker(root, worker.job_path, worker.port)
    try:
        if result['status'] == 'measured_candidate_passed':
            effective = worker.job_path.with_suffix('.capture-task.json')
            task = corpus.read(effective) if effective.exists() else worker.job['task']
            # A different slot may have filled a quota or published overlapping
            # geometry since dispatch. Recheck under this sole publication owner.
            skip = corpus.candidate_screen(root, task)
            if skip:
                raise ValueError(skip)
            destination = corpus.publish_candidate(root, result, task, test_report)
            result = dict(status='accepted', path=str(destination), labels=20)
    except Exception as error:
        if result.get('shard'):
            shard = Path(result['shard'])
            if shard.parent == root/'quarantine' and shard.exists():
                corpus.retain_rejection_diagnostics(root, {**result, 'errors': [str(error)]})
        result = dict(status='rejected', errors=[str(error)], traceback=traceback.format_exc())
    state['completed'][worker.id] = result
    remember_failure(state,worker.job['task'],result)
    state['last_result'] = {**result, 'job': worker.id}
    corpus.log('collection_job_finished', gpu=(worker.port-2400)//100, job=worker.id,
        status=result['status'], errors=result.get('errors', []))
    return None


def run(root, plan_path, test_report):
    from lane_dataset.storage import Store
    from lane_dataset.cli import validate
    output = root/'reports/automatic'
    output.mkdir(parents=True, exist_ok=True)
    ports = [port for _, port, _ in Config().workers]
    if len(ports) != 2:
        raise ValueError('this supervisor requires the two approved collection GPUs')
    with world_owner('automatic-corpus-supervisor', root):
        corpus.prepare_output(root)
        Store(root).recover()
        state_path = output/'state.json'
        state = corpus.read(state_path) if state_path.exists() else dict(completed={}, plan=str(plan_path))
        state['status'] = 'running'
        workers = {}
        # Recover every live child before selecting any new job, including jobs
        # whose prior parent's last status write had not completed.
        for process, job_path, port in discover_workers(root):
            if port not in ports or job_path.parent != output.resolve():
                raise ValueError('surviving collection worker has unexpected port or job directory')
            workers[port] = Worker(root, job_path, port, process=process)
        queue = iter(corpus.candidate_queue(corpus.read(plan_path), root))
        pending = []
        exhausted = False
        save(root, state, workers)
        while True:
            # Refill both slots before doing slow independent publication work.
            for port in ports:
                if port in workers:
                    continue
                for _ in range(100):
                    ready = next((i for i, (t, _) in enumerate(pending) if not busy_group(t, workers)), None)
                    if ready is not None:
                        original, prepared = pending.pop(ready)
                    elif not exhausted:
                        try:
                            original, prepared = next(queue)
                        except StopIteration:
                            exhausted = True
                            break
                    else:
                        break
                    job_id = original['episode_id']
                    if any(w.id == job_id for w in workers.values()):
                        continue
                    if job_id in state['completed']:
                        old=state['completed'][job_id]
                        if not retry_after_policy_change(old):continue
                        state.setdefault('previous_results',{}).setdefault(job_id,[]).append(old)
                        archive_previous_result(output/(job_id+'.job.result.json'))
                        del state['completed'][job_id]
                    if busy_group(original, workers):
                        pending.append((original, prepared))
                        continue
                    try:
                        task = copy.deepcopy(original) if prepared else corpus.normalize_task(original)
                        cached=cached_failure(state,task)
                        if cached:
                            state['completed'][job_id]=dict(status='geometry_failure_cached',
                                capture_policy_version=CAPTURE_POLICY_VERSION,**cached)
                            continue
                        capture_task = output/(job_id+'.job.capture-task.json')
                        actual = task
                        published = root/'accepted'/actual['episode_id']
                        if published.exists():
                            state['completed'][job_id] = dict(status='accepted', path=str(published), labels=20,
                                resumed_after_publication=True)
                            continue
                        skip = corpus.candidate_screen(root, actual)
                        if skip:
                            state['completed'][job_id] = dict(status=skip)
                            continue
                        job_path = output/(job_id+'.job.json')
                        write_json(job_path, dict(id=job_id, task=task))
                        result_path = job_path.with_suffix('.result.json')
                        if result_path.exists():
                            result = corpus.read(result_path)
                            # Recover unfinished admission without recapturing.
                            if result['status'] in ('completed_capture_needs_postprocessing', 'measured_candidate_passed'):
                                workers[port] = Worker(root, job_path, port, reprocess=True)
                            elif corpus.needs_recovery_policy_retry(result):
                                marker = result_path.with_name(result_path.stem+'.before-recovery-policy.json')
                                if marker.exists():
                                    state['completed'][job_id] = result
                                    continue
                                shutil.copy2(result_path, marker)
                                result_path.unlink()
                                workers[port] = Worker(root, job_path, port)
                            else:
                                state['completed'][job_id] = result
                                continue
                        else:
                            workers[port] = Worker(root, job_path, port)
                        save(root, state, workers)
                        break
                    except Exception as error:
                        state['completed'][job_id] = dict(status='rejected', errors=[str(error)], traceback=traceback.format_exc())
                        corpus.log('candidate_rejected', job=job_id, error=str(error))
            for port, worker in list(workers.items()):
                if not worker.poll():
                    continue
                worker.stage = 'independent_admission_and_publication'
                save(root, state, workers)
                replacement = finish(root, state, worker, test_report)
                worker.close()
                if replacement:
                    workers[port] = replacement
                else:
                    del workers[port]
                save(root, state, workers)
            status = save(root, state, workers)
            if status['accepted_total'] == 5000 and not workers:
                validation = validate(root)
                if validation['status'] == 'release_ready':
                    index = corpus.read(root/'training-index.partial.json')
                    index['complete'] = True
                    write_json(root/'training-index.json', index)
                    state['status'] = 'complete'
                    break
                state['last_result'] = validation
            if exhausted and not pending and not workers:
                state['status'] = 'candidate_pool_exhausted_with_reported_deficits'
                break
            time.sleep(2)
        corpus.log('corpus_status', **save(root, state, workers))
