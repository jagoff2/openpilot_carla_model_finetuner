"""Resumable CD210 centerline fine-tuning concurrent with accepted-data collection."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback

import psutil
import torch

from lane_dataset.loader import SequenceEpisode
from lane_dataset.review_hold import (ReviewHoldError, active_holds,
    assert_episode_not_held, assert_checkpoint_not_held, episode_name)
from .adapters import model_centerline_loss
from .cd210 import ROLLOUT_PLAN, backward, student_rollout
from .cd210_model import create_model, export_onnx, load_checkpoint, save_checkpoint
from .smoke import parameter_digest
from .targets import episode_corridors
from .centerline import OBJECTIVE_VERSION
from .checkpoint_store import (checkpoint_path, immutable_json, preserve,
    protect_validation, protected_cohorts, SELECTION_METRIC, selection_report, lateral_score)
from lane_dataset.config import DEFAULT_ROOT, write_json
from uuid import uuid4


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    write_json(path, value)


@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'trainer.lock').open('a+b') as stream:
        stream.seek(0); stream.write(b'0'); stream.flush(); stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('another trainer owns this run directory') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt': msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(stream,fcntl.LOCK_UN)


@dataclass(frozen=True)
class TrainingConfig:
    corpus: str
    cpu_threads: int = 2
    passes: int = 4
    vision_lr: float = 1e-6
    policy_lr: float = 1e-5
    weight_decay: float = .01
    gradient_clip: float = 1.
    validation_interval: int = 10
    poll_seconds: float = 10.
    device: str = 'cpu'

    def __post_init__(self):
        if self.device != 'cpu':
            raise ValueError('this concurrent run reserves both GPUs for CARLA and trains on CPU')
        if min(self.cpu_threads,self.passes,self.validation_interval) < 1:
            raise ValueError('thread, pass and validation counts must be positive')
        values = (self.vision_lr,self.policy_lr,self.weight_decay,self.gradient_clip,self.poll_seconds)
        if not all(math.isfinite(x) and x >= 0 for x in values) or min(self.vision_lr,self.policy_lr,self.gradient_clip,self.poll_seconds) <= 0:
            raise ValueError('invalid training hyperparameters')


def discover(corpus, known=None):
    """Read only the published index; enforce immutable manifests and split isolation."""
    corpus = Path(corpus).resolve()
    holds = active_holds(corpus)  # Also fail closed before an index exists.
    final = corpus/'training-index.json'
    index_path = final if final.exists() else corpus/'training-index.partial.json'
    if not index_path.exists(): return [],False
    index = json.loads(index_path.read_text(encoding='utf-8'))
    known = known or {}
    rows, seen_paths, groups = [],set(),{}
    for relative in index['episodes']:
        path = (corpus/relative).resolve()
        if corpus/'accepted' not in path.parents or path in seen_paths:
            raise ValueError('training index must contain unique accepted manifest paths inside the corpus')
        assert_episode_not_held(path, corpus=corpus)
        seen_paths.add(path)
        raw = path.read_bytes(); manifest = json.loads(raw)
        split = manifest.get('split')
        group = manifest.get('geometry_group_id')
        if (manifest.get('status') != 'accepted' or not manifest.get('admission',{}).get('passed')
                or split not in ('train','validation','test') or not group
                or manifest.get('config_sha256') != index.get('config_sha256')):
            raise ValueError(f'invalid admitted manifest or corpus binding: {path}')
        if group in groups and groups[group] != split:
            raise ValueError('geometry group leaks between corpus splits')
        groups[group] = split
        identity = str(path.relative_to(corpus))
        row = dict(id=identity, path=str(path), sha256=hashlib.sha256(raw).hexdigest(), split=split,
            group=group,family=manifest['family'],labels=int(manifest['accepted_label_count']))
        if identity in known and any(known[identity][k] != row[k] for k in ('sha256','split','group')):
            raise ValueError(f'previously observed accepted manifest changed: {identity}')
        rows.append(row)
    missing = set(known)-{r['id'] for r in rows}
    if any(episode_name(identity).casefold() not in holds for identity in missing):
        raise ValueError('published accepted index lost previously observed episodes')
    return rows,bool(index.get('complete'))


def select_next(rows,counts,passes,failed=None):
    failed = failed or {}
    available = [r for r in rows if r['split']=='train' and counts.get(r['sha256'],0)<passes
                 and r['sha256'] not in failed]
    family_steps = Counter()
    for row in rows:
        if row['split']=='train': family_steps[row['family']] += counts.get(row['sha256'],0)
    return min(available,key=lambda r:(counts.get(r['sha256'],0),family_steps[r['family']],r['id'])) if available else None


def make_optimizer(model,cfg):
    return torch.optim.AdamW([
        dict(params=list(model.vision.parameters()),lr=cfg.vision_lr,name='vision'),
        dict(params=list(model.policy.parameters()),lr=cfg.policy_lr,name='policy')],weight_decay=cfg.weight_decay)


def commit(directory,model,optimizer,state):
    """Two checkpoint slots plus one atomic pointer: weights and sampler commit together."""
    directory = Path(directory)
    old_path = directory/'resume.json'
    old = json.loads(old_path.read_text()) if old_path.exists() else None
    slot = 1-int(old['slot']) if old else 0
    path = directory/f'checkpoint-{slot}.pt'
    sha = save_checkpoint(model,path,optimizer=optimizer,step=state['step'],training_state=state)
    pointer = dict(slot=slot,file=path.name,sha256=sha,step=state['step'])
    atomic_json(old_path,pointer)
    return pointer


def restore(directory,cfg):
    directory = Path(directory)
    pointer = json.loads((directory/'resume.json').read_text())
    path = directory/pointer['file']
    if path.name not in ('checkpoint-0.pt','checkpoint-1.pt') or path.parent.resolve()!=directory.resolve() or digest(path)!=pointer['sha256']:
        raise ValueError('committed training checkpoint failed verification')
    model,package = load_checkpoint(path)
    state = package['training_state']
    if not state or state['config'] != asdict(cfg) or state['step'] != package['step'] or state['step'] != pointer['step']:
        raise ValueError('resume configuration or sampler/optimizer step binding changed')
    optimizer = make_optimizer(model,cfg)
    optimizer.load_state_dict(package['optimizer'])
    return model,optimizer,state


def migrate_preservation(directory, cfg, *, lateral_selection=False):
    """Explicit source migration; archive both slots before rebinding code."""
    directory = Path(directory)
    model, optimizer, state = restore(directory, cfg)
    current = {p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    previous = state['source_hashes']
    changed = {name for name in previous.keys() | current.keys() if previous.get(name) != current.get(name)}
    if state.get('objective_version') != OBJECTIVE_VERSION:
        raise ValueError('training objective changed; automatic migration of the old run is forbidden')
    if not changed:
        return None
    allowed = {'production.py', 'checkpoint_store.py'} if lateral_selection else {
        'production.py', 'cd210_model.py', 'checkpoint_store.py'}
    if not changed <= allowed:
        raise ValueError(f'preservation migration refuses unrelated training changes: {sorted(changed)}')
    saved = []
    for source in sorted(directory.glob('checkpoint-[01].pt')):
        package = torch.load(source,map_location='cpu',weights_only=True)
        saved.append(preserve(directory,dict(file=source.name,sha256=digest(source),step=package['step']),
            kind='pre_preservation_migration'))
    best = directory/'best.json'
    if best.exists():
        record = json.loads(best.read_text())
        preserve(directory,record,validation=record,kind='previous_best')
    parameter_sha = parameter_digest(model)
    audit = dict(step=state['step'],previous_sources=previous,current_sources=current,
        preserved=saved,parameter_sha256=parameter_sha,optimizer_unchanged=True,
        purpose='lateral_only_selection' if lateral_selection else 'checkpoint_preservation')
    immutable_json(directory/'preserved/migrations'/(uuid4().hex+'.json'),audit)
    state['source_hashes'] = current
    state['selection_metric'] = SELECTION_METRIC
    state['last_validation'] = None
    commit(directory,model,optimizer,state)
    return audit


@torch.no_grad()
def evaluate_episode(episode,student,reference,progress=None):
    corridors = episode_corridors(episode)
    ticks = 0
    def evaluate(model,stage):
        nonlocal ticks
        ticks = 0
        def forward(inputs):
            nonlocal ticks
            with torch.no_grad(): result = model(inputs)
            ticks += 1
            if progress and ticks % 100 == 0: progress(stage,ticks,len(episode))
            return result.detach()
        return student_rollout(episode,forward,device='cpu')[0]
    base = evaluate(reference,'validation_reference')
    prediction = evaluate(student,'validation_student')
    if not bool(torch.isfinite(prediction).all()): raise ValueError('nonfinite validation prediction')
    loss,report = model_centerline_loss(prediction,base,corridors,adapter=ROLLOUT_PLAN)
    baseline,baseline_report = model_centerline_loss(base,base,corridors,adapter=ROLLOUT_PLAN)
    if not report['valid_labels']: raise ValueError('validation episode has no supported centerline labels')
    lateral = ROLLOUT_PLAN(prediction).xy[:,:,1]-ROLLOUT_PLAN(base).xy[:,:,1]
    return dict(**selection_report(loss,baseline,report,baseline_report),labels=report['valid_labels'],
        geometry=report,baseline_geometry=baseline_report,
        path_lateral_change_mean_m=float(lateral.abs().mean()),
        velocity_change_mean=float((prediction[:,:495].reshape(-1,33,15)[:,:,3:6]-base[:,:495].reshape(-1,33,15)[:,:,3:6]).abs().mean()),
        other_policy_outputs_change_mean=float((prediction[:,495:1000]-base[:,495:1000]).abs().mean()))


class Trainer:
    def __init__(self,cfg,directory):
        self.cfg,self.directory = cfg,Path(directory).resolve()
        self.directory.mkdir(parents=True,exist_ok=True)
        self.events = self.directory/'events.jsonl'
        self.active = None
        self.started = time.time()
        if (self.directory/'resume.json').exists():
            self.model,self.optimizer,self.state = restore(self.directory,cfg)
            assert_checkpoint_not_held(cfg.corpus, self.state)
        else:
            self.model = create_model().eval()
            self.optimizer = make_optimizer(self.model,cfg)
            self.state = dict(config=asdict(cfg),step=0,counts={},known={},failed={},last_validation=None,
                objective_version=OBJECTIVE_VERSION,
                selection_metric=SELECTION_METRIC,
                initial_parameter_sha256=parameter_digest(self.model),source_hashes={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')})
            commit(self.directory,self.model,self.optimizer,self.state)
        if self.state.get('objective_version') != OBJECTIVE_VERSION:
            raise ValueError('training objective changed; preserve this run and explicitly qualify a new run before resuming')
        current_sources = {p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
        if self.state['source_hashes'] != current_sources:
            raise ValueError('training implementation changed since checkpoint; preserve run identity before migration')
        self.reference = create_model().eval().requires_grad_(False)
        self.reference_digest = parameter_digest(self.reference)
        if self.reference_digest != self.state['initial_parameter_sha256']:
            raise ValueError('frozen CD210 baseline differs from the original training run')
        self.log('started',config=asdict(cfg),resumed_step=self.state['step'])

    def log(self,event,**fields):
        row = dict(time=time.time(),event=event,step=self.state['step'])
        row.update(fields)
        with self.events.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
        print(json.dumps(row,allow_nan=False),flush=True)

    def status(self,phase,**fields):
        process = psutil.Process()
        collector_path = Path(self.cfg.corpus)/'reports/automatic/status.json'
        try:
            collection = json.loads(collector_path.read_text())
            collection = {k:collection[k] for k in ('accepted_total','pid','active_workers','updated')}
        except (OSError,ValueError,KeyError): collection = None
        atomic_json(self.directory/'status.json',dict(updated=time.time(),pid=os.getpid(),phase=phase,
            objective_version=OBJECTIVE_VERSION,
            selection_metric=SELECTION_METRIC,
            step=self.state['step'],optimizer_steps=self.state['step'],device='cpu',cpu_threads=self.cfg.cpu_threads,
            rss_bytes=process.memory_info().rss,cpu_seconds=sum(process.cpu_times()[:2]),
            uptime_seconds=time.time()-self.started,active=self.active,training_config=asdict(self.cfg),
            trained_episodes=len(self.state['counts']),last_validation=self.state['last_validation'],
            failed_episodes=self.state['failed'],collection=collection,**fields))

    def progress(self,stage,tick,total):
        self.status(stage,tick=tick,total=total)
        if tick % 300 == 0: self.log('progress',stage=stage,tick=tick,total=total,episode=self.active)

    def train_one(self,row):
        if row['split']!='train': raise ValueError('only the training split can update weights')
        assert_checkpoint_not_held(self.cfg.corpus, self.state)
        assert_episode_not_held(row['path'], corpus=self.cfg.corpus)
        self.active = row['id'];self.status('training');start=time.perf_counter()
        before = parameter_digest(self.model)
        with SequenceEpisode(row['path']) as episode:
            if episode.manifest['split']!='train' or digest(episode.path)!=row['sha256']:
                raise ValueError('training manifest identity changed before loading')
            prediction,report = backward(episode,self.model,self.reference,progress=self.progress)
        if not bool(torch.isfinite(prediction).all()) or not math.isfinite(report['loss']):
            raise ValueError('nonfinite training prediction/loss')
        gradients = {}
        for role in ('vision','policy'):
            gs = [p.grad for p in getattr(self.model,role).parameters() if p.grad is not None]
            if not gs or not all(bool(torch.isfinite(g).all()) for g in gs): raise ValueError(f'invalid {role} gradients')
            gradients[role] = sum(int(g.count_nonzero()) for g in gs)
            if not gradients[role]: raise ValueError(f'no {role} gradient reached the actual checkpoint')
        norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.cfg.gradient_clip,error_if_nonfinite=True))
        # A review hold may arrive while recurrence/backward is running.
        assert_checkpoint_not_held(self.cfg.corpus, self.state)
        assert_episode_not_held(row['path'], corpus=self.cfg.corpus)
        self.optimizer.step()
        if not all(bool(torch.isfinite(p).all()) for p in self.model.parameters()): raise ValueError('nonfinite updated weights')
        after = parameter_digest(self.model)
        if after == before: raise ValueError('optimizer failed to change model parameters')
        if parameter_digest(self.reference)!=self.reference_digest: raise ValueError('frozen original checkpoint changed')
        self.state['step'] += 1
        self.state['counts'][row['sha256']] = self.state['counts'].get(row['sha256'],0)+1
        pointer = commit(self.directory,self.model,self.optimizer,self.state)
        metrics = dict(episode=row['id'],manifest_sha256=row['sha256'],pass_number=self.state['counts'][row['sha256']],
            loss_before_update=report['loss'],valid_labels=report['valid_labels'],backward_calls=report['backward_calls'],
            selection_metric=SELECTION_METRIC,
            lateral_score_before_update=sum(lateral_score(r) for r in report['label_diagnostics'])/report['valid_labels'],
            progress_guard_before_update=sum(r['progress_loss'] for r in report['label_diagnostics'])/report['valid_labels'],
            nonzero_gradients=gradients,gradient_norm_before_clip=norm,parameter_sha256_before=before,
            parameter_sha256_after=after,checkpoint=pointer,elapsed_seconds=time.perf_counter()-start)
        self.log('optimizer_step_committed',**metrics)
        self.status('optimizer_step_committed',last_step=metrics)
        self.active = None
        return metrics

    def score(self, model, selected):
        cohort = hashlib.sha256(''.join(sorted(r['sha256'] for r in selected)).encode()).hexdigest()
        results = []
        for row in selected:
            self.active = row['id']; self.status('validation')
            with SequenceEpisode(row['path']) as episode:
                if episode.manifest['split']!='validation' or digest(episode.path)!=row['sha256']:
                    raise ValueError('validation manifest identity mismatch')
                result = evaluate_episode(episode,model,self.reference,self.progress)
            results.append(dict(episode=row['id'],manifest_sha256=row['sha256'],**result))
        labels = sum(r['labels'] for r in results)
        result = dict(step=self.state['step'],cohort=cohort,episodes=len(results),labels=labels,
            objective_version=OBJECTIVE_VERSION,
            selection_metric=SELECTION_METRIC,
            loss=sum(r['loss']*r['labels'] for r in results)/labels,
            baseline_loss=sum(r['baseline_loss']*r['labels'] for r in results)/labels,
            optimization_loss=sum(r['optimization_loss']*r['labels'] for r in results)/labels,
            baseline_optimization_loss=sum(r['baseline_optimization_loss']*r['labels'] for r in results)/labels,
            progress_guard_loss=sum(r['progress_guard_loss']*r['labels'] for r in results)/labels,
            baseline_progress_guard_loss=sum(r['baseline_progress_guard_loss']*r['labels'] for r in results)/labels,
            test_split_used=False,rows=results)
        return result

    def validate(self,rows):
        assert_checkpoint_not_held(self.cfg.corpus, self.state)
        selected = [r for r in rows if r['split']=='validation']
        if not selected: return
        result = self.score(self.model, selected)
        best_path = self.directory/'best.json'
        incumbent = None
        if best_path.exists():
            best = json.loads(best_path.read_text())
            preserve(self.directory, best, validation=best, kind='previous_best')
            if (best['cohort'] == result['cohort'] and best.get('rows')
                    and best.get('selection_metric') == SELECTION_METRIC):
                incumbent = best
            else:
                model, _ = load_checkpoint(checkpoint_path(self.directory, best))
                incumbent = self.score(model, selected)
                incumbent['sha256'] = best['sha256']
                prior = best
                if not best.get('rows'):
                    prior = json.loads((self.directory/f'validation-step-{best["step"]:06d}.json').read_text())
                    if prior['cohort'] != best['cohort']:
                        raise ValueError('legacy best validation cohort changed')
                incumbent['protected_cohorts'] = best.get('protected_cohorts') or protected_cohorts(prior)
                del model
        assert_checkpoint_not_held(self.cfg.corpus, self.state)
        for row in selected:
            assert_episode_not_held(row['path'], corpus=self.cfg.corpus)
        immutable_json(self.directory/'validation'/f'step-{self.state["step"]:06d}-{uuid4().hex}.json',result)
        self.state['last_validation'] = {k:v for k,v in result.items() if k!='rows'}
        pointer = commit(self.directory,self.model,self.optimizer,self.state)
        promoted = protect_validation(self.directory, pointer, result, incumbent)
        self.log('validation_completed',**self.state['last_validation'],best_promoted=promoted)
        self.active = None

    def run(self,*,max_steps=None):
        while True:
            if (self.directory/'stop.request').exists():
                self.status('stopped_at_checkpoint'); return
            if max_steps is not None and self.state['step']>=max_steps:
                self.status('step_limit_reached'); return
            assert_checkpoint_not_held(self.cfg.corpus, self.state)
            try:
                rows,complete = discover(self.cfg.corpus,self.state['known'])
            except (OSError,json.JSONDecodeError) as error:
                self.status('waiting_for_index',error=str(error));time.sleep(self.cfg.poll_seconds);continue
            self.state['known'].update({r['id']:r for r in rows})
            row = select_next(rows,self.state['counts'],self.cfg.passes,self.state['failed'])
            last = self.state['last_validation']
            cohort = hashlib.sha256(''.join(sorted(r['sha256'] for r in rows if r['split']=='validation')).encode()).hexdigest()
            validation_due = (last is None or last.get('selection_metric') != SELECTION_METRIC
                or (last['cohort']!=cohort) or self.state['step']-last['step']>=self.cfg.validation_interval
                or (self.state['step']==1 and last['step']==0) or (row is None and last['step']!=self.state['step']))
            if validation_due and any(r['split']=='validation' for r in rows):
                self.validate(rows)
            if row is None:
                if complete and not self.state['failed']:
                    self.status('exporting_final_candidate')
                    from lane_release.export import prepare
                    releases = self.directory/'final-exports'
                    output = prepare(self.directory, releases)
                    atomic_json(self.directory/'completion.json',dict(step=self.state['step'],exports=str(output),
                        final_validation=self.state['last_validation'],driving_release_qualified=False))
                    self.status('training_complete');return
                self.active = None; self.status('waiting_for_new_accepted_training_data')
                time.sleep(self.cfg.poll_seconds);continue
            try:
                self.train_one(row)
            except ReviewHoldError as error:
                self.log('review_hold', episode=row['id'], error=str(error))
                self.status('blocked_review_hold', error=str(error))
                raise
            except Exception as error:
                self.log('episode_update_failed',episode=row['id'],error=str(error),traceback=traceback.format_exc())
                self.model,self.optimizer,self.state = restore(self.directory,self.cfg)
                self.state['failed'][row['sha256']] = dict(episode=row['id'],error=str(error))
                self.state['known'].update({r['id']:r for r in rows})
                commit(self.directory,self.model,self.optimizer,self.state)
                self.active = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--run-dir',type=Path,default=Path(__file__).resolve().parents[1]/'runs/cd210-centerline-production')
    parser.add_argument('--cpu-threads',type=int,default=2)
    parser.add_argument('--passes',type=int,default=4)
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--migrate-preservation',action='store_true',
        help='archive current checkpoints and migrate only persistence/export source bindings')
    parser.add_argument('--migrate-lateral-selection',action='store_true',
        help='preserve checkpoints and switch validation/promotion to lateral quality only')
    args = parser.parse_args()
    cfg = TrainingConfig(corpus=str(args.corpus.resolve()),cpu_threads=args.cpu_threads,passes=args.passes)
    torch.set_num_threads(cfg.cpu_threads); torch.set_num_interop_threads(1)
    if os.name=='nt': psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    with run_lock(args.run_dir):
        trainer = None
        try:
            if args.migrate_preservation or args.migrate_lateral_selection:
                migrate_preservation(args.run_dir,cfg,lateral_selection=args.migrate_lateral_selection)
            trainer = Trainer(cfg,args.run_dir)
            trainer.run(max_steps=args.max_steps)
        except BaseException as error:
            if trainer is not None:
                trainer.status('interrupted' if isinstance(error,KeyboardInterrupt) else 'error',error=str(error))
            raise


if __name__=='__main__':
    main()
