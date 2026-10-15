"""Follow admitted data and resume the same isolated mixed CD210 run.

Run with the training Python environment and --corpus / --run-dir. The same
command resumes saved weights, AdamW state and replay counts after interruption.
No model loads before both objectives have admitted train and validation data.
Create <run-dir>/stop.request to stop at a saved checkpoint; remove that request
before restarting. Existing legacy trainers and collection processes are untouched.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import json
import math
import os
from pathlib import Path
import time
from uuid import uuid4

import torch

from lane_dataset.config import sha256_file, write_json
from lane_dataset.review_hold import ReviewHoldError, assert_checkpoint_not_held, assert_episode_not_held
from lane_training.cd210_model import create_model, load_checkpoint
from lane_training.checkpoint_store import checkpoint_path, immutable_copy, immutable_json, protected_cohorts
from lane_training.production import make_optimizer, run_lock
from lane_training.smoke import parameter_digest
from .mixed import (KINDS,OBJECTIVE_VERSION,SELECTION_METRIC,Recipe,EpisodeRef,fingerprint,
    discover,qualification,open_episode,backward_pair,evaluate,can_promote_mixed)
from .mixed_runner import source_bindings,_source_guard,_save,_lineage


SCHEMA = 'cd210_mixed_follow_run_v1'


@dataclass(frozen=True)
class FollowConfig:
    passes: int = 4
    cycle_steps: int = 1
    poll_seconds: float = 10.

    def __post_init__(self):
        if (type(self.passes) is not int or self.passes<1 or type(self.cycle_steps) is not int
                or not 1<=self.cycle_steps<=100 or not math.isfinite(self.poll_seconds)
                or not 0<self.poll_seconds<=60):
            raise ValueError('invalid follow replay/cycle/poll limits')


def _read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def _same(a,b,message):
    if fingerprint(a)!=fingerprint(b):raise ValueError(message)


def _member(directory,name):
    path=(directory/name).resolve()
    if directory not in path.parents:raise ValueError('mixed follow artifact escapes run directory')
    return path


def _binding(directory,path):
    return dict(path=path.relative_to(directory).as_posix(),sha256=sha256_file(path))


def _bound(directory,binding):
    path=_member(directory,binding['path'])
    if sha256_file(path)!=binding['sha256']:raise ValueError('mixed follow artifact integrity failed: '+str(path))
    return _read(path)


def epoch_rows(directory,binding,recipe):
    document=_bound(directory,binding)
    rows=[EpisodeRef(**value) for value in document['rows']]
    _same(qualification(rows,recipe),document['qualification'],'immutable epoch index changed')
    if not document['qualification']['ready']:raise ValueError('epoch has incomplete admitted cohorts')
    return rows


def _epoch(directory,rows,recipe):
    document=dict(rows=[asdict(r) for r in rows],qualification=qualification(rows,recipe))
    path=directory/'epochs'/(document['qualification']['index_sha256']+'.json')
    immutable_json(path,document)
    return _binding(directory,path)


def refresh_rows(previous,current,state):
    """Only append validation; never rewrite an accepted identity or trained row."""
    by_path={(r.kind,str(Path(r.path).resolve()).casefold()):r for r in current}
    if len(by_path)!=len(current):raise ValueError('duplicate current mixed manifest')
    for old in previous:
        new=by_path.get((old.kind,str(Path(old.path).resolve()).casefold()))
        if new is not None and asdict(new)!=asdict(old):
            raise ValueError('previously admitted mixed manifest changed')
        if new is None and (old.split=='validation' or state['counts'].get(old.sha256,0)>0):
            raise ValueError('protected validation or trained data disappeared from the mixed corpus')
    return sorted(current,key=lambda row:row.key)


def _guard_data(root,rows,state,selected=()):
    assert_checkpoint_not_held(root,state)
    selected={row.key for row in selected}
    for row in rows:
        if row.split!='validation' and row.key not in selected and not state['counts'].get(row.sha256,0):continue
        assert_episode_not_held(Path(row.path).parent/'manifest.json',corpus=root)
        if sha256_file(row.path)!=row.sha256:raise ValueError('admitted mixed manifest changed during update')


def _score(directory,model,reference,rows,root,recipe,state,progress):
    selected=sorted((r for r in rows if r.split=='validation'),key=lambda r:r.key)
    identity=dict(run_id=state['run_id'],parameter_sha256=parameter_digest(model),
        stock_parameter_sha256=state['stock_parameter_sha256'],recipe_sha256=fingerprint(recipe.document()),
        validation_index_sha256=fingerprint([asdict(row) for row in selected]))
    path=directory/'scores'/(fingerprint(identity)+'.json')
    _guard_data(root,rows,state)
    if path.exists():
        document=_read(path);_same(document['identity'],identity,'cached mixed score identity changed')
        # Reuse computation, while still checking retained admitted payloads.
        for row in selected:
            with open_episode(row,root):pass
    else:
        document=dict(identity=identity,result=evaluate(rows,root,model,reference,recipe=recipe,progress=progress))
        immutable_json(path,document)
    _guard_data(root,rows,state)
    return document['result'],_binding(directory,path)


def _protected(result,incumbent):
    values=protected_cohorts(result,incumbent)
    # An enlarged cohort may not hide regression on either earlier data type.
    for kind in KINDS:
        cohort=sorted(row['manifest_sha256'] for row in result['rows'] if row['kind']==kind)
        if cohort not in values:values.append(cohort)
    return values


def _score_receipt(directory,binding,state,rows,parameter_sha):
    document=_bound(directory,binding)
    expected=dict(run_id=state['run_id'],parameter_sha256=parameter_sha,
        stock_parameter_sha256=state['stock_parameter_sha256'],recipe_sha256=state['config']['recipe_sha256'],
        validation_index_sha256=fingerprint([asdict(row) for row in sorted(rows,key=lambda r:r.key) if row.split=='validation']))
    _same(document['identity'],expected,'saved comparison model/cohort identity changed')
    return document['result']


def publish_step(directory,pointer,state,rows,incumbent,*,publish_pointer=True):
    """Idempotently complete a committed step after any publication interruption."""
    result=_score_receipt(directory,state['score_binding'],state,rows,state['parameter_sha256'])
    _same(result,state['last_validation'],'saved step differs from its immutable heldout score')
    comparison=state.get('comparison_best')
    if incumbent is None:
        if comparison is not None:raise ValueError('step comparison lost its prior best checkpoint')
        compared=None
    else:
        if comparison is None or comparison['sha256']!=incumbent['sha256']:
            raise ValueError('step comparison does not follow the prior immutable best')
        score=_score_receipt(directory,comparison['score_binding'],state,rows,incumbent['parameter_sha256'])
        compared=dict(score,sha256=incumbent['sha256'],protected_cohorts=incumbent['protected_cohorts'])
    validation=dict(result,step=pointer['step'])
    immutable_json(directory/'validation'/f'{pointer["step"]:06d}.json',validation)
    if not can_promote_mixed(result,compared):return incumbent,False
    record=dict(validation,**pointer,kind='mixed_validated_candidate',epoch_binding=state['epoch_binding'],
        parameter_sha256=state['parameter_sha256'],previous_best_sha256=None if incumbent is None else incumbent['sha256'],
        protected_cohorts=_protected(result,incumbent))
    immutable_json(directory/'preserved'/'promotions'/f'{pointer["step"]:06d}-{pointer["sha256"]}.json',record)
    if publish_pointer:write_json(directory/'best.json',record)
    return record,True


def recover(directory,run,recipe):
    """Recover committed optimizer steps even if latest/best pointer writes failed."""
    documents=sorted((directory/'steps').glob('*.json'))
    if not documents:return None,None,None,None
    latest=None;best=None;states=[];winners=[];epochs={}
    for number,path in enumerate(documents):
        value=_read(path);state=value['state']
        pointer={key:value[key] for key in ('checkpoint','sha256','step')}
        if path.name!=f'{number:06d}.json' or pointer['step']!=number or state['step']!=number:
            raise ValueError('mixed committed step chain is missing or unordered')
        _same(state['config'],run,'mixed resume configuration changed')
        _same(state['source_hashes'],run['source_hashes'],'mixed resume source binding changed')
        if state.get('previous_checkpoint_sha256')!=(None if latest is None else latest['sha256']):
            raise ValueError('mixed checkpoint predecessor chain changed')
        checkpoint=checkpoint_path(directory,pointer)
        if not checkpoint.is_file():raise ValueError('committed mixed checkpoint is missing')
        key=state['epoch_binding']['sha256']
        if key not in epochs:epochs[key]=epoch_rows(directory,state['epoch_binding'],recipe)
        rows=epochs[key]
        candidate,promoted=publish_step(directory,pointer,state,rows,best,publish_pointer=False)
        if promoted:best=candidate;winners.append(best)
        latest=pointer;states.append(state)
    old_best=directory/'best.json'
    if old_best.exists():
        saved=_read(old_best)
        if not any(fingerprint(saved)==fingerprint(winner) for winner in winners):
            raise ValueError('best pointer is not a verified promotion in this mixed run')
    if best is not None and sha256_file(checkpoint_path(directory,best))!=best['sha256']:
        raise ValueError('prior mixed best integrity failed during resume')
    old_latest=directory/'latest.json'
    if old_latest.exists():
        saved=_read(old_latest)
        if not any(saved.get('sha256')==_read(path)['sha256'] and saved.get('step')==i for i,path in enumerate(documents)):
            raise ValueError('latest pointer is not a committed step in this mixed run')
    checkpoint=checkpoint_path(directory,latest)
    if sha256_file(checkpoint)!=latest['sha256']:raise ValueError('latest mixed checkpoint integrity failed')
    model,package=load_checkpoint(checkpoint)
    _same(package['training_state'],states[-1],'latest optimizer checkpoint and committed state disagree')
    if package['step']!=latest['step'] or package.get('optimizer') is None:
        raise ValueError('mixed resume checkpoint lacks its optimizer/step')
    if parameter_digest(model)!=states[-1]['parameter_sha256']:
        raise ValueError('mixed resumed parameter digest changed')
    model.eval();optimizer=make_optimizer(model,recipe);optimizer.load_state_dict(package['optimizer'])
    if best is not None:write_json(old_best,best)
    write_json(old_latest,latest)
    return model,optimizer,deepcopy(states[-1]),best


def _claim(directory,root,recipe,settings,seed_checkpoint,seed_sha256):
    path=directory/'run.json'
    if path.exists():
        run=_read(path)
        if run.get('schema')!=SCHEMA or Path(run['corpus']).resolve()!=root:
            raise ValueError('directory belongs to another training run/corpus')
        saved_recipe=Recipe(**{f.name:run['recipe'][f.name] for f in fields(Recipe)})
        saved_settings=FollowConfig(**run['follow'])
        if recipe is not None:_same(recipe.document(),run['recipe'],'mixed resume recipe changed')
        if settings is not None:_same(asdict(settings),run['follow'],'mixed resume follow settings changed')
        _same(saved_recipe.document(),run['recipe'],'mixed objective version changed')
        if seed_checkpoint is not None or seed_sha256 is not None:
            expected=dict(path=str(Path(seed_checkpoint).resolve()),sha256=seed_sha256) if seed_checkpoint is not None else None
            _same(expected,run['seed'],'mixed resume seed identity changed')
        _source_guard(run['source_hashes'])
        return run,saved_recipe,saved_settings
    if any(p.name!='trainer.lock' and not p.name.startswith('.record-') for p in directory.iterdir()):
        raise FileExistsError('follow run directory contains unrelated files')
    recipe,settings=recipe or Recipe(),settings or FollowConfig()
    if (seed_checkpoint is None)!=(seed_sha256 is None):raise ValueError('seed path and exact SHA-256 required together')
    seed=None
    if seed_checkpoint is not None:
        seed=dict(path=str(Path(seed_checkpoint).resolve()),sha256=seed_sha256)
        if sha256_file(seed['path'])!=seed_sha256:raise ValueError('mixed seed SHA-256 mismatch')
    run=dict(schema=SCHEMA,run_id=uuid4().hex,corpus=str(root),recipe=recipe.document(),
        recipe_sha256=fingerprint(recipe.document()),follow=asdict(settings),seed=seed,
        source_hashes=source_bindings(),fixed_snapshot=False,snapshot_policy='immutable_epoch_at_cycle_boundary',
        production_process_modified=False)
    immutable_json(path,run)
    if seed:immutable_copy(seed['path'],directory/'seed'/(seed_sha256+'.pt'),seed_sha256)
    return run,recipe,settings


def _pause(directory,seconds):
    deadline=time.monotonic()+seconds
    while not (directory/'stop.request').exists() and time.monotonic()<deadline:
        time.sleep(min(1.,max(0.,deadline-time.monotonic())))


def follow(corpus,directory,*,recipe=None,settings=None,seed_checkpoint=None,seed_sha256=None,
           max_steps=None,max_cycles=None,progress=None):
    """Wait, update at most cycle_steps per frozen epoch, refresh, and resume."""
    if any(value is not None and (type(value) is not int or value<1) for value in (max_steps,max_cycles)):
        raise ValueError('optional total-step/cycle bounds must be positive integers')
    root,directory=Path(corpus).resolve(),Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    with run_lock(directory):
        run,recipe,settings=_claim(directory,root,recipe,settings,seed_checkpoint,seed_sha256)
        torch.set_num_threads(recipe.cpu_threads)
        def emit(event,**values):
            row=dict(event=event,time=time.time(),pid=os.getpid(),run_id=run['run_id'],**values)
            with (directory/'events.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            write_json(directory/'status.json',row);print(json.dumps(row,allow_nan=False),flush=True)
        def advancing(stage,tick,total):
            emit('progress',stage=stage,tick=tick,total=total)
            if progress:progress(stage,tick,total)
        model=optimizer=state=reference=best=None
        previous=[];cycles=io_failures=0
        try:
            model,optimizer,state,best=recover(directory,run,recipe)
            if state is not None:
                assert_checkpoint_not_held(root,state)
                previous=epoch_rows(directory,state['epoch_binding'],recipe)
                emit('resumed',step=state['step'],best_step=None if best is None else best['step'])
            while True:
                if (directory/'stop.request').exists():
                    emit('stopped_at_checkpoint',step=None if state is None else state['step']);break
                if max_steps is not None and state is not None and state['step']>=max_steps:
                    emit('total_step_limit_reached',step=state['step']);break
                if max_cycles is not None and cycles>=max_cycles:break
                cycles+=1
                _source_guard(run['source_hashes'])
                if state is not None:assert_checkpoint_not_held(root,state)
                try:
                    rows,complete=discover(root,verify_payloads=False);io_failures=0
                except (OSError,json.JSONDecodeError) as error:
                    io_failures+=1
                    if io_failures>=3:raise
                    emit('waiting_for_atomic_index',error=str(error),attempt=io_failures)
                    _pause(directory,settings.poll_seconds);continue
                rows=sorted(rows,key=lambda row:row.key)
                qualified=qualification(rows,recipe)
                if not qualified['ready']:
                    if state is not None:
                        refresh_rows(previous,rows,state)
                    emit('waiting_for_admitted_mixed_cohorts',qualification=qualified)
                    _pause(directory,settings.poll_seconds);continue
                if state is not None:rows=refresh_rows(previous,rows,state)
                epoch=_epoch(directory,rows,recipe)
                if reference is None:
                    emit('loading_frozen_cd210_baseline')
                    reference=create_model().eval().requires_grad_(False)
                if state is None:
                    seed_state={}
                    if run['seed']:
                        seed=directory/'seed'/(run['seed']['sha256']+'.pt')
                        if not seed.exists():immutable_copy(run['seed']['path'],seed,run['seed']['sha256'])
                        if sha256_file(seed)!=run['seed']['sha256']:raise ValueError('retained seed integrity failed')
                        model,package=load_checkpoint(seed);seed_state=package.get('training_state') or {}
                    else:model=create_model()
                    model.eval();optimizer=make_optimizer(model,recipe)
                    known,counts=_lineage(seed_state,rows)
                    state=dict(run_id=run['run_id'],step=0,config=run,known=known,counts=counts,sample_counts={},
                        objective_version=OBJECTIVE_VERSION,selection_metric=SELECTION_METRIC,
                        initial_parameter_sha256=parameter_digest(model),parameter_sha256=parameter_digest(model),
                        stock_parameter_sha256=parameter_digest(reference),source_hashes=run['source_hashes'],
                        previous_checkpoint_sha256=None,epoch_binding=epoch,comparison_best=None)
                    result,binding=_score(directory,model,reference,rows,root,recipe,state,advancing)
                    state.update(last_validation=result,score_binding=binding)
                    _source_guard(run['source_hashes']);_guard_data(root,rows,state)
                    pointer=_save(directory,model,optimizer,state)
                    best,promoted=publish_step(directory,pointer,state,rows,None)
                    emit('initial_validation_committed',step=0,best_promoted=promoted)
                if parameter_digest(reference)!=state['stock_parameter_sha256']:
                    raise ValueError('immutable original CD210 baseline changed on resume')
                state['known'].update({row.key:row.lineage() for row in rows})
                state['epoch_binding']=epoch;previous=rows
                emit('epoch_ready',step=state['step'],epoch=epoch,cohorts=qualified['cohorts'])
                for _ in range(settings.cycle_steps):
                    if (directory/'stop.request').exists() or (max_steps is not None and state['step']>=max_steps):break
                    selected=[]
                    for kind in KINDS:
                        eligible=[r for r in rows if r.kind==kind and r.split=='train' and state['sample_counts'].get(r.key,0)<settings.passes]
                        if not eligible:break
                        selected.append(min(eligible,key=lambda r:(state['sample_counts'].get(r.key,0),r.key)))
                    if len(selected)!=2:
                        emit('waiting_for_new_accepted_training_data',step=state['step']);break
                    comparison=None
                    if best is not None:
                        if best['parameter_sha256']==parameter_digest(model):incumbent_model=model
                        else:
                            incumbent_path=checkpoint_path(directory,best)
                            if sha256_file(incumbent_path)!=best['sha256']:raise ValueError('prior mixed best integrity failed')
                            incumbent_model,_=load_checkpoint(incumbent_path);incumbent_model.eval()
                        _,binding=_score(directory,incumbent_model,reference,rows,root,recipe,state,advancing)
                        comparison=dict(sha256=best['sha256'],score_binding=binding)
                        if incumbent_model is not model:del incumbent_model
                    emit('mixed_backward',next_step=state['step']+1,episodes=[row.path for row in selected])
                    try:
                        with open_episode(selected[0],root) as old,open_episode(selected[1],root) as behavior:
                            report=backward_pair(old,behavior,model,reference,recipe=recipe,progress=advancing)
                        _source_guard(run['source_hashes']);_guard_data(root,rows,state,selected)
                    except ReviewHoldError:
                        model.zero_grad(set_to_none=True)
                        assert_checkpoint_not_held(root,state)
                        emit('uncommitted_data_hold_refresh',step=state['step']);break
                    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),recipe.gradient_clip,error_if_nonfinite=True))
                    optimizer.step()
                    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):raise ValueError('nonfinite optimizer update')
                    updated=parameter_digest(model)
                    if updated==report['student_parameter_sha256']:raise ValueError('optimizer did not change model parameters')
                    if parameter_digest(reference)!=state['stock_parameter_sha256']:raise ValueError('frozen original changed')
                    latest=_read(directory/'latest.json')
                    state.update(step=state['step']+1,previous_checkpoint_sha256=latest['sha256'],
                        parameter_sha256=updated,comparison_best=comparison)
                    for row in selected:
                        state['counts'][row.sha256]=state['counts'].get(row.sha256,0)+1
                        state['sample_counts'][row.key]=state['sample_counts'].get(row.key,0)+1
                    result,binding=_score(directory,model,reference,rows,root,recipe,state,advancing)
                    state.update(last_validation=result,score_binding=binding)
                    _source_guard(run['source_hashes']);_guard_data(root,rows,state,selected)
                    pointer=_save(directory,model,optimizer,state)
                    best,promoted=publish_step(directory,pointer,state,rows,best)
                    emit('mixed_step_committed',step=state['step'],checkpoint=pointer,best_promoted=promoted,
                        best_step=None if best is None else best['step'],gradient_norm_before_clip=norm,cohorts=result['cohorts'])
                if max_steps is not None and state['step']>=max_steps:continue
                _pause(directory,settings.poll_seconds)
            return dict(run_id=run['run_id'],step=None if state is None else state['step'],cycles=cycles,
                best=best,automatic_follow_enabled=True,driving_release_qualified=False)
        except BaseException as error:
            emit('failed_prior_checkpoints_preserved',error=type(error).__name__+': '+str(error))
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,default=Path(r'G:\lane-centering-dataset-v1'))
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--legacy-weight',type=float)
    parser.add_argument('--cpu-threads',type=int)
    parser.add_argument('--passes',type=int)
    parser.add_argument('--cycle-steps',type=int)
    parser.add_argument('--poll-seconds',type=float)
    parser.add_argument('--seed-checkpoint',type=Path)
    parser.add_argument('--seed-sha256')
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--max-cycles',type=int)
    args=parser.parse_args(argv)
    recipe=None if args.legacy_weight is None and args.cpu_threads is None else Recipe(
        legacy_weight=.5 if args.legacy_weight is None else args.legacy_weight,
        behavior_weight=.5 if args.legacy_weight is None else 1.-args.legacy_weight,
        cpu_threads=2 if args.cpu_threads is None else args.cpu_threads)
    settings=None if all(getattr(args,name) is None for name in ('passes','cycle_steps','poll_seconds')) else FollowConfig(
        passes=4 if args.passes is None else args.passes,cycle_steps=1 if args.cycle_steps is None else args.cycle_steps,
        poll_seconds=10. if args.poll_seconds is None else args.poll_seconds)
    follow(args.corpus,args.run_dir,recipe=recipe,settings=settings,seed_checkpoint=args.seed_checkpoint,
        seed_sha256=args.seed_sha256,max_steps=args.max_steps,max_cycles=args.max_cycles)
    return 0


if __name__=='__main__':raise SystemExit(main())
