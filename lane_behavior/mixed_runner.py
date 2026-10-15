"""Bounded, isolated mixed CD210 training. Defaults to read-only qualification.

Example qualification:
  python -m lane_behavior.mixed_runner --corpus G:\\lane-centering-dataset-v1
Training additionally requires --steps N and an unused --run-dir. This extension
does not resume or alter an existing production run, export, or device model.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time
from uuid import uuid4

import torch

from lane_dataset.config import sha256_file, write_json
from lane_dataset.review_hold import assert_checkpoint_not_held, assert_episode_not_held
from lane_training.cd210_model import create_model, load_checkpoint, save_checkpoint
from lane_training.checkpoint_store import immutable_copy, immutable_json, preserve, protected_cohorts
from lane_training.production import make_optimizer, run_lock
from lane_training.smoke import parameter_digest
from .mixed import (OBJECTIVE_VERSION, SELECTION_METRIC, KINDS, Recipe, fingerprint,
    discover, qualification, open_episode, backward_pair, evaluate, can_promote_mixed)


def source_bindings():
    project = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(project)):sha256_file(p) for package in ('lane_dataset','lane_training','lane_behavior')
            for p in sorted((project/package).glob('*.py'))}


def _source_guard(expected):
    if source_bindings() != expected:
        raise ValueError('mixed training implementation changed; checkpoint preserved and run stopped')


def _save(directory, model, optimizer, state):
    """Save every step append-only; no rotating slot may destroy a candidate."""
    temporary = directory/('.checkpoint-'+uuid4().hex+'.pt')
    try:
        digest = save_checkpoint(model,temporary,optimizer=optimizer,step=state['step'],training_state=state)
        path = directory/'preserved'/'checkpoints'/(digest+'.pt')
        immutable_copy(temporary,path,digest)
    finally:
        temporary.unlink(missing_ok=True)
        temporary.with_suffix('.pt.partial').unlink(missing_ok=True)
    pointer = dict(checkpoint=str(path.relative_to(directory)),sha256=digest,step=state['step'])
    immutable_json(directory/'steps'/f'{state["step"]:06d}.json',dict(**pointer,state=state))
    # This pointer only locates the latest saved step. It never designates best.
    write_json(directory/'latest.json',pointer)
    return pointer


def _publish_validation(directory, pointer, result, incumbent):
    result = dict(result,step=pointer['step'])
    immutable_json(directory/'validation'/f'{pointer["step"]:06d}.json',result)
    # Every checkpoint already exists immutably, including worse-than-stock
    # candidates. A best pointer can move only under the mixed cohort rules.
    record = preserve(directory,pointer,validation=result,kind='mixed_validated_candidate')
    promoted = can_promote_mixed(result,incumbent)
    if promoted:
        record['protected_cohorts'] = protected_cohorts(result,incumbent)
        immutable_json(directory/'preserved'/'promotions'/f'{pointer["step"]:06d}-{pointer["sha256"]}.json',record)
        write_json(directory/'best.json',record)
        return record,True
    return incumbent,False


def _lineage(seed_state, rows):
    known = dict(seed_state.get('known',{}))
    counts = dict(seed_state.get('counts',{}))
    if any(not isinstance(value,int) or value < 0 for value in counts.values()):
        raise ValueError('seed checkpoint has invalid training counts')
    if counts and not known:
        raise ValueError('seed checkpoint lacks traceable training lineage')
    known_hashes = {value.get('sha256') for value in known.values() if isinstance(value,dict)}
    if any(count and digest not in known_hashes for digest,count in counts.items()):
        raise ValueError('seed checkpoint contains untraceable trained manifests')
    for row in rows:
        known[row.key] = row.lineage()
    return known,counts


def _dataset_guard(rows, root):
    for row in rows:
        if row.split == 'test':continue
        assert_episode_not_held(Path(row.path).parent/'manifest.json',corpus=root)
        if sha256_file(row.path) != row.sha256:
            raise ValueError('fixed mixed snapshot changed; prior checkpoints preserved')


def run_bounded(corpus, directory, *, steps, recipe=None, seed_checkpoint=None,
                seed_sha256=None, passes=4, progress=None):
    """Train a frozen admitted snapshot in a new run, validating every update.

    No follow/resume/export integration is implied. A stop.request in this new
    directory stops at the last complete immutable step; both collection GPUs
    remain available to the existing collection process.
    """
    recipe = recipe or Recipe()
    if not isinstance(steps,int) or steps <= 0 or not isinstance(passes,int) or passes <= 0:
        raise ValueError('bounded training requires positive steps and passes')
    if (seed_checkpoint is None) != (seed_sha256 is None):
        raise ValueError('seed path and exact SHA-256 must be supplied together')
    root,directory = Path(corpus).resolve(),Path(directory).resolve()
    if directory.exists():
        raise FileExistsError('mixed run directory must be unused; previous runs are immutable')
    sources = source_bindings()
    rows,complete = discover(root)
    qualified = qualification(rows,recipe)
    if not qualified['ready']:
        raise ValueError('mixed training is not ready: '+', '.join(qualified['missing_required_cohorts']))
    seed_binding = None
    if seed_checkpoint is not None:
        seed = Path(seed_checkpoint).resolve()
        if sha256_file(seed) != seed_sha256:
            raise ValueError('seed checkpoint SHA-256 mismatch')
        seed_binding = dict(path=str(seed),sha256=seed_sha256)
    _source_guard(sources)
    directory.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(recipe.cpu_threads)
    with run_lock(directory):
        events = directory/'events.jsonl'
        def emit(event,**fields):
            value = dict(event=event,pid=os.getpid(),time=time.time(),**fields)
            with events.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(value,allow_nan=False)+'\n');stream.flush()
            write_json(directory/'status.json',value)
            print(json.dumps(value,allow_nan=False),flush=True)
        def advancing(stage,tick,total):
            emit('progress',stage=stage,tick=tick,total=total)
            if progress:progress(stage,tick,total)
        run = dict(schema='cd210_mixed_bounded_run_v1',run_id=uuid4().hex,
            recipe=recipe.document(),recipe_sha256=qualified['recipe_sha256'],corpus=str(root),
            index_sha256=qualified['index_sha256'],source_hashes=sources,seed=seed_binding,
            fixed_snapshot=True,max_steps=steps,passes=passes,legacy_index_complete=complete,
            production_process_modified=False,export_integration_required=True)
        immutable_json(directory/'run.json',run)
        immutable_json(directory/'index.json',dict(rows=[asdict(row) for row in rows],qualification=qualified))
        emit('loading_cd210',run_id=run['run_id'])
        try:
            seed_state = {}
            if seed_binding:
                seed_copy = directory/'seed'/(seed_sha256+'.pt')
                immutable_copy(seed,seed_copy,seed_sha256)
                student,package = load_checkpoint(seed_copy)
                seed_state = package.get('training_state') or {}
                if not isinstance(seed_state,dict):raise ValueError('seed checkpoint has invalid training state')
            else:
                student = create_model()
            student.eval()
            reference = create_model().eval().requires_grad_(False)
            stock_sha = parameter_digest(reference)
            known,counts = _lineage(seed_state,rows)
            state = dict(run_id=run['run_id'],step=0,config=run,known=known,counts=counts,
                sample_counts={},objective_version=OBJECTIVE_VERSION,selection_metric=SELECTION_METRIC,
                initial_parameter_sha256=parameter_digest(student),stock_parameter_sha256=stock_sha,
                seed_checkpoint=seed_binding,source_hashes=sources,last_validation=None)
            assert_checkpoint_not_held(root,state)
            optimizer = make_optimizer(student,recipe)
            _source_guard(sources)
            initial = evaluate(rows,root,student,reference,recipe=recipe,progress=advancing)
            _source_guard(sources)
            _dataset_guard(rows,root)
            assert_checkpoint_not_held(root,state)
            state['last_validation'] = initial
            pointer = _save(directory,student,optimizer,state)
            incumbent,promoted = _publish_validation(directory,pointer,initial,None)
            emit('initial_validation',step=0,best_promoted=promoted,cohorts=initial['cohorts'],checkpoint=pointer)
            for number in range(1,steps+1):
                if (directory/'stop.request').exists():
                    emit('stopped_at_checkpoint',step=state['step']);break
                selected = []
                for kind in KINDS:
                    eligible = [r for r in rows if r.kind == kind and r.split == 'train'
                                and state['sample_counts'].get(r.key,0) < passes]
                    if not eligible:break
                    selected.append(min(eligible,key=lambda r:(state['sample_counts'].get(r.key,0),r.key)))
                if len(selected) != 2:
                    emit('snapshot_pass_limit_reached',step=state['step']);break
                assert_checkpoint_not_held(root,state)
                _source_guard(sources)
                emit('mixed_backward',step=number,episodes=[r.path for r in selected])
                with open_episode(selected[0],root) as legacy,open_episode(selected[1],root) as behavior:
                    report = backward_pair(legacy,behavior,student,reference,recipe=recipe,progress=advancing)
                assert_checkpoint_not_held(root,state)
                _source_guard(sources)
                _dataset_guard(rows,root)
                norm = float(torch.nn.utils.clip_grad_norm_(student.parameters(),recipe.gradient_clip,error_if_nonfinite=True))
                optimizer.step()
                if not all(bool(torch.isfinite(p).all()) for p in student.parameters()):
                    raise ValueError('optimizer produced nonfinite weights; prior checkpoints retained')
                if parameter_digest(student) == report['student_parameter_sha256']:
                    raise ValueError('optimizer did not change model parameters')
                if parameter_digest(reference) != stock_sha:
                    raise ValueError('immutable stock baseline changed')
                state['step'] = number
                for row in selected:
                    state['sample_counts'][row.key] = state['sample_counts'].get(row.key,0)+1
                    state['counts'][row.sha256] = state['counts'].get(row.sha256,0)+1
                result = evaluate(rows,root,student,reference,recipe=recipe,progress=advancing)
                assert_checkpoint_not_held(root,state)
                _source_guard(sources)
                _dataset_guard(rows,root)
                state['last_validation'] = result
                pointer = _save(directory,student,optimizer,state)
                incumbent,promoted = _publish_validation(directory,pointer,result,incumbent)
                emit('mixed_step_committed',step=number,best_promoted=promoted,checkpoint=pointer,
                    gradient_norm_before_clip=norm,cohorts=result['cohorts'],loss_before_update=report['loss'])
            summary = dict(step=state['step'],latest=pointer,best=incumbent,run_id=run['run_id'],
                automatic_collection_modified=False,automatic_follow_enabled=False,export_integration_required=True,
                driving_release_qualified=False)
            immutable_json(directory/'completion.json',summary)
            emit('bounded_training_finished',step=state['step'],best_available=incumbent is not None)
            return summary
        except BaseException as error:
            emit('failed_prior_checkpoints_preserved',error=type(error).__name__+': '+str(error))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,default=Path(r'G:\lane-centering-dataset-v1'))
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--steps',type=int,default=0,help='0: read-only admission qualification; positive: new bounded run')
    parser.add_argument('--passes',type=int,default=4)
    parser.add_argument('--legacy-weight',type=float,default=.5)
    parser.add_argument('--cpu-threads',type=int,default=2)
    parser.add_argument('--seed-checkpoint',type=Path)
    parser.add_argument('--seed-sha256')
    args = parser.parse_args(argv)
    recipe = Recipe(legacy_weight=args.legacy_weight,behavior_weight=1.-args.legacy_weight,cpu_threads=args.cpu_threads)
    if args.steps < 0:parser.error('--steps must be nonnegative')
    if not args.steps:
        rows,_ = discover(args.corpus)
        report = qualification(rows,recipe)
        print(json.dumps(report,indent=2,allow_nan=False))
        return 0 if report['ready'] else 2
    if args.run_dir is None:parser.error('positive --steps requires a new --run-dir')
    run_bounded(args.corpus,args.run_dir,steps=args.steps,recipe=recipe,
        seed_checkpoint=args.seed_checkpoint,seed_sha256=args.seed_sha256,passes=args.passes)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
