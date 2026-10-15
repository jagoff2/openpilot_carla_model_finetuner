"""Lifecycle tests use tiny CD210 ABI models; no production training is started."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import pytest
import torch

from lane_behavior import mixed as m, mixed_follow as f, mixed_runner as runner
from lane_dataset.config import sha256_file
from lane_dataset.review_hold import ReviewHoldError
from lane_training.smoke import parameter_digest
from test_behavior_training import models
from test_behavior_mixed import artifacts, legacy_targets
from test_centerline_objective import straight


def read(path):return json.loads(Path(path).read_text())


def fake_load(path):
    package=torch.load(path,weights_only=True)
    model=models()[0];model.load_state_dict(package['model'])
    return model,package


@pytest.fixture
def harness(tmp_path,monkeypatch):
    """Real AdamW, checkpoint, immutable receipt and restart code, synthetic loss."""
    rows=[]
    for kind in m.KINDS:
        for split in ('train','validation'):
            path=tmp_path/'accepted'/(kind+'-'+split)/('manifest.json' if kind=='legacy' else 'behavior-manifest.json')
            path.parent.mkdir(parents=True);path.write_text(json.dumps(dict(kind=kind,split=split)))
            rows.append(m.EpisodeRef(kind,str(path),sha256_file(path),split,kind+split,'test-fixture',1))
    sources={'unit-test-source':'fixed'}
    monkeypatch.setattr(f,'source_bindings',lambda:sources.copy())
    monkeypatch.setattr(runner,'source_bindings',lambda:sources.copy())
    monkeypatch.setattr(f,'discover',lambda *a,**k:(list(rows),False))
    monkeypatch.setattr(f,'create_model',lambda:models()[0])
    monkeypatch.setattr(f,'load_checkpoint',fake_load)
    monkeypatch.setattr(f,'open_episode',lambda *a,**k:nullcontext(a[0]))
    def backward(old,behavior,model,reference,**kwargs):
        model.zero_grad(set_to_none=True);digest=parameter_digest(model)
        sum(p.square().sum() for p in model.parameters()).backward()
        return dict(student_parameter_sha256=digest)
    def evaluate(rows,root,model,reference,recipe,**kwargs):
        value=float(sum(p.detach().square().sum() for p in model.parameters()))
        base=float(sum(p.detach().square().sum() for p in reference.parameters()))
        selected=sorted((r for r in rows if r.split=='validation'),key=lambda r:r.key)
        scores=[dict(kind=r.kind,episode=r.path,manifest_sha256=r.sha256,labels=1,loss=value,baseline_loss=base) for r in selected]
        return dict(objective_version=m.OBJECTIVE_VERSION,selection_metric=m.SELECTION_METRIC,
            recipe_sha256=m.fingerprint(recipe.document()),cohort=m.fingerprint([r.sha256 for r in selected]),
            loss=value,baseline_loss=base,rows=scores,test_split_used=False,
            cohorts={kind:dict(loss=value,baseline_loss=base,labels=sum(r.kind==kind for r in selected),
                cohort=m.fingerprint(sorted(r.sha256 for r in selected if r.kind==kind))) for kind in m.KINDS})
    monkeypatch.setattr(f,'backward_pair',backward);monkeypatch.setattr(f,'evaluate',evaluate)
    monkeypatch.setattr(f,'_pause',lambda *a:None)
    return tmp_path,tmp_path/'follow',rows,m.Recipe(vision_lr=.01,policy_lr=.01),sources


def test_waits_for_both_admitted_cohorts_without_loading_models(harness,monkeypatch):
    root,directory,rows,recipe,_=harness
    rows[:]=[r for r in rows if r.kind=='legacy']
    monkeypatch.setattr(f,'create_model',lambda:pytest.fail('model loaded before independent admission'))
    result=f.follow(root,directory,recipe=recipe,max_cycles=2)
    assert result['step'] is None and not (directory/'steps').exists()
    identity=(directory/'run.json').read_bytes()
    (directory/'stop.request').touch()
    result=f.follow(root,directory,max_cycles=2)
    assert result['cycles']==0 and (directory/'run.json').read_bytes()==identity


def test_restart_restores_adamw_counts_and_never_repeats_saved_step(harness):
    root,directory,rows,recipe,_=harness
    first=f.follow(root,directory,recipe=recipe,max_steps=1)
    immutable={p:p.read_bytes() for p in directory.rglob('*') if p.is_file() and p.parent.name in ('checkpoints','steps','validation','promotions')}
    second=f.follow(root,directory,max_steps=2)
    assert first['step']==1 and second['step']==2 and first['run_id']==second['run_id']
    for path,before in immutable.items():assert path.read_bytes()==before
    latest=read(directory/'latest.json');package=torch.load(directory/latest['checkpoint'],weights_only=True)
    assert all(value==2 for value in package['training_state']['sample_counts'].values())
    assert {int(value['step']) for value in package['optimizer']['state'].values()}=={2}
    assert len(list((directory/'steps').glob('*.json')))==3
    assert len(list((directory/'preserved'/'promotions').glob('*.json')))==2


@pytest.mark.parametrize('stage',['latest','validation'])
def test_recovers_committed_step_after_pointer_or_validation_publication_failure(harness,monkeypatch,stage):
    root,directory,rows,recipe,_=harness
    if stage=='latest':
        real=runner.write_json
        def interrupt(path,value):
            if Path(path).name=='latest.json' and value['step']==1:raise OSError('simulated publication interruption')
            return real(path,value)
        monkeypatch.setattr(runner,'write_json',interrupt)
    else:
        real=f.publish_step
        def interrupt(directory,pointer,*args,**kwargs):
            if pointer['step']==1:raise OSError('simulated publication interruption')
            return real(directory,pointer,*args,**kwargs)
        monkeypatch.setattr(f,'publish_step',interrupt)
    with pytest.raises(OSError,match='publication'):f.follow(root,directory,recipe=recipe,max_steps=1)
    before=read(directory/'steps'/'000001.json')['sha256']
    monkeypatch.setattr(runner if stage=='latest' else f,'write_json' if stage=='latest' else 'publish_step',real)
    result=f.follow(root,directory,max_steps=1)
    assert result['step']==1 and result['best']['sha256']==before
    assert read(directory/'latest.json')['sha256']==before
    assert (directory/'validation'/'000001.json').exists()
    assert len(list((directory/'steps').glob('*.json')))==2


def test_expanding_data_creates_new_epoch_and_rescores_same_incumbent(harness,monkeypatch):
    root,directory,rows,recipe,_=harness
    first=f.follow(root,directory,recipe=recipe,max_steps=1)
    for kind in m.KINDS:
        old=next(r for r in rows if r.kind==kind and r.split=='validation')
        path=Path(old.path).parent.with_name(kind+'-new-validation')/Path(old.path).name
        path.parent.mkdir();path.write_text(json.dumps(dict(new=True,kind=kind,split='validation')))
        rows.append(replace(old,path=str(path),sha256=sha256_file(path),group=old.group+'new'))
    second=f.follow(root,directory,max_steps=2)
    assert second['step']==2 and len(list((directory/'epochs').glob('*.json')))==2
    state=read(directory/'steps'/'000002.json')['state']
    score=f._bound(directory,state['comparison_best']['score_binding'])
    assert score['identity']['parameter_sha256']==first['best']['parameter_sha256']
    assert len(score['result']['rows'])==4
    protected=second['best']['protected_cohorts']
    assert all(cohort in protected for cohort in first['best']['protected_cohorts'])
    assert any(len(cohort)==1 for cohort in protected)


def test_new_data_is_adopted_only_after_current_update_cycle(harness,monkeypatch):
    root,directory,rows,recipe,_=harness
    real=f.backward_pair;added=False
    def arriving(*args,**kwargs):
        nonlocal added
        if not added:
            old=next(r for r in rows if r.kind=='behavior' and r.split=='validation')
            path=Path(old.path).parent.with_name('new-heldout')/Path(old.path).name
            path.parent.mkdir();path.write_text('{"fresh": true}')
            rows.append(replace(old,path=str(path),sha256=sha256_file(path),group='fresh'))
            added=True
        return real(*args,**kwargs)
    monkeypatch.setattr(f,'backward_pair',arriving)
    f.follow(root,directory,recipe=recipe,settings=f.FollowConfig(cycle_steps=2),max_steps=3)
    states=[read(directory/'steps'/f'{i:06d}.json')['state'] for i in (1,2,3)]
    assert states[0]['epoch_binding']==states[1]['epoch_binding']!=states[2]['epoch_binding']
    assert [len(state['last_validation']['rows']) for state in states]==[2,2,3]


def test_single_writer_lock_and_exhausted_replay_budget(harness):
    root,directory,rows,recipe,_=harness
    with f.run_lock(directory):
        with pytest.raises((RuntimeError,OSError)):f.follow(root,directory,recipe=recipe,max_cycles=1)
    result=f.follow(root,directory,recipe=recipe,settings=f.FollowConfig(passes=1),max_cycles=2)
    assert result['step']==1
    assert read(directory/'status.json')['event']=='waiting_for_new_accepted_training_data'


def test_corrupt_winning_checkpoint_does_not_rewrite_best_pointer(harness):
    root,directory,rows,recipe,_=harness
    result=f.follow(root,directory,recipe=recipe,max_steps=1)
    pointer=(directory/'best.json').read_bytes()
    (directory/result['best']['checkpoint']).write_bytes(b'corrupt checkpoint')
    with pytest.raises(ValueError,match='integrity'):f.follow(root,directory,max_steps=2)
    assert (directory/'best.json').read_bytes()==pointer


def test_prior_behavior_subset_regression_cannot_hide_in_enlarged_cohort(harness):
    root,directory,rows,recipe,_=harness
    old=f.follow(root,directory,recipe=recipe,max_steps=1)['best']
    incumbent=deepcopy(old);candidate=deepcopy(old)
    behavior=next(row for row in candidate['rows'] if row['kind']=='behavior')
    behavior['loss']+=.001
    for result in (incumbent,candidate):
        result['rows'].append(dict(kind='behavior',manifest_sha256='new',labels=100,loss=.2))
        result['cohort']='expanded';result['cohorts']['behavior'].update(labels=101,cohort='expanded-behavior')
    candidate['loss']-=.01;candidate['cohorts']['behavior']['loss']-=.01
    assert not m.can_promote_mixed(candidate,incumbent)


def test_worse_candidate_preserves_prior_best_bytes(harness,monkeypatch):
    root,directory,rows,recipe,_=harness
    f.follow(root,directory,recipe=recipe,max_steps=1)
    pointer=(directory/'best.json').read_bytes();best=read(directory/'best.json')
    artifact=(directory/best['checkpoint']).read_bytes();real=f.backward_pair
    def bad(*args,**kwargs):
        report=real(*args,**kwargs)
        for p in args[2].parameters():p.grad.mul_(-100)
        return report
    monkeypatch.setattr(f,'backward_pair',bad)
    f.follow(root,directory,max_steps=2)
    assert (directory/'best.json').read_bytes()==pointer
    assert (directory/best['checkpoint']).read_bytes()==artifact
    assert read(directory/'latest.json')['step']==2


@pytest.mark.parametrize('change',['source','recipe','trained_missing','validation_missing','manifest_changed','latest_corrupt','unknown_best'])
def test_resume_rejects_identity_integrity_and_data_changes(harness,change):
    root,directory,rows,recipe,sources=harness
    f.follow(root,directory,recipe=recipe,max_steps=1)
    winner=read(directory/'best.json');before=(directory/winner['checkpoint']).read_bytes()
    kwargs={}
    if change=='source':sources['unit-test-source']='changed'
    elif change=='recipe':kwargs['recipe']=replace(recipe,legacy_weight=.25,behavior_weight=.75)
    elif change=='trained_missing':rows[:]=[r for r in rows if r.kind!='behavior' or r.split!='train']
    elif change=='validation_missing':rows[:]=[r for r in rows if r.kind!='behavior' or r.split!='validation']
    elif change=='manifest_changed':rows[0]=replace(rows[0],sha256='x'*64)
    elif change=='latest_corrupt':(directory/'latest.json').write_text(json.dumps(dict(step=1,sha256='x'*64)))
    elif change=='unknown_best':(directory/'best.json').write_text(json.dumps(dict(step=100,sha256='x'*64)))
    with pytest.raises(ValueError):f.follow(root,directory,max_steps=2,**kwargs)
    assert (directory/winner['checkpoint']).read_bytes()==before


def test_review_hold_stops_resume_without_altering_checkpoint(harness):
    root,directory,rows,recipe,_=harness
    f.follow(root,directory,recipe=recipe,max_steps=1)
    (root/'reports').mkdir();(root/'reports'/'review-holds.json').write_text(json.dumps(dict(schema_version=1,
        episodes={'behavior-train':dict(status='held',reasons=['independent review'])})))
    with pytest.raises(ReviewHoldError):f.follow(root,directory,max_steps=2)
    assert read(directory/'latest.json')['step']==1


def test_real_admitted_loaders_mixed_gradients_and_validation_resume(artifacts,monkeypatch):
    import lane_training.production as production
    legacy_targets(monkeypatch)
    monkeypatch.setattr(production,'episode_corridors',lambda _:[replace(straight(),label_index=100)])
    root=artifacts[0].parents[2];paths=list(artifacts)
    for path in artifacts:
        destination=path.parent.with_name(path.parent.name+'-validation');shutil.copytree(path.parent,destination)
        new=destination/path.name;document=read(new);document['split']='validation';new.write_text(json.dumps(document));paths.append(new)
    rows=[m.EpisodeRef('legacy' if path.name=='manifest.json' else 'behavior',str(path),sha256_file(path),
        'train' if i<2 else 'validation',str(i),'straight_recovery',1 if path.name=='manifest.json' else 2) for i,path in enumerate(paths)]
    monkeypatch.setattr(f,'discover',lambda *a,**k:(rows,False))
    monkeypatch.setattr(f,'create_model',lambda:models()[0]);monkeypatch.setattr(f,'load_checkpoint',fake_load)
    monkeypatch.setattr(f,'source_bindings',lambda:{'fixture':'fixed'});monkeypatch.setattr(runner,'source_bindings',lambda:{'fixture':'fixed'})
    monkeypatch.setattr(f,'_pause',lambda *a:None)
    directory=root/'real-mixed-follow'
    first=f.follow(root,directory,recipe=m.Recipe(vision_lr=.01,policy_lr=.01),max_steps=1)
    second=f.follow(root,directory,max_steps=2)
    assert first['step']==1 and second['step']==2
    state=read(directory/'steps'/'000002.json')['state']
    assert {r['kind'] for r in state['last_validation']['rows']}==set(m.KINDS)
    assert all(value==2 for value in state['sample_counts'].values())
