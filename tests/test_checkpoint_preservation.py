import json
from pathlib import Path

import pytest
import torch

from lane_training import checkpoint_store as store
from lane_training import production as production
from lane_training.cd210_model import export_onnx
from lane_release import export


def result(loss, *, step=1, cohort='cohort-a', baseline=3.0, losses=None):
    losses = losses or {'a': loss}
    return dict(step=step, cohort=cohort, loss=loss, baseline_loss=baseline,
        rows=[dict(manifest_sha256=k, loss=v, labels=20) for k,v in losses.items()])


def checkpoint(root, data, step):
    path=root/f'checkpoint-{step % 2}.pt'
    path.write_bytes(data)
    return dict(file=path.name, sha256=store.sha256_file(path), step=step)


def test_worse_and_tied_checkpoints_cannot_replace_best(tmp_path):
    first=checkpoint(tmp_path,b'better than stock',1)
    assert store.protect_validation(tmp_path,first,result(1.0))
    pointer=(tmp_path/'best.json').read_bytes()
    best=json.loads(pointer);saved=store.checkpoint_path(tmp_path,best)
    for step,loss in [(2,2.),(3,3.5),(4,1.)]:
        candidate=checkpoint(tmp_path,f'candidate-{step}'.encode(),step)
        assert not store.protect_validation(tmp_path,candidate,result(loss,step=step),best)
        assert (tmp_path/'best.json').read_bytes()==pointer
        assert saved.read_bytes()==b'better than stock'
    # Even the rotating original slot can be overwritten without touching the best.
    checkpoint(tmp_path,b'rotated',5)
    assert saved.read_bytes()==b'better than stock'


def test_new_better_winner_preserves_all_prior_winners(tmp_path):
    first=checkpoint(tmp_path,b'first winner',1)
    store.protect_validation(tmp_path,first,result(2.))
    old=json.loads((tmp_path/'best.json').read_text())
    second=checkpoint(tmp_path,b'second winner',2)
    assert store.protect_validation(tmp_path,second,result(1.,step=2),old)
    best=json.loads((tmp_path/'best.json').read_text())
    assert best['sha256']==second['sha256']
    assert store.checkpoint_path(tmp_path,old).read_bytes()==b'first winner'
    assert store.checkpoint_path(tmp_path,best).read_bytes()==b'second winner'
    assert len(list((tmp_path/'preserved/promotions').glob('*.json')))==2


def test_changed_cohort_requires_comparable_incumbent_and_rejects_old_cohort_regression():
    previous=dict(result(1.),protected_cohorts=[['a']])
    challenger=result(.7,cohort='expanded',losses={'a':1.2,'b':.2})
    with pytest.raises(ValueError,match='same validation cohort'):
        store.can_promote(challenger,previous)
    rescored=dict(result(1.5,cohort='expanded',losses={'a':1.,'b':2.}),protected_cohorts=[['a']])
    assert not store.can_promote(challenger,rescored)
    improved=result(.5,cohort='expanded',losses={'a':.8,'b':.2})
    assert store.can_promote(improved,rescored)
    assert store.protected_cohorts(improved,rescored)==[['a'],['a','b']]


def test_missing_protected_data_and_nonfinite_scores_fail_closed():
    incumbent=dict(result(2.),protected_cohorts=[['missing']])
    with pytest.raises(ValueError,match='missing'):
        store.can_promote(result(1.),incumbent)
    with pytest.raises(ValueError,match='finite'):
        store.can_promote(result(float('nan')))


def test_crash_before_best_pointer_commit_preserves_old_best_and_new_candidate(tmp_path,monkeypatch):
    first=checkpoint(tmp_path,b'old',1)
    store.protect_validation(tmp_path,first,result(2.))
    pointer=(tmp_path/'best.json').read_bytes();best=json.loads(pointer)
    second=checkpoint(tmp_path,b'new',2)
    original=store.write_json
    def fail_pointer(path,value):
        if Path(path).name=='best.json':raise OSError('simulated crash')
        original(path,value)
    monkeypatch.setattr(store,'write_json',fail_pointer)
    with pytest.raises(OSError,match='crash'):
        store.protect_validation(tmp_path,second,result(1.,step=2),best)
    assert (tmp_path/'best.json').read_bytes()==pointer
    assert store.checkpoint_path(tmp_path,best).read_bytes()==b'old'
    assert (tmp_path/'preserved/checkpoints'/f'{second["sha256"]}.pt').read_bytes()==b'new'


def test_immutable_artifact_collision_and_corruption_never_overwrite(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'good')
    destination=tmp_path/'saved';store.immutable_copy(source,destination)
    source.write_bytes(b'bad')
    with pytest.raises(ValueError,match='immutable artifact'):
        store.immutable_copy(source,destination)
    assert destination.read_bytes()==b'good'
    with pytest.raises(ValueError,match='changed'):
        store.immutable_copy(source,tmp_path/'new',expected='0'*64)
    assert not (tmp_path/'new').exists()


def test_legacy_best_is_preserved_before_promotion(tmp_path):
    old=result(2.);(tmp_path/'best.pt').write_bytes(b'legacy')
    old.update(checkpoint='best.pt',sha256=store.sha256_file(tmp_path/'best.pt'))
    store.write_json(tmp_path/'best.json',old)
    candidate=checkpoint(tmp_path,b'new',2)
    assert store.protect_validation(tmp_path,candidate,result(1.,step=2),old)
    assert (tmp_path/'best.pt').read_bytes()==b'legacy'
    assert (tmp_path/'preserved/checkpoints'/f'{old["sha256"]}.pt').read_bytes()==b'legacy'


def test_export_retains_incumbent_for_ties_worse_and_historical_regression():
    old=dict(result(1.),incumbent_export=True,kind='fine_tuned',protected_cohorts=[['a']])
    for candidate in [result(1.,step=10),result(2.,step=20)]:
        selected=export.select_best([candidate,old])
        assert selected['incumbent_export'] and not selected['promote_export']
    expanded=dict(result(1.5,cohort='expanded',losses={'a':1.,'b':2.}),
        incumbent_export=True,kind='fine_tuned',protected_cohorts=[['a']])
    candidate=result(.7,step=20,cohort='expanded',losses={'a':1.2,'b':.2})
    assert export.select_best([candidate,expanded])['incumbent_export']
    candidate=result(.5,step=21,cohort='expanded',losses={'a':.8,'b':.2})
    assert export.select_best([candidate,expanded])['promote_export']


def test_export_exact_stock_tie_keeps_stock():
    stock=dict(result(3.,step=0),kind='original_cd210')
    trained=result(3.,step=9)
    assert export.select_best([trained,stock])['kind']=='original_cd210'


def test_existing_full_model_export_cannot_be_overwritten(tmp_path):
    path=tmp_path/'driving_vision.onnx';path.write_bytes(b'previous full model')
    with pytest.raises(FileExistsError,match='already exists'):
        export_onnx(None,tmp_path)
    assert path.read_bytes()==b'previous full model'


def test_successful_device_package_resume_is_read_only(tmp_path,monkeypatch):
    archive=tmp_path/'good.zip';archive.write_bytes(b'previous verified package')
    store.write_json(tmp_path/'payload.json',dict(files={}))
    store.write_json(tmp_path/'status.json',dict(runtime_ready=True,archive=str(archive),
        archive_sha256=store.sha256_file(archive)))
    monkeypatch.setattr(export.subprocess,'run',lambda *a,**k:pytest.fail('must not rebuild'))
    export.compile_remote(tmp_path,export.DEFAULT_DEVICE)
    assert archive.read_bytes()==b'previous verified package'


def test_failed_export_does_not_change_best_pointer(tmp_path,monkeypatch):
    releases=tmp_path/'releases';releases.mkdir()
    pointer=releases/'best-export.json';pointer.write_text('{"previous":"winner"}')
    before=pointer.read_bytes()
    monkeypatch.setattr(export,'select_checkpoint',lambda *a:(None,dict(promote_export=True),[]))
    def fail(*a):raise RuntimeError('export failure')
    monkeypatch.setattr(export,'export_models',fail)
    with pytest.raises(RuntimeError,match='export failure'):
        export.prepare(tmp_path/'run',releases)
    assert pointer.read_bytes()==before


def test_training_completion_exports_best_not_latest(tmp_path,monkeypatch):
    trainer=production.Trainer.__new__(production.Trainer)
    trainer.directory=tmp_path
    trainer.cfg=production.TrainingConfig(corpus=str(tmp_path))
    trainer.state=dict(step=8,known={},counts={},failed={},last_validation=dict(step=8,cohort='unused'))
    trainer.status=lambda *a,**k:None
    trainer.model=object()
    monkeypatch.setattr(production,'discover',lambda *a:([],True))
    calls=[]
    monkeypatch.setattr(export,'prepare',lambda run,releases:calls.append(run) or releases/'best')
    monkeypatch.setattr(production,'export_onnx',lambda *a:pytest.fail('must not export latest weights'))
    trainer.run()
    assert calls==[tmp_path]
    assert json.loads((tmp_path/'completion.json').read_text())['exports'].endswith('best')


@pytest.mark.parametrize('lateral_selection',[False,True])
def test_preservation_migration_keeps_weights_optimizer_step_and_sampler(tmp_path,monkeypatch,lateral_selection):
    from dataclasses import asdict
    from test_production_training import Tiny
    model=Tiny();cfg=production.TrainingConfig(corpus=str(tmp_path))
    optimizer=production.make_optimizer(model,cfg)
    (model.policy(model.vision(torch.ones(1,1)))**2).sum().backward();optimizer.step()
    sources={p.name:production.digest(p) for p in Path(production.__file__).parent.glob('*.py')}
    sources['production.py']='old-persistence-source'
    sources.pop('checkpoint_store.py')
    state=dict(config=asdict(cfg),step=7,counts={'episode':2},known={},failed={},
        last_validation={'step':7},source_hashes=sources)
    production.commit(tmp_path,model,optimizer,state)
    state['step']=8
    production.commit(tmp_path,model,optimizer,state)
    def load(path):
        package=torch.load(path,weights_only=True)
        restored=Tiny();restored.load_state_dict(package['model'])
        return restored,package
    monkeypatch.setattr(production,'load_checkpoint',load)
    before=production.parameter_digest(model)
    audit=production.migrate_preservation(tmp_path,cfg,lateral_selection=lateral_selection)
    restored,restored_optimizer,saved=production.restore(tmp_path,cfg)
    assert production.parameter_digest(restored)==before==audit['parameter_sha256']
    assert saved['step']==8 and saved['counts']=={'episode':2}
    assert saved['last_validation'] is None
    assert saved['selection_metric']==production.SELECTION_METRIC
    assert audit['purpose']==('lateral_only_selection' if lateral_selection else 'checkpoint_preservation')
    assert len(audit['preserved'])==2
    for old,new in zip(optimizer.state.values(),restored_optimizer.state.values()):
        assert old.keys()==new.keys()
        assert all(torch.equal(old[k],new[k]) for k in old)
    for record in audit['preserved']:
        assert store.sha256_file(store.checkpoint_path(tmp_path,record))==record['sha256']
