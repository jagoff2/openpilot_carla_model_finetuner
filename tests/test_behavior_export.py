from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import onnx
from onnx import helper, TensorProto
import pytest
import torch

from lane_behavior import export as e
from lane_behavior import mixed as m
from lane_behavior import mixed_runner as runner
from lane_dataset.config import sha256_file
from lane_dataset.review_hold import ReviewHoldError
from test_behavior_mixed import artifacts
from test_behavior_training import models


@pytest.fixture
def mixed_run(artifacts,monkeypatch):
    """Synthetic unit-test receipts only, never production admission artifacts."""
    root=artifacts[0].parents[2];paths=list(artifacts)
    for path in artifacts:
        destination=path.parent.with_name(path.parent.name+'-validation')
        shutil.copytree(path.parent,destination)
        new=destination/path.name;manifest=json.loads(new.read_text());manifest['split']='validation'
        new.write_text(json.dumps(manifest));paths.append(new)
    rows=[m.EpisodeRef('legacy' if path.name=='manifest.json' else 'behavior',str(path),sha256_file(path),
        'train' if index<2 else 'validation',str(index),'straight_recovery',1 if path.name=='manifest.json' else 2)
        for index,path in enumerate(paths)]
    recipe=m.Recipe();qualification=m.qualification(rows,recipe)
    directory=root/'mixed-run';directory.mkdir()
    run=dict(schema='cd210_mixed_bounded_run_v1',run_id='test-only',fixed_snapshot=True,
        corpus=str(root),recipe=recipe.document(),recipe_sha256=qualification['recipe_sha256'],
        index_sha256=qualification['index_sha256'],source_hashes=runner.source_bindings())
    (directory/'run.json').write_text(json.dumps(run))
    (directory/'index.json').write_text(json.dumps(dict(rows=[asdict(row) for row in rows],qualification=qualification)))
    heldout=sorted((row for row in rows if row.split=='validation'),key=lambda row:row.key)
    scores=[dict(kind=row.kind,episode=row.path,manifest_sha256=row.sha256,labels=row.labels,loss=1.,baseline_loss=2.) for row in heldout]
    validation=dict(objective_version=m.OBJECTIVE_VERSION,selection_metric=m.SELECTION_METRIC,
        recipe_sha256=run['recipe_sha256'],cohort=m.fingerprint([row.sha256 for row in heldout]),
        rows=scores,loss=1.,baseline_loss=2.,test_split_used=False,
        cohorts={row.kind:dict(loss=1.,baseline_loss=2.,labels=row.labels,cohort=m.fingerprint([row.sha256])) for row in heldout})
    state=dict(step=1,run_id=run['run_id'],config=run,source_hashes=run['source_hashes'],
        known={row.key:row.lineage() for row in rows},counts={row.sha256:1 for row in rows if row.split=='train'},
        sample_counts={row.key:1 for row in rows if row.split=='train'},last_validation=validation,
        objective_version=m.OBJECTIVE_VERSION,selection_metric=m.SELECTION_METRIC,stock_parameter_sha256='1'*64)
    student,_=models();optimizer=torch.optim.SGD(student.parameters(),lr=.01)
    pointer=runner._save(directory,student,optimizer,state)
    winner,promoted=runner._publish_validation(directory,pointer,validation,None);assert promoted
    monkeypatch.setattr(e,'discover',lambda *args,**kwargs:(rows,False))
    return directory,rows,winner


def test_selects_only_exact_committed_mixed_promotion_and_keeps_objective(mixed_run):
    run,rows,best=mixed_run
    selection,returned=e.select_validated_checkpoint(run)
    assert selection['step']==1 and selection['source_checkpoint_sha256']==best['sha256']
    assert selection['objective_version']==m.OBJECTIVE_VERSION and selection['selection_metric']==m.SELECTION_METRIC
    assert selection['objective_recipe']['legacy_weight']==selection['objective_recipe']['behavior_weight']==.5
    assert set(selection['cohorts'])=={'legacy','behavior'} and not selection['test_split_used']
    assert selection['source_receipts']['promotion']['sha256']
    assert returned==rows


def test_follow_export_uses_winning_epoch_and_rejects_changed_epoch(mixed_run):
    from lane_behavior import mixed_follow as follow
    from lane_training.checkpoint_store import immutable_json
    from lane_training.smoke import parameter_digest
    old,rows,old_best=mixed_run
    destination=old.parent/'follow-export-fixture';destination.mkdir()
    run=deepcopy(json.loads((old/'run.json').read_text()))
    run.update(schema=follow.SCHEMA,fixed_snapshot=False,snapshot_policy='immutable_epoch_at_cycle_boundary',
        follow=dict(passes=4,cycle_steps=1,poll_seconds=10.))
    run.pop('index_sha256');immutable_json(destination/'run.json',run)
    epoch=follow._epoch(destination,rows,m.Recipe())
    package=torch.load(old/old_best['checkpoint'],weights_only=True)
    state=deepcopy(package['training_state']);state.update(config=run,epoch_binding=epoch)
    student,_=models();state['parameter_sha256']=parameter_digest(student)
    optimizer=torch.optim.AdamW(student.parameters(),lr=.01)
    pointer=runner._save(destination,student,optimizer,state)
    best=dict(old_best,**pointer,epoch_binding=epoch,parameter_sha256=state['parameter_sha256'],previous_best_sha256=None)
    immutable_json(destination/'best.json',best)
    immutable_json(destination/'validation'/'000001.json',dict(state['last_validation'],step=1))
    immutable_json(destination/'preserved'/'promotions'/f'000001-{pointer["sha256"]}.json',best)
    # A newer snapshot file must not redefine the selected checkpoint's cohort.
    immutable_json(destination/'epochs'/'unselected-new.json',{'unselected':True})
    selected,returned=e.select_validated_checkpoint(destination)
    assert returned==rows and selected['source_receipts']['index']['sha256']==epoch['sha256']
    assert Path(selected['source_receipts']['index']['path']).parent==destination/'epochs'
    epoch_path=destination/epoch['path'];epoch_path.write_text('{}')
    with pytest.raises(ValueError,match='epoch integrity'):e.select_validated_checkpoint(destination)


@pytest.mark.parametrize('change',['unstepped','legacy_metric','worse_behavior','missing_receipt','changed_receipt','wrong_checkpoint'])
def test_export_rejects_unvalidated_or_changed_selection(mixed_run,change):
    run,_,best=mixed_run
    if change=='missing_receipt':
        (run/'validation'/'000001.json').unlink()
    elif change=='changed_receipt':
        path=run/'validation'/'000001.json';value=json.loads(path.read_text());value['loss']=.8;path.write_text(json.dumps(value))
    else:
        value=json.loads((run/'best.json').read_text())
        if change=='unstepped':value['step']=0
        if change=='legacy_metric':value['selection_metric']='legacy'
        if change=='worse_behavior':value['cohorts']['behavior']['loss']=3.
        if change=='wrong_checkpoint':value['sha256']='f'*64
        (run/'best.json').write_text(json.dumps(value))
    with pytest.raises((ValueError,FileNotFoundError)):e.select_validated_checkpoint(run)


def test_current_hold_blocks_export_without_destroying_candidate(mixed_run):
    run,rows,best=mixed_run
    root=run.parent;(root/'reports').mkdir()
    (root/'reports'/'review-holds.json').write_text(json.dumps(dict(schema_version=1,
        episodes={'unmarked-test':dict(status='held',reasons=['new review'])})))
    checkpoint=run/best['checkpoint'];before=checkpoint.read_bytes()
    with pytest.raises(ReviewHoldError):e.select_validated_checkpoint(run)
    assert checkpoint.read_bytes()==before


def test_existing_output_is_never_replaced(tmp_path,monkeypatch):
    original=tmp_path/'prior.onnx';original.write_bytes(b'prior better model')
    monkeypatch.setattr(e,'select_validated_checkpoint',lambda *args:pytest.fail('existing output must reject first'))
    with pytest.raises(FileExistsError,match='unused'):e.prepare(tmp_path/'run',tmp_path)
    assert original.read_bytes()==b'prior better model'


def test_mixed_identity_is_carried_by_onnx_pickle_and_export_report(tmp_path):
    roles={}
    for role in ('vision','policy'):
        graph=helper.make_graph([helper.make_node('Identity',['x'],['y'])],'fixture',
            [helper.make_tensor_value_info('x',TensorProto.FLOAT,[1])],
            [helper.make_tensor_value_info('y',TensorProto.FLOAT,[1])])
        model=helper.make_model(graph);helper.set_model_props(model,dict(model_checkpoint='old',base_model_checkpoint='CD210'))
        onnx.save(model,tmp_path/(role+'.onnx'))
        metadata=tmp_path/(role+'_metadata.pkl')
        with metadata.open('wb') as stream:pickle.dump(dict(model_checkpoint='old',input_shapes={'x':(1,)}),stream)
        roles[role]=dict(onnx=role+'.onnx',metadata=metadata.name,artifact=role+'.pkl')
    (tmp_path/'export.json').write_text(json.dumps(dict(exports={role:{'sha256':'before'} for role in roles})))
    selection=dict(source_checkpoint_sha256='c'*64,recipe_sha256='r'*64,source_run_id='run',
        cohort='cohort',cohorts={'legacy':{'loss':1.},'behavior':{'loss':1.}},stock_parameter_sha256='s'*64,
        step=24,objective_recipe=m.Recipe().document())
    identity=e._stamp_identity(tmp_path,roles,selection)
    assert identity.startswith('local/cd210-mixed/')
    for role,entry in roles.items():
        graph=onnx.load(tmp_path/entry['onnx']);props={p.key:p.value for p in graph.metadata_props}
        assert props['fine_tuning_objective']==m.OBJECTIVE_VERSION and props['model_checkpoint']==identity
        assert props['base_model_checkpoint']=='CD210'
        with (tmp_path/entry['metadata']).open('rb') as stream:metadata=pickle.load(stream)
        assert metadata['model_checkpoint']==identity and metadata['fine_tuning_selection_metric']==m.SELECTION_METRIC
        assert 'cd210_mixed_s24' in entry['metadata'] and 'cd210_mixed_s24' in entry['artifact']
        assert entry['onnx_sha256']==sha256_file(tmp_path/entry['onnx'])
    report=json.loads((tmp_path/'export.json').read_text())
    assert report['objective_version']==m.OBJECTIVE_VERSION
    assert report['selection_metric']==m.SELECTION_METRIC


def test_prepare_reuses_export_and_trace_without_legacy_selection_or_device(mixed_run,monkeypatch):
    run,rows,best=mixed_run
    def load(path):
        return models()[0],torch.load(path,map_location='cpu',weights_only=True)
    monkeypatch.setattr(e,'load_checkpoint',load)
    monkeypatch.setattr(e.legacy_export,'select_checkpoint',lambda *args:pytest.fail('legacy reselection is forbidden'))
    monkeypatch.setattr(e.legacy_export,'compile_remote',lambda *args:pytest.fail('no device action authorized'))
    def export_models(model,selection,output):
        assert set(selection['cohorts'])=={'legacy','behavior'}
        roles={}
        for role in ('vision','policy'):
            graph=helper.make_graph([helper.make_node('Identity',['x'],['y'])],'unit-test-only',
                [helper.make_tensor_value_info('x',TensorProto.FLOAT,[1])],
                [helper.make_tensor_value_info('y',TensorProto.FLOAT,[1])])
            model=helper.make_model(graph);helper.set_model_props(model,dict(model_checkpoint='old',base_model_checkpoint='CD210'))
            onnx.save(model,output/(role+'.onnx'))
            with (output/(role+'_metadata.pkl')).open('wb') as stream:pickle.dump(dict(model_checkpoint='old'),stream)
            roles[role]=dict(onnx=role+'.onnx',metadata=role+'_metadata.pkl',artifact=role+'.pkl',inputs=['x'])
        (output/'export.json').write_text(json.dumps(dict(exports={role:dict(sha256='old') for role in roles})))
        return roles
    def trace(output,roles,validation):
        assert len(validation)==1 and validation[0]['kind']=='legacy' and validation[0]['split']=='validation'
        np.savez(output/'verification.npz',fixture=np.zeros(1))
        value=dict(passed=True,scope='unit-test-stub-not-real-model-parity')
        (output/'onnx-verification.json').write_text(json.dumps(value));return value
    monkeypatch.setattr(e.legacy_export,'export_models',export_models)
    monkeypatch.setattr(e.legacy_export,'verification_trace',trace)
    monkeypatch.setattr(e.legacy_export,'bundle_template',lambda roles,selection:dict(models=list(roles),is20hz=True))
    output=run.parent/'mixed-export'
    prior=run.parent/'best-export.json';prior.write_text('prior output stays untouched')
    assert e.prepare(run,output)==output
    payload=json.loads((output/'payload.json').read_text())
    assert payload['objective_version']==m.OBJECTIVE_VERSION
    assert payload['recipe_sha256']==best['recipe_sha256']
    assert set(payload['heldout_cohorts'])=={'legacy','behavior'}
    assert payload['bundle']['ref'].startswith('cd210-mixed-s1-')
    assert 'selection.json' in payload['files']
    for name,digest in payload['files'].items():assert sha256_file(output/name)==digest
    assert prior.read_text()=='prior output stays untouched'
    assert not json.loads((output/'status.json').read_text())['runtime_ready']
    assert sha256_file(output/'checkpoint.pt')==best['sha256']
