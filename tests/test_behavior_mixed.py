import copy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
import torch

from lane_behavior import mixed as m
from lane_behavior import mixed_runner as runner
from lane_behavior.loader import AdmittedBehaviorSequence
from lane_dataset.config import sha256_file
from lane_dataset.loader import SequenceEpisode
from lane_dataset.review_hold import ReviewHoldError
from lane_training.cd210 import backward
from lane_training.smoke import parameter_digest
from test_behavior_training import write_fixture as behavior_fixture, refresh, models
from test_loader import write_fixture as legacy_fixture
from test_centerline_objective import straight


@pytest.fixture
def artifacts(tmp_path):
    legacy_root = tmp_path/'accepted'/'legacy';legacy_root.mkdir(parents=True)
    old = legacy_fixture(legacy_root,count=301)
    value = json.loads(old.read_text())
    value.update(split='train',admission={'passed':True},accepted_label_count=1,
                 geometry_group_id='old',family='straight_recovery',config_sha256='config')
    old.write_text(json.dumps(value))
    behavior = behavior_fixture(tmp_path)
    return old,behavior


def legacy_targets(monkeypatch):
    import lane_training.rollout as rollout
    monkeypatch.setattr(rollout,'episode_corridors',lambda _: [replace(straight(),label_index=100)])


def test_mixed_gradient_is_exact_weighted_sum_at_same_weights(artifacts,monkeypatch):
    legacy_targets(monkeypatch)
    student,reference=models()
    a,b=copy.deepcopy(student),copy.deepcopy(student)
    recipe=m.Recipe(legacy_weight=.25,behavior_weight=.75)
    before=parameter_digest(student);base=parameter_digest(reference)
    with SequenceEpisode(artifacts[0]) as old,AdmittedBehaviorSequence(artifacts[1]) as behavior:
        _,old_report=backward(old,a,reference)
        _,behavior_report=m.behavior_backward(behavior,b,reference)
        expected={name:.25*dict(a.named_parameters())[name].grad+.75*dict(b.named_parameters())[name].grad
                  for name,_ in student.named_parameters()}
        report=m.backward_pair(old,behavior,student,reference,recipe=recipe)
    for name,p in student.named_parameters():torch.testing.assert_close(p.grad,expected[name])
    assert report['loss']==pytest.approx(.25*old_report['loss']+.75*behavior_report['loss'])
    assert report['cohorts']['legacy']['valid_labels']==1
    assert report['cohorts']['behavior']['valid_labels']==2
    assert not report['optimizer_step_performed'] and not report['test_split_used']
    assert parameter_digest(student)==before and parameter_digest(reference)==base
    assert all(p.grad is None for p in reference.parameters())
    optimizer=torch.optim.SGD(student.parameters(),lr=.001)
    optimizer.step()
    assert parameter_digest(student)!=before and parameter_digest(reference)==base


@pytest.mark.parametrize('kind,split',[('legacy','validation'),('legacy','test'),('behavior','validation'),('behavior','test')])
def test_heldout_never_updates_weights(artifacts,monkeypatch,kind,split):
    legacy_targets(monkeypatch)
    path=artifacts[0 if kind=='legacy' else 1]
    value=json.loads(path.read_text());value['split']=split;path.write_text(json.dumps(value))
    student,reference=models();before=parameter_digest(student)
    with SequenceEpisode(artifacts[0]) as old,AdmittedBehaviorSequence(artifacts[1]) as behavior:
        with pytest.raises(ValueError,match='only train split'):m.backward_pair(old,behavior,student,reference)
    assert parameter_digest(student)==before
    assert all(p.grad is None for p in student.parameters())


def test_late_legacy_review_hold_discards_both_gradients(artifacts,monkeypatch):
    legacy_targets(monkeypatch)
    root=artifacts[0].parents[2];(root/'reports').mkdir()
    def progress(stage,tick,total):
        if stage=='behavior:student' and tick==100:
            (root/'reports'/'review-holds.json').write_text(json.dumps(dict(schema_version=1,
                episodes={'legacy':dict(status='held',reasons=['review'])})))
    student,reference=models()
    with SequenceEpisode(artifacts[0]) as old,AdmittedBehaviorSequence(artifacts[1]) as behavior:
        with pytest.raises(ReviewHoldError):m.backward_pair(old,behavior,student,reference,progress=progress)
    assert all(p.grad is None for p in student.parameters())


def test_second_backward_failure_cannot_leave_first_gradient(artifacts,monkeypatch):
    legacy_targets(monkeypatch)
    def fail(*args,**kwargs):raise RuntimeError('damaged behavior payload')
    monkeypatch.setattr(m,'behavior_backward',fail)
    student,reference=models()
    with SequenceEpisode(artifacts[0]) as old,AdmittedBehaviorSequence(artifacts[1]) as behavior:
        with pytest.raises(RuntimeError,match='damaged'):m.backward_pair(old,behavior,student,reference)
    assert all(p.grad is None for p in student.parameters())


def test_behavior_validation_is_forward_only_with_separate_lateral_score(artifacts):
    path=artifacts[1];value=json.loads(path.read_text());value['split']='validation';path.write_text(json.dumps(value))
    student,reference=models();before=parameter_digest(student);base=parameter_digest(reference)
    with AdmittedBehaviorSequence(path) as episode:report=m.evaluate_behavior(episode,student,reference)
    assert report['labels']==2 and len(report['label_scores'])==2
    assert report['loss']>0
    assert parameter_digest(student)==before and parameter_digest(reference)==base
    assert all(p.grad is None for p in (*student.parameters(),*reference.parameters()))
    sample=dict(lateral_loss=1.,heading_loss=2.,clearance_loss=3.,origin_loss=4.,obstacle_loss=5.)
    assert m.behavior_lateral_score(sample)==pytest.approx(m.lateral_score(sample)+m.LossConfig().clearance_weight*5.)


def candidate(legacy=1.,behavior=1.,*,base=2.):
    cohorts={kind:dict(loss=value,baseline_loss=base,labels=10,cohort=kind)
             for kind,value in zip(m.KINDS,(legacy,behavior))}
    rows=[dict(manifest_sha256=kind,loss=value,labels=10) for kind,value in zip(m.KINDS,(legacy,behavior))]
    return dict(selection_metric=m.SELECTION_METRIC,recipe_sha256='recipe',cohort='both',
        loss=(legacy+behavior)/2,baseline_loss=base,cohorts=cohorts,rows=rows)


def test_cohort_regression_cannot_hide_in_better_combined_score():
    old=candidate();old['protected_cohorts']=[['legacy','behavior']]
    assert m.can_promote_mixed(candidate(.9,.9),old)
    assert not m.can_promote_mixed(candidate(.1,1.1),old)
    assert not m.can_promote_mixed(candidate(1.1,.1),old)
    assert not m.can_promote_mixed(candidate(.1,2.1))
    assert not m.can_promote_mixed(candidate(2.,2.))
    changed=candidate(.9,.9);changed['cohorts']['legacy']['cohort']='different'
    with pytest.raises(ValueError,match='exact same'):m.can_promote_mixed(changed,old)


def test_recipe_and_missing_cohort_qualification_are_explicit():
    with pytest.raises(ValueError,match='sum to one'):m.Recipe(legacy_weight=.2,behavior_weight=.2)
    with pytest.raises(ValueError,match='cpu'):m.Recipe(device='cuda:0')
    rows=[m.EpisodeRef('legacy','somewhere','hash','train','g','straight_recovery',20)]
    report=m.qualification(rows)
    assert not report['ready']
    assert report['missing_required_cohorts']==['legacy:validation','behavior:train','behavior:validation']
    assert report['recipe']['legacy_weight']==report['recipe']['behavior_weight']==.5


def test_discovery_replays_bound_supplement_split_not_just_manifest(artifacts):
    from test_behavior_admission import split_fixture
    from lane_behavior.admission import verify_split
    root=artifacts[0].parents[2]
    (root/'training-index.partial.json').write_text(json.dumps(dict(episodes=[str(artifacts[0].relative_to(root))],
        config_sha256='config',complete=False)))
    task,assignment=split_fixture(root,baseline_road='1',candidate_road='2')
    assignment['split']='train'
    task_path=artifacts[1].parent/'candidate-task.json';task_path.write_text(json.dumps(task))
    assignment['candidate_task_sha256']=sha256_file(task_path)
    # The supplement is not yet published while its independent split proof is
    # produced. Only this synthetic test temporarily moves its own manifest.
    hidden=artifacts[1].with_suffix('.fixture');artifacts[1].rename(hidden)
    audit=verify_split(root,task,sha256_file(task_path),assignment)
    hidden.rename(artifacts[1]);assert audit['passed']
    (artifacts[1].parent/'behavior-split.json').write_text(json.dumps(audit))
    refresh(artifacts[1])
    rows,complete=m.discover(root)
    assert {r.kind for r in rows}=={'legacy','behavior'} and not complete
    assert not m.qualification(rows)['ready']
    # New immutable baseline versions must not rewrite/revoke the original
    # valid admission simply because collection has continued concurrently.
    original_receipt=(artifacts[1].parent/'behavior-split.json').read_bytes()
    from lane_dataset.config import write_json
    current=json.loads((root/'splits.json').read_text());current['revision']=2
    newer=root/'splits-added.json';write_json(newer,current)
    write_json(root/'qualification.json',dict(split_binding=dict(path=str(newer),sha256=sha256_file(newer))))
    refreshed,_=m.discover(root,verify_payloads=False)
    assert refreshed==rows and (artifacts[1].parent/'behavior-split.json').read_bytes()==original_receipt
    # Rehashing a false assignment cannot override overlap with the frozen road.
    task['start']['road_id']='1';task['route']['section']['road_intervals'][0]['road_id']='1'
    task_path.write_text(json.dumps(task));refresh(artifacts[1])
    with pytest.raises(ValueError,match='exact-task'):m.discover(root,verify_payloads=False)


@pytest.mark.parametrize('change',['compatible_addition','conflicting_addition','prior_removal','original_file_changed','assignment_changed'])
def test_mixed_split_refresh_preserves_original_and_reaudits_current_geometry(tmp_path,change):
    from test_behavior_admission import split_fixture
    from lane_behavior.admission import verify_split
    from lane_dataset.config import write_json
    task,assignment=split_fixture(tmp_path)
    saved=verify_split(tmp_path,task,assignment['candidate_task_sha256'],assignment)
    assert saved['passed'];before=copy.deepcopy(saved)
    original=tmp_path/'splits.json';original_bytes=original.read_bytes()
    current=json.loads(original.read_text())
    if change in ('compatible_addition','conflicting_addition'):
        group=copy.deepcopy(current['groups'][0]);group['group_id']='new'
        group['sections'][0]['section_id']='new'
        group['sections'][0]['road_intervals'][0]['road_id']='3' if change=='compatible_addition' else '2'
        current['groups'].append(group)
        current['assignments'].append(dict(current['assignments'][0],group_id='new'))
    elif change=='prior_removal':current['groups']=[];current['assignments']=[]
    elif change=='original_file_changed':original.write_text('{}')
    elif change=='assignment_changed':saved['assignment']['candidate_task_sha256']='b'*64
    newer=tmp_path/'splits-new.json';write_json(newer,current)
    write_json(tmp_path/'qualification.json',dict(split_binding=dict(path=str(newer),sha256=sha256_file(newer))))
    if change=='compatible_addition':
        audit=m.refresh_behavior_split(tmp_path,task,assignment['candidate_task_sha256'],saved,'validation')
        assert audit['passed'] and audit['baseline_sha256']==sha256_file(newer)
        assert saved==before and original.read_bytes()==original_bytes
    else:
        with pytest.raises(ValueError):m.refresh_behavior_split(tmp_path,task,'a'*64,saved,'validation')


def test_no_admitted_behavior_starts_no_model_or_run_directory(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'discover',lambda _: ([],False))
    monkeypatch.setattr(runner,'create_model',lambda:pytest.fail('model creation before admission'))
    destination=tmp_path/'new-run'
    with pytest.raises(ValueError,match='not ready'):runner.run_bounded(tmp_path,destination,steps=1)
    assert not destination.exists()
    with pytest.raises(FileExistsError,match='unused'):runner.run_bounded(tmp_path,tmp_path,steps=1)


def test_cli_defaults_to_read_only_qualification(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(runner,'discover',lambda _: ([],False))
    monkeypatch.setattr(runner,'run_bounded',lambda *a,**k:pytest.fail('dry qualification started training'))
    assert runner.main(['--corpus',str(tmp_path)])==2
    report=json.loads(capsys.readouterr().out)
    assert not report['ready'] and not report['optimizer_step_performed']
    assert not list(tmp_path.iterdir())


def test_checkpoint_and_best_artifacts_never_overwritten_by_worse_step(tmp_path):
    student,_=models();optimizer=torch.optim.SGD(student.parameters(),lr=.01)
    old=runner._save(tmp_path,student,optimizer,dict(step=0))
    winner,promoted=runner._publish_validation(tmp_path,old,candidate(),None)
    assert promoted
    best_bytes=(tmp_path/'best.json').read_bytes();old_bytes=(tmp_path/old['checkpoint']).read_bytes()
    with torch.no_grad():student.policy.value.add_(.1)
    new=runner._save(tmp_path,student,optimizer,dict(step=1))
    same,promoted=runner._publish_validation(tmp_path,new,candidate(.1,1.1),winner)
    assert not promoted and same==winner
    assert (tmp_path/'best.json').read_bytes()==best_bytes
    assert (tmp_path/old['checkpoint']).read_bytes()==old_bytes
    assert (tmp_path/new['checkpoint']).is_file()
    assert old['checkpoint']!=new['checkpoint']
    with pytest.raises(ValueError,match='immutable artifact differs'):
        runner._save(tmp_path,student,optimizer,dict(step=0))
    assert (tmp_path/old['checkpoint']).read_bytes()==old_bytes


def test_behavior_lineage_keeps_canonical_review_hold_identity(tmp_path):
    row=m.EpisodeRef('behavior',str(tmp_path/'accepted'/'behavior-one'/'behavior-manifest.json'),
        'sha','train','g','straight_recovery',20)
    known,counts=runner._lineage({},[row]);assert known[row.key]['id'].endswith('manifest.json')
    state=dict(known=known,counts={'sha':1})
    (tmp_path/'reports').mkdir()
    (tmp_path/'reports'/'review-holds.json').write_text(json.dumps(dict(schema_version=1,
        episodes={'behavior-one':dict(status='held',reasons=['review'])})))
    with pytest.raises(ReviewHoldError):runner.assert_checkpoint_not_held(tmp_path,state)
    with pytest.raises(ValueError,match='untraceable'):
        runner._lineage(dict(known={'other':{'sha256':'other'}},counts={'missing':1}),[])


def test_bounded_runner_real_mixed_backward_and_both_forward_validation_cohorts(artifacts,monkeypatch):
    import shutil
    import lane_training.production as production
    legacy_targets(monkeypatch)
    monkeypatch.setattr(production,'episode_corridors',lambda _: [replace(straight(),label_index=100)])
    root=artifacts[0].parents[2]
    paths=list(artifacts)
    for path in artifacts:
        destination=path.parent.with_name(path.parent.name+'-validation')
        shutil.copytree(path.parent,destination)
        new=destination/path.name
        value=json.loads(new.read_text());value['split']='validation';new.write_text(json.dumps(value))
        paths.append(new)
    rows=[]
    for i,path in enumerate(paths):
        kind='legacy' if path.name=='manifest.json' else 'behavior'
        split='train' if i<2 else 'validation'
        rows.append(m.EpisodeRef(kind,str(path),sha256_file(path),split,str(i),'straight_recovery',1 if kind=='legacy' else 2))
    monkeypatch.setattr(runner,'discover',lambda _: (rows,False))
    # The production differentiable model is replaced only by a two-parameter
    # CD210-ABI model. Loaders, both real backward functions, actual optimizer,
    # heldout losses, checkpoint store, and promotion all execute unmodified.
    monkeypatch.setattr(runner,'create_model',lambda:models()[0])
    monkeypatch.setattr(runner,'source_bindings',lambda:{'synthetic-test-source':'fixed'})
    destination=root/'mixed-run'
    result=runner.run_bounded(root,destination,steps=1,recipe=m.Recipe(vision_lr=.01,policy_lr=.01))
    assert result['step']==1 and not result['automatic_follow_enabled']
    assert len(list((destination/'steps').glob('*.json')))==2
    latest=torch.load(destination/result['latest']['checkpoint'],weights_only=True)
    assert latest['step']==1 and latest['training_state']['sample_counts']=={rows[0].key:1,rows[1].key:1}
    assert set(latest['training_state']['last_validation']['cohorts'])==set(m.KINDS)
    assert not latest['training_state']['last_validation']['test_split_used']
