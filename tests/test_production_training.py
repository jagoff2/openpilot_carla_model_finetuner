from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from lane_training import production as p


def corpus(tmp_path,splits=('train','validation','test')):
    paths = []
    for i,split in enumerate(splits):
        path = tmp_path/'accepted'/f'episode-{i}'/'manifest.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(status='accepted',admission={'passed':True},split=split,
            geometry_group_id=f'g-{i}',family='turn',accepted_label_count=20,config_sha256='config')))
        paths.append(str(path.relative_to(tmp_path)))
    index = tmp_path/'training-index.partial.json'
    index.write_text(json.dumps(dict(episodes=paths,config_sha256='config',complete=False)))
    return index


def test_following_index_keeps_validation_test_out_and_limits_early_replay(tmp_path):
    index = corpus(tmp_path)
    rows,complete = p.discover(tmp_path)
    assert not complete
    first = p.select_next(rows,{},4)
    assert first['split']=='train'
    assert p.select_next(rows,{first['sha256']:4},4) is None
    new = tmp_path/'accepted/new/manifest.json';new.parent.mkdir()
    data = json.loads(Path(first['path']).read_text());data['geometry_group_id']='new-group'
    new.write_text(json.dumps(data))
    listing=json.loads(index.read_text());listing['episodes'].append(str(new.relative_to(tmp_path)))
    index.write_text(json.dumps(listing))
    expanded,_=p.discover(tmp_path,{r['id']:r for r in rows})
    assert p.select_next(expanded,{first['sha256']:1},4)['path']==str(new.resolve())


def test_modified_accepted_manifests_and_cross_split_geometry_are_rejected(tmp_path):
    corpus(tmp_path)
    rows,_=p.discover(tmp_path)
    path=Path(rows[0]['path']);m=json.loads(path.read_text());m['appearance']=99
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='manifest changed'):
        p.discover(tmp_path,{r['id']:r for r in rows})
    m['geometry_group_id']=rows[1]['group'];path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='leaks'):
        p.discover(tmp_path)


def test_index_path_escape_and_missing_previously_seen_entry_are_rejected(tmp_path):
    index=corpus(tmp_path)
    rows,_=p.discover(tmp_path)
    listing=json.loads(index.read_text());listing['episodes']=listing['episodes'][1:]
    index.write_text(json.dumps(listing))
    with pytest.raises(ValueError,match='lost previously'):
        p.discover(tmp_path,{r['id']:r for r in rows})
    listing['episodes']=['../escape.json'];index.write_text(json.dumps(listing))
    with pytest.raises(ValueError,match='inside the corpus'):
        p.discover(tmp_path)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision=torch.nn.Linear(1,1)
        self.policy=torch.nn.Linear(1,1)


def test_checkpoint_crash_before_pointer_preserves_consistent_previous_step(tmp_path,monkeypatch):
    cfg=p.TrainingConfig(corpus=str(tmp_path))
    model=Tiny();optimizer=p.make_optimizer(model,cfg)
    state=dict(step=0,config=asdict(cfg),counts={})
    p.commit(tmp_path,model,optimizer,state)
    old=json.loads((tmp_path/'resume.json').read_text())
    state['step']=1;state['counts']['some-manifest']=1
    with torch.no_grad():model.vision.weight.add_(1)
    real_atomic=p.atomic_json
    monkeypatch.setattr(p,'atomic_json',lambda *args: (_ for _ in ()).throw(OSError('crash before pointer')))
    with pytest.raises(OSError):p.commit(tmp_path,model,optimizer,state)
    assert json.loads((tmp_path/'resume.json').read_text())==old
    package=torch.load(tmp_path/old['file'],weights_only=True)
    assert package['step']==0 and package['training_state']['counts']=={}
    monkeypatch.setattr(p,'atomic_json',real_atomic)
    new=p.commit(tmp_path,model,optimizer,state)
    assert new['slot']!=old['slot']
    package=torch.load(tmp_path/new['file'],weights_only=True)
    assert package['step']==1 and package['training_state']['counts']=={'some-manifest':1}


def test_double_writer_is_rejected_and_lock_releases(tmp_path):
    with p.run_lock(tmp_path):
        with pytest.raises(RuntimeError,match='another trainer'):
            with p.run_lock(tmp_path):pass
    with p.run_lock(tmp_path):pass


def test_validation_is_forward_only_and_never_touches_optimizer(monkeypatch):
    from test_cd210_training import Episode,model
    from test_centerline_objective import straight
    from dataclasses import replace
    episode=Episode()
    monkeypatch.setattr(p,'episode_corridors',lambda _: [replace(straight(),label_index=i) for i in (100,104)])
    student=model(2.);original=model().requires_grad_(False)
    before=p.parameter_digest(student)
    result=p.evaluate_episode(episode,student,original)
    assert result['labels']==2 and result['loss']>0
    assert p.parameter_digest(student)==before
    assert all(x.grad is None for x in student.parameters())
    assert all(x.grad is None for x in original.parameters())


def test_config_keeps_both_collection_gpus_reserved():
    with pytest.raises(ValueError,match='both GPUs'):
        p.TrainingConfig(corpus='x',device='cuda:0')
    with pytest.raises(ValueError,match='hyperparameters'):
        p.TrainingConfig(corpus='x',policy_lr=float('nan'))


def test_resume_restores_optimizer_and_sampling_state(tmp_path,monkeypatch):
    model=Tiny();cfg=p.TrainingConfig(corpus=str(tmp_path));optimizer=p.make_optimizer(model,cfg)
    (model.policy(model.vision(torch.ones(1,1)))**2).sum().backward();optimizer.step()
    state=dict(step=1,config=asdict(cfg),counts={'manifest':1})
    p.commit(tmp_path,model,optimizer,state)
    def loader(path):
        package=torch.load(path,weights_only=True);restored=Tiny();restored.load_state_dict(package['model'])
        return restored,package
    monkeypatch.setattr(p,'load_checkpoint',loader)
    restored,optim,saved=p.restore(tmp_path,cfg)
    assert p.parameter_digest(restored)==p.parameter_digest(model)
    assert saved['counts']=={'manifest':1} and saved['step']==1
    assert all(int(value['step'])==1 for value in optim.state.values())
    with pytest.raises(ValueError,match='configuration'):
        p.restore(tmp_path,p.TrainingConfig(corpus=str(tmp_path),passes=5))
