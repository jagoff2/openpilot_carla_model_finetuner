import copy
import json

import numpy as np
import pytest
import torch

from lane_behavior.loader import AdmittedBehaviorSequence, ADMISSION_SCHEMA, SCHEMA, REQUIRED_CHECKS
from lane_behavior.records import build_record,digest
from lane_behavior.target import ObservedRoad,ObservedObstacle
from lane_behavior.training import behavior_backward
from lane_dataset.config import sha256_file
from lane_dataset.geometry import EvidenceFrame,ObservableReference
from lane_dataset.preprocessing import PACKED_FRAME_SHAPE,reference_module
from lane_dataset.review_hold import ReviewHoldError
from lane_training.cd210 import CD210SplitModel


def write_fixture(corpus,*,count=311):
    root=corpus/'accepted'/'unmarked-test';root.mkdir(parents=True)
    frames=[dict(index=i,frame_id=1000+i,timestamp=i*.05,desire=[1,0,0,0,0,0,0,0],
        traffic_convention=[1,0],recorded_speed=999.,recorded_steering=999.) for i in range(count)]
    (root/'frames.jsonl').write_text('\n'.join(json.dumps(f) for f in frames))
    np.savez(root/'labels.npz',label_indices=np.array([100,110],np.int64))
    archives={};archive=reference_module('frame_archive_v2')
    for camera,base in (('narrow',10),('wide',80)):
        path=root/f'{camera}.zip'
        with archive.FrameArchiveWriter(path,dtype=np.uint8,frame_shape=PACKED_FRAME_SHAPE,expected_count=count) as writer:
            for i in range(count):writer.write(i,np.full(PACKED_FRAME_SHAPE,(base+i)%256,np.uint8))
            writer.finalize(complete=True)
        archives[camera]=dict(path=path.name,sha256=sha256_file(path))
    (root/'road-proof.json').write_text(json.dumps(dict(passed=True,fixture='synthetic-unit-test-only')))
    (root/'obstacle-proof.json').write_text(json.dumps(dict(passed=True,fixture='synthetic-unit-test-only')))
    records=[]
    for i in (100,110):
        evidence=(EvidenceFrame(1000+i,i*.05,'wide'),)
        x=np.arange(-10.,61.)
        road=ObservedRoad(ObservableReference(np.c_[x,x*0-4.5,x*0],np.c_[x,x*0+4.5,x*0],
            x,[evidence]*len(x),i*.05))
        bindings={'road-proof.json':sha256_file(root/'road-proof.json')}
        obstacles=()
        if i==110:
            obstacles=(ObservedObstacle('parked',[[20,3.2],[24,3.2],[24,4.8],[20,4.8]],'parked_vehicle',evidence,'static'),)
            bindings['obstacle-proof.json']=sha256_file(root/'obstacle-proof.json')
        value=build_record(road,obstacles,label_frame_id=1000+i,label_timestamp_s=i*.05,
            evidence_bindings=bindings,ego_position_world=[0.,2.25,0.],ego_yaw_rad=0.)
        assert value['moving_supervision']
        records.append(value)
    (root/'behavior-records.json').write_text(json.dumps(records))
    manifest=dict(schema=SCHEMA,status='accepted',training_ready=True,capture_scope='production',split='train',
        accepted_label_count=2,frames_file='frames.jsonl',labels_file='labels.npz',behavior_records_file='behavior-records.json',
        admission_file='admission.json',archives=archives,files={})
    report=dict(schema=ADMISSION_SCHEMA,passed=True,method='independent_raw_evidence_and_physical_replay',
        checks={k:True for k in REQUIRED_CHECKS},validated_records=[dict(label_index=i,record_sha256=r['record_sha256'],
            verified_evidence_bindings=r['evidence_bindings']) for i,r in zip((100,110),records)])
    path=root/'behavior-manifest.json'
    refresh(path,manifest,report)
    return path


def refresh(path,manifest=None,report=None):
    root=path.parent
    manifest=manifest or json.loads(path.read_text())
    report=report or json.loads((root/'admission.json').read_text())
    inputs={p.name:sha256_file(p) for p in root.iterdir() if p.is_file() and p.name not in ('behavior-manifest.json','admission.json')}
    report['input_files']=inputs
    (root/'admission.json').write_text(json.dumps(report))
    manifest['files']={**inputs,'admission.json':sha256_file(root/'admission.json')}
    path.write_text(json.dumps(manifest))


@pytest.fixture
def artifact(tmp_path):
    return write_fixture(tmp_path)


class Vision(torch.nn.Module):
    def __init__(self,value):
        super().__init__();self.value=torch.nn.Parameter(torch.tensor(value))
    def forward(self,inputs):
        assert set(inputs)=={'img','big_img'}
        feature=self.value+inputs['img'][0,-1,0,0].float()/255*.01
        return torch.cat((torch.zeros((1,1064)),feature.expand(1,512)),dim=1)


class Policy(torch.nn.Module):
    def __init__(self,value):
        super().__init__();self.value=torch.nn.Parameter(torch.tensor(value));self.history=[]
    def forward(self,inputs):
        assert set(inputs)=={'features_buffer','desire_pulse','traffic_convention'}
        assert inputs['features_buffer'].shape==(1,25,512)
        features=inputs['features_buffer']
        self.history.append(float(features[0,0,0].detach()))
        y=self.value+.2*features[0,-1,0]+.01*features[0,:-1,0].mean()
        x=torch.linspace(0.,40.,33)
        plan=torch.stack([x,y.expand(33),*([torch.zeros(33)]*13)],dim=1).reshape(1,495)
        return torch.cat([plan,torch.zeros((1,505))],dim=1)


def models():
    student=CD210SplitModel(Vision(.2),Policy(.5))
    original=CD210SplitModel(Vision(0.),Policy(0.))
    original.requires_grad_(False)
    return student,original


def test_cd210_behavior_backward_reaches_vision_and_policy_without_weight_updates(artifact):
    student,original=models()
    before={name:value.clone() for name,value in student.state_dict().items()}
    original_before={name:value.clone() for name,value in original.state_dict().items()}
    with AdmittedBehaviorSequence(artifact) as episode:
        assert len(episode.behavior_labels)==2
        assert episode.behavior_labels[0].corridor.obstacles==()
        assert len(episode.behavior_labels[1].corridor.obstacles)==1
        predictions,report=behavior_backward(episode,student,original)
        assert predictions.shape==(2,1512)
        assert report['backward_calls']==report['valid_labels']==2
        assert [r['label_index'] for r in report['label_diagnostics']]==[100,110]
        assert report['label_diagnostics'][0]['obstacle_rows']==0
        assert report['label_diagnostics'][1]['obstacle_rows']==1
        assert report['student_ticks']==report['reference_ticks']==111
        assert not report['reference_has_gradients'] and not report['optimizer_step_performed']
        assert set(episode.selected_targets())=={'label_indices'}
    assert student.vision.value.grad is not None and abs(student.vision.value.grad)>0
    assert student.policy.value.grad is not None and abs(student.policy.value.grad)>0
    assert student.policy.history[100]!=original.policy.history[100]
    for name,value in student.state_dict().items():assert torch.equal(value,before[name])
    for name,value in original.state_dict().items():assert torch.equal(value,original_before[name])
    assert all(p.grad is None for p in original.parameters())


@pytest.mark.parametrize('change',[dict(status='development'),dict(training_ready=False),dict(capture_scope='development')])
def test_development_capture_never_reaches_behavior_training(artifact,change):
    manifest=json.loads(artifact.read_text());manifest.update(change);artifact.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='development captures reject'):AdmittedBehaviorSequence(artifact)


def test_missing_independent_admission_or_evidence_binding_rejected(artifact):
    report=json.loads((artifact.parent/'admission.json').read_text());report['checks']['physical_quality']=False
    refresh(artifact,report=report)
    with pytest.raises(ValueError,match='admission checks'):AdmittedBehaviorSequence(artifact)
    report['checks']['physical_quality']=True
    report['validated_records'][0]['verified_evidence_bindings']={'road-proof.json':'0'*64}
    refresh(artifact,report=report)
    with pytest.raises(ValueError,match='evidence bindings'):AdmittedBehaviorSequence(artifact)


def test_hash_corruption_and_future_record_rejected(artifact):
    proof=artifact.parent/'road-proof.json';proof.write_text('{}')
    with pytest.raises(ValueError,match='file hash mismatch'):AdmittedBehaviorSequence(artifact)
    # Restore binding hashes, then attempt a self-consistently hashed future claim.
    proof.write_text(json.dumps(dict(passed=True,fixture='synthetic-unit-test-only')))
    records=json.loads((artifact.parent/'behavior-records.json').read_text())
    records[0]['road']['evidence_frames'][0][0].update(frame_id=1111,timestamp_s=5.55)
    records[0]['record_sha256']=digest({k:v for k,v in records[0].items() if k!='record_sha256'})
    (artifact.parent/'behavior-records.json').write_text(json.dumps(records))
    report=json.loads((artifact.parent/'admission.json').read_text())
    report['validated_records'][0]['record_sha256']=records[0]['record_sha256']
    refresh(artifact,report=report)
    with pytest.raises(ValueError,match='future, stale'):AdmittedBehaviorSequence(artifact)


def test_retention_and_exact_record_index_mapping_rejected(artifact):
    report=json.loads((artifact.parent/'admission.json').read_text())
    report['validated_records'][0]['label_index']=110
    refresh(artifact,report=report)
    with pytest.raises(ValueError,match='mapping'):AdmittedBehaviorSequence(artifact)
    report['validated_records'][0]['label_index']=100
    np.savez(artifact.parent/'labels.npz',label_indices=np.array([90,110]))
    refresh(artifact,report=report)
    with pytest.raises(ValueError,match='five seconds history'):AdmittedBehaviorSequence(artifact)


def test_review_hold_live_guard_and_shared_storage_refusal(artifact):
    with AdmittedBehaviorSequence(artifact) as episode:
        student,original=models()
        student.register_buffer('shared',torch.ones(1));original.register_buffer('shared',student.shared)
        with pytest.raises(ValueError,match='share parameter or buffer'):behavior_backward(episode,student,original)
        corpus=artifact.parents[2];(corpus/'reports').mkdir()
        (corpus/'reports/review-holds.json').write_text(json.dumps(dict(schema_version=1,
            episodes={'unmarked-test':dict(status='held',reasons=['review pending'])})))
        with pytest.raises(ReviewHoldError):episode.frame(0)
        with pytest.raises(ReviewHoldError):episode.assert_admitted()


def test_late_hold_clears_partial_gradients(artifact):
    student,original=models()
    corpus=artifact.parents[2];(corpus/'reports').mkdir()
    def progress(phase,tick,total):
        if phase=='student' and tick==200:
            (corpus/'reports/review-holds.json').write_text(json.dumps(dict(schema_version=1,
                episodes={'unmarked-test':dict(status='held',reasons=['late hold'])})))
    with AdmittedBehaviorSequence(artifact) as episode:
        with pytest.raises(ReviewHoldError):behavior_backward(episode,student,original,progress=progress,complete_sequence=True)
    assert all(p.grad is None for p in student.parameters())
