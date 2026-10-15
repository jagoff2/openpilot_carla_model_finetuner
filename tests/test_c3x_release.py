"""Checkpoint publication, native ABI preservation, and device export gates."""
import json
from pathlib import Path
import pickle
import subprocess

import numpy as np
import onnx
import pytest
import torch

from lane_release import export
from lane_release.device_build import assess,check_files
from lane_training.cd210 import DEFAULT_CHECKPOINT
from lane_training.cd210_model import create_model


def pointer(root,name,data,step):
    (root/name).write_bytes(data)
    value=dict(file=name,sha256=export.sha(root/name),step=step)
    (root/'resume.json').write_text(json.dumps(value))
    return value


def test_checkpoint_snapshot_retries_concurrent_publication(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir();pointer(run,'checkpoint-0.pt',b'old',7)
    real=export.shutil.copyfile;calls=[]
    def copying(source,dest):
        result=real(source,dest);calls.append(source)
        if len(calls)==1:pointer(run,'checkpoint-1.pt',b'committed-new',8)
        return result
    monkeypatch.setattr(export.shutil,'copyfile',copying)
    result=export.snapshot_pointer(run,run/'resume.json',tmp_path/'copy.pt')
    assert result['step']==8 and (tmp_path/'copy.pt').read_bytes()==b'committed-new'
    assert len(calls)==2


def test_checkpoint_pointer_cannot_escape_run(tmp_path):
    (tmp_path/'resume.json').write_text(json.dumps(dict(file='../bad.pt',sha256='x',step=1)))
    with pytest.raises(ValueError,match='unexpected checkpoint'):
        export.snapshot_pointer(tmp_path,tmp_path/'resume.json',tmp_path/'copy.pt')


def test_corrupt_checkpoint_never_snapshots(tmp_path,monkeypatch):
    pointer(tmp_path,'checkpoint-0.pt',b'saved',1)
    (tmp_path/'checkpoint-0.pt').write_bytes(b'corrupt')
    monkeypatch.setattr(export.time,'sleep',lambda _:None)
    with pytest.raises(RuntimeError,match='consistent'):
        export.snapshot_pointer(tmp_path,tmp_path/'resume.json',tmp_path/'copy.pt')
    assert not (tmp_path/'copy.pt').exists()


def test_previous_slot_excludes_uncommitted_newer_update(tmp_path):
    output=tmp_path/'out';output.mkdir()
    torch.save(dict(step=9),tmp_path/'checkpoint-0.pt')
    latest=dict(step=8,source_checkpoint_sha256='latest')
    assert list(export.snapshot_previous(tmp_path,latest,output))==[]
    torch.save(dict(step=7),tmp_path/'checkpoint-0.pt')
    previous=list(export.snapshot_previous(tmp_path,latest,output))
    assert len(previous)==1 and previous[0]['step']==7


def test_selection_uses_score_not_latest_and_can_retain_original():
    latest=dict(step=8,loss=2.6);earlier=dict(step=7,loss=2.4);original=dict(step=0,loss=2.5)
    assert export.choose([latest,earlier,original]) is earlier
    assert export.choose([latest,original]) is original
    assert export.choose([dict(step=1,loss=2.5),original])['step']==0
    with pytest.raises(ValueError,match='finite'):export.choose([dict(step=9,loss=float('nan'))])


def test_payload_tampering_and_path_escape_are_rejected(tmp_path):
    payload=tmp_path/'model.onnx';payload.write_bytes(b'model')
    manifest=dict(files={'model.onnx':export.sha(payload)})
    check_files(tmp_path,manifest)
    payload.write_bytes(b'modified')
    with pytest.raises(ValueError,match='integrity'):check_files(tmp_path,manifest)
    with pytest.raises(ValueError,match='integrity'):check_files(tmp_path,dict(files={'../escape':'hash'}))


def test_cpu_checks_cannot_claim_qcom_readiness():
    data=np.zeros((12,1000),np.float32)
    result=assess(data,data,[.01]*12,backend='CPU')
    assert result['passed'] and not result['runtime_ready']
    device=assess(data,data,[.01]*12,backend='QCOM')
    assert device['runtime_ready'] and not device['driving_qualified']


def test_mixed_precision_accepts_relative_forward_drift_but_checks_lateral_and_all_heads():
    reference=np.zeros((12,1000),np.float32)
    reference[:,0]=100.
    actual=reference.copy();actual[:,0]+=.7
    assert not assess(reference,actual,[.02]*12,backend='QCOM')['passed']
    assert assess(reference,actual,[.02]*12,backend='QCOM',lateral_precision=True)['passed']
    actual[:,1]=.3
    assert not assess(reference,actual,[.02]*12,backend='QCOM',lateral_precision=True)['passed']
    actual[:,1]=0.;actual[:,990]=.5
    assert not assess(reference,actual,[.02]*12,backend='QCOM',lateral_precision=True)['passed']


@pytest.mark.parametrize('failure',['position','yaw','other_head','latency','nan','negative_time'])
def test_runtime_gate_rejects_bad_predictions_or_timings(failure):
    reference=np.zeros((12,1000),np.float32);actual=reference.copy();timings=[.01]*12
    if failure=='position':actual[:,0]=.3
    if failure=='yaw':actual[:,11]=.03
    if failure=='other_head':actual[:,990]=.5
    if failure=='latency':timings=[.06]*12
    if failure=='nan':actual[:,0]=np.nan
    if failure=='negative_time':timings[5]=-.01
    result=assess(reference,actual,timings,backend='QCOM')
    assert not result['runtime_ready']
    json.dumps(result,allow_nan=False)


def test_offline_device_retains_verified_package_and_resume(tmp_path,monkeypatch):
    data=tmp_path/'graph.onnx';data.write_bytes(b'verified')
    (tmp_path/'payload.json').write_text(json.dumps(dict(files={data.name:export.sha(data)})))
    def offline(*args,**kwargs):raise subprocess.TimeoutExpired(args[0],20)
    monkeypatch.setattr(export.subprocess,'run',offline)
    with pytest.raises(RuntimeError,match='verified ONNX retained'):
        export.compile_remote(tmp_path,export.DEFAULT_DEVICE)
    pending=json.loads((tmp_path/'device-pending.json').read_text())
    assert pending['device']=='iq@192.168.1.95' and not pending['runtime_ready']
    assert '--resume' in pending['resume_command'] and data.read_bytes()==b'verified'


def test_remote_command_quotes_arguments_and_rejects_ssh_options():
    result=export.ssh_command('iq@192.168.1.95',['python','a file.py','literal; $(touch x)'])
    assert result[-1]=="python 'a file.py' 'literal; $(touch x)'"
    with pytest.raises(ValueError):export.ssh_command('-oProxyCommand=bad',['true'])


def test_export_preserves_native_shapes_dtypes_and_slices(tmp_path):
    model=create_model().eval()
    selected=dict(source_checkpoint_sha256='a'*64,step=9)
    roles=export.export_models(model,selected,tmp_path)
    report=json.loads((tmp_path/'export.json').read_text())
    for role,entry in roles.items():
        graph=onnx.load(tmp_path/entry['onnx']);source=getattr(model,role).graph
        assert graph.graph.input==source.graph.input and graph.graph.output==source.graph.output
        assert graph.graph.node==source.graph.node
        props={p.key:p.value for p in graph.metadata_props}
        assert props['model_checkpoint']=='local/cd210-centerline/'+'a'*64
        assert props['fine_tuning_step']=='9'
        metadata=pickle.loads((tmp_path/entry['metadata']).read_bytes())
        original=pickle.loads((DEFAULT_CHECKPOINT.parent/f'driving_{role}_c210m_metadata.pkl').read_bytes())
        for key in original.keys()-{'model_checkpoint'}:
            assert metadata[key]==original[key]
        assert metadata['model_checkpoint']==props['model_checkpoint']
        assert report['exports'][role]['sha256']==export.sha(tmp_path/entry['onnx'])==entry['onnx_sha256']
        assert entry['artifact'].endswith('_tinygrad.pkl')
    bundle=export.bundle_template(roles,selected)
    assert bundle['is20hz'] and bundle['generation']==12 and bundle['runner']=='tinygrad'
    assert {r['type'] for r in bundle['models']}=={'vision','policy'}
