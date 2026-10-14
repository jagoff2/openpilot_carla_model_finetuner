"""Real pinned CD210 graph regressions; CPU only, no collector interaction."""
import hashlib

import numpy as np
import pytest
import torch

onnx = pytest.importorskip('onnx')
pytest.importorskip('onnxruntime')

from lane_training.adapters import model_centerline_loss
from lane_training.cd210 import ROLLOUT_PLAN
from lane_training.cd210_model import create_model, save_checkpoint, load_checkpoint, export_onnx
from lane_training.fetch_cd210 import EXPORT_DIR, EXPORTS, verify_file
from lane_training.onnx_model import ReleasedModel
from lane_training.smoke import parameter_digest
from lane_training.verify_cd210 import float32_graph, runtime
from test_centerline_objective import straight


@pytest.fixture
def actual_model():
    if not all((EXPORT_DIR/v['file']).exists() for v in EXPORTS.values()):
        pytest.skip('fetch the pinned CD210 ONNX files with lane_training.fetch_cd210')
    return create_model().eval()


def inputs():
    rng = torch.Generator().manual_seed(210)
    return dict(img=torch.randint(0,256,(1,12,128,256),dtype=torch.uint8,generator=rng),
        big_img=torch.randint(0,256,(1,12,128,256),dtype=torch.uint8,generator=rng),
        desire_pulse=torch.zeros(1,25,8), traffic_convention=torch.tensor([[1.,0.]]),
        features_buffer=torch.randn(1,24,512,generator=rng)*.01)


def test_unverified_onnx_is_rejected_before_loading(tmp_path):
    bad = tmp_path/'unverified.onnx'
    bad.write_bytes(b'not the selected checkpoint')
    with pytest.raises(ValueError,match='hash mismatch'):
        ReleasedModel(bad, expected_sha256=EXPORTS['policy']['sha256'])
    with pytest.raises(ValueError,match='hash verification'):
        verify_file(bad, EXPORTS['policy'])


@pytest.mark.parametrize('role',['vision','policy'])
def test_actual_cd210_float32_matches_independent_onnx_runtime(actual_model, role):
    model = actual_model
    data = inputs()
    with torch.no_grad():
        vision = model.vision({k:data[k] for k in ('img','big_img')})
        data['features_buffer'] = torch.cat([data['features_buffer'],vision[:,None,1064:]],dim=1)
        component = getattr(model,role)
        selected = {k:data[k] for k in component.input_names}
        result = component(selected).numpy()
    reference = runtime(float32_graph(component.graph)).run(None,{k:v.numpy() for k,v in selected.items()})[0]
    assert np.allclose(result,reference,atol=2e-4,rtol=2e-4)


def test_actual_weights_take_centerline_step_and_roundtrip_without_changing_sources(actual_model, tmp_path):
    model = actual_model
    data = inputs()
    before = parameter_digest(model)
    prediction = model(data)
    original = prediction.detach().clone()
    corridor = straight(end=150.)
    loss, info = model_centerline_loss(prediction,original,[corridor],adapter=ROLLOUT_PLAN)
    assert info['valid_labels'] == 1 and loss > 0
    loss.backward()
    for role in ('vision','policy'):
        grads = [p.grad for p in getattr(model,role).parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(int(g.count_nonzero()) for g in grads) > 0
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-4)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step()
    with torch.no_grad():
        result = model(data)
        after_loss,_ = model_centerline_loss(result,original,[corridor],adapter=ROLLOUT_PLAN)
    assert after_loss < loss
    assert parameter_digest(model) != before
    path = tmp_path/'trained-test-copy.pt'
    save_checkpoint(model,path,optimizer=optimizer,step=1)
    restored, package = load_checkpoint(path)
    assert package['step'] == 1 and package['optimizer']['param_groups'][0]['lr'] == 1e-4
    assert parameter_digest(restored) == parameter_digest(model)
    with torch.no_grad():
        assert torch.equal(restored(data), result)
    exported = export_onnx(model,tmp_path/'onnx')
    for role,entry in EXPORTS.items():
        verify_file(EXPORT_DIR/entry['file'],entry)
        assert exported[role]['sha256'] != entry['sha256']
        emitted = onnx.load(exported[role]['path'])
        expected = getattr(model,role).graph
        assert emitted.graph.node == expected.graph.node
        assert emitted.metadata_props == expected.metadata_props
    with pytest.raises(ValueError,match='source checkpoint'):
        model.policy.export(EXPORT_DIR/EXPORTS['policy']['file'])


def test_checkpoint_with_wrong_model_identity_is_rejected(tmp_path):
    path = tmp_path/'wrong.pt'
    torch.save({'format':'another-model'},path)
    with pytest.raises(ValueError,match='selected CD210'):
        load_checkpoint(path)
