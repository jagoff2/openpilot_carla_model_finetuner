from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from lane_training import centerline_loss
from lane_training.cd210 import CD210SplitModel, POLICY_PLAN, ROLLOUT_PLAN, backward, student_rollout, verify_checkpoint
from lane_training.adapters import model_centerline_loss
from test_centerline_objective import straight


class Episode:
    def __init__(self, count=105):
        self.frames = [dict(desire=[0,1,0,0,0,0,0,0] if i >= 100 else [1,0,0,0,0,0,0,0],
            traffic_convention=[1,0], action_t=[float('nan')]*2) for i in range(count)]
        self.labels = dict(label_indices=np.array([100,104]))
    def __len__(self): return len(self.frames)
    def frame(self, i): return np.full((2,6,128,256), i+1, np.uint8)
    def selected_targets(self): return self.labels


class Vision(torch.nn.Module):
    def __init__(self, bias=0.):
        super().__init__(); self.bias = torch.nn.Parameter(torch.tensor(bias))
        self.images = []
    def forward(self, inputs):
        assert set(inputs) == {'img','big_img'}
        assert inputs['img'].dtype == torch.uint8
        value = inputs['img'][0,6,0,0].float()+self.bias
        self.images.append((int(inputs['img'][0,0,0,0]), int(inputs['img'][0,6,0,0])))
        return value.expand(1,1576)


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.offset = torch.nn.Parameter(torch.tensor(.2))
        self.observed = []
    def forward(self, inputs):
        assert set(inputs) == {'desire_pulse','traffic_convention','features_buffer'}
        features = inputs['features_buffer']
        assert features.shape == (1,25,512)
        self.observed.append((features.detach().clone(), inputs['desire_pulse'].detach().clone()))
        plan = torch.zeros(1,33,15)
        plan[0,:,0] = torch.linspace(0,20,33)
        plan[0,:,1] = self.offset+.001*features[0,-1,0]
        return torch.cat([plan.reshape(1,495), torch.zeros(1,505)], dim=1)


def model(bias=0.): return CD210SplitModel(Vision(bias), Policy())


def test_cd210_uses_current_vision_plus_24_prior_features_and_stock_camera_spacing():
    network = model()
    prediction, _ = student_rollout(Episode(), network)
    assert prediction.shape == (2,1512)
    assert network.vision.images[0] == (0,1)
    assert network.vision.images[100] == (97,101)
    history, pulse = network.policy.observed[100]
    assert history[0,:,0].tolist() == list(range(5,102,4))
    assert pulse[0,-1,1] == 1 and pulse[:,:,0].count_nonzero() == 0
    assert network.policy.observed[104][1][0,-1,1] == 0
    loss, _ = model_centerline_loss(prediction, prediction.detach(), [straight(), straight()], adapter=ROLLOUT_PLAN)
    loss.backward()
    assert network.policy.offset.grad > 0
    assert network.vision.bias.grad > 0


def test_streamed_gradients_match_batch_and_each_model_uses_its_own_features(monkeypatch):
    import lane_training.rollout as integration
    episode = Episode()
    corridors = [replace(straight(), label_index=i) for i in (100,104)]
    monkeypatch.setattr(integration, 'episode_corridors', lambda episode: corridors)
    student = model(2.)
    reference = model(0.).requires_grad_(False)
    batch_student, batch_reference = deepcopy(student), deepcopy(reference)
    actual, report = backward(episode, student, reference)
    reference_prediction, _ = student_rollout(episode, batch_reference)
    prediction, _ = student_rollout(episode, batch_student)
    loss, _ = model_centerline_loss(prediction, reference_prediction, corridors, adapter=ROLLOUT_PLAN)
    loss.backward()
    assert torch.allclose(actual, prediction.detach())
    for (name,a), (_,b) in zip(student.named_parameters(), batch_student.named_parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-6), name
    assert report['student_ticks'] == 105 == report['reference_ticks']
    assert report['backward_calls'] == 2 and not report['reference_has_gradients']
    assert reference.policy.observed[100][0][0,0,0] == 5
    assert student.policy.observed[100][0][0,0,0] == 7


def test_shared_original_weights_are_rejected_before_rollout():
    network = model().requires_grad_(False)
    with pytest.raises(ValueError, match='separate frozen'):
        backward(Episode(), network, network)
    shared = deepcopy(network)
    shared.policy.offset = network.policy.offset
    with pytest.raises(ValueError, match='share parameter storage'):
        backward(Episode(), shared, network)


def test_cd210_output_layout_uses_policy_means_without_uncertainty_or_desire():
    raw = torch.arange(1000, dtype=torch.float32)[None].requires_grad_()
    plan = POLICY_PLAN(raw)
    assert plan.xy.shape == (1,33,2)
    assert plan.xy[0,-1].tolist() == [480.,481.]
    assert plan.yaw[0,-1] == 491
    plan.xy.sum().backward()
    assert raw.grad[:,495:].count_nonzero() == 0


@pytest.mark.parametrize('count', [9,17,65])
def test_canonical_loss_supports_other_model_point_counts(count):
    path = torch.stack([torch.linspace(0,30,count), torch.zeros(count)], dim=1)[None]
    loss, report = centerline_loss(path.requires_grad_(), path.detach(), [straight()])
    loss.backward()
    assert loss < 1e-8 and report['path_samples'] > count


def test_predicted_body_heading_changes_clearance_even_with_centered_positions():
    path = torch.stack([torch.linspace(0,30,33), torch.zeros(33)], dim=1)[None]
    yaw = torch.full((1,33), torch.pi/2, requires_grad=True)
    loss, report = centerline_loss(path, path, [straight()], predicted_yaw=yaw)
    loss.backward()
    assert report['max_clearance_violation_m'] == pytest.approx(.55, abs=1e-5)
    assert torch.isfinite(yaw.grad).all()


def test_checkpoint_verifier_refuses_unselected_bundle(tmp_path):
    path = tmp_path/'bundle.json'; path.write_text('{"internalName":"different-model","is20hz":true}')
    with pytest.raises(ValueError, match='CD210'):
        verify_checkpoint(path)
