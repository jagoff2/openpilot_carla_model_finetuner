from dataclasses import replace

import numpy as np
import pytest
import torch

from lane_training.centerline import (LossConfig, OBJECTIVE_VERSION, recovery_target,
    centerline_loss, _correction, _derivative_maxima)
from lane_training import production as p
from test_centerline_objective import straight, curved
from test_production_training import Tiny


def shifted(corridor, offset=.25, angle=0.):
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return replace(corridor, left=corridor.left@rotation.T-[0, offset],
                   right=corridor.right@rotation.T-[0, offset])


def test_centered_aligned_lane_keeps_exact_observed_centerline():
    for corridor in (straight(), curved()):
        target = recovery_target(corridor)
        assert target.length == 0
        q = np.linspace(0, 20, 101)
        np.testing.assert_array_equal(target.sample(q)[0], corridor.sample(q)[0])


@pytest.mark.parametrize('offset,angle', [(.25, 0), (-.25, 0), (.25, .08), (-.25, -.08)])
def test_recovery_anchors_origin_and_heading_and_joins_observed_lane(offset, angle):
    corridor = shifted(straight(), offset, angle)
    target = recovery_target(corridor)
    assert target is not None and target.length > 0
    center, tangent, *_ = target.sample(np.array([0., target.length, target.length+10]))
    np.testing.assert_allclose(center[0], [0, 0], atol=1e-12)
    np.testing.assert_allclose(tangent[0], [1, 0], atol=1e-12)
    np.testing.assert_allclose(center[1:], corridor.sample(np.array([target.length, target.length+10]))[0])
    delta, derivative, second, _ = _correction(np.array([target.length]), target.length,
        target.offset, target.direction_delta)
    np.testing.assert_allclose(np.r_[delta.ravel(), derivative.ravel(), second.ravel()], 0, atol=1e-12)
    cfg = LossConfig()
    maxima = _derivative_maxima(target.length, target.offset, target.direction_delta)
    assert maxima[0] <= cfg.recovery_second_derivative_limit_inv_m
    assert maxima[1] <= cfg.recovery_third_derivative_limit_inv_m2
    dense = _correction(np.linspace(0, target.length, 10001), target.length,
        target.offset, target.direction_delta)
    assert np.linalg.norm(dense[2], axis=1).max() <= maxima[0]+1e-10
    assert np.linalg.norm(dense[3], axis=1).max() <= maxima[1]+1e-10


def test_feasible_recovery_beats_translated_lane_and_gradient_corrects_origin():
    corridor = shifted(straight())
    target = recovery_target(corridor)
    x = np.linspace(0, 20, 65)
    reference = torch.tensor(np.c_[x, x*0][None], dtype=torch.float32)
    good_xy, good_tangent, *_ = target.sample(x)
    good = torch.tensor(good_xy[None], dtype=torch.float32)
    yaw = torch.tensor(np.arctan2(good_tangent[:, 1], good_tangent[:, 0])[None], dtype=torch.float32)
    good_loss, info = centerline_loss(good, reference, [corridor], predicted_yaw=yaw)
    bad = torch.tensor(corridor.sample(x)[0][None], dtype=torch.float32, requires_grad=True)
    bad_loss, bad_info = centerline_loss(bad, reference, [corridor])
    bad_loss.backward()
    assert info['objective_version'] == OBJECTIVE_VERSION
    assert good_loss < .01
    assert bad_loss > 2
    assert bad_info['origin_loss'] > 1.9
    assert bad.grad[0, 0, 1] < 0  # Gradient descent moves translated y=-.25 back toward ego y=0.
    assert info['recovery_labels'] == 1


def test_offset_recovery_does_not_soften_far_curve_or_footprint_penalty():
    corridor = shifted(curved())
    target = recovery_target(corridor)
    assert target is not None
    q = np.linspace(target.length+1, 22, 100)
    np.testing.assert_array_equal(target.sample(q)[0], corridor.sample(q)[0])
    extent = 15*np.pi/2
    reference = torch.tensor([[[0., 0.], [extent, 0.]]])
    shortcut = torch.tensor([[[0., 0.], [15., 14.75]]], requires_grad=True)
    loss, info = centerline_loss(shortcut, reference, [corridor])
    loss.backward()
    assert info['mean_abs_lateral_m'] > 1
    assert info['max_clearance_violation_m'] > 1
    assert torch.isfinite(shortcut.grad).all()


def test_missing_or_gapped_recovery_support_cannot_invent_early_centering():
    corridor = shifted(straight(end=4.))
    assert recovery_target(corridor) is None
    x = torch.linspace(0, 10, 33)
    path = torch.stack([x, x*0], dim=-1)[None].requires_grad_()
    loss, info = centerline_loss(path, path.detach(), [corridor])
    loss.backward()
    assert info['valid_labels'] == 0 and info['unsupported_recovery_labels'] == 1
    assert path.grad.count_nonzero() == 0
    longer = shifted(straight())
    valid = longer.valid.copy()
    valid[np.searchsorted(longer.s, 3.)] = False
    assert recovery_target(replace(longer, valid=valid)) is None


def test_student_shortening_cannot_change_recovery_length_or_support():
    corridor = shifted(straight())
    x = np.linspace(0, 20, 33)
    reference = torch.tensor(np.c_[x, x*0][None], dtype=torch.float32)
    good = torch.tensor(recovery_target(corridor).sample(x)[0][None], dtype=torch.float32)
    _, normal = centerline_loss(good, reference, [corridor])
    loss, short = centerline_loss(good*.01, reference, [corridor])
    assert short['path_samples'] == normal['path_samples']
    assert short['recovery_length_max_m'] == normal['recovery_length_max_m']
    assert short['progress_loss'] > 5


def test_old_objective_run_cannot_resume_or_auto_migrate(tmp_path, monkeypatch):
    cfg = p.TrainingConfig(corpus=str(tmp_path))
    (tmp_path/'resume.json').write_text('{}')
    state = dict(step=12, counts={}, known={}, source_hashes={})
    monkeypatch.setattr(p, 'restore', lambda *_: (Tiny(), None, state))
    monkeypatch.setattr(p, 'commit', lambda *_: pytest.fail('old objective must not overwrite checkpoint'))
    with pytest.raises(ValueError, match='training objective changed'):
        p.Trainer(cfg, tmp_path)
    with pytest.raises(ValueError, match='automatic migration'):
        p.migrate_preservation(tmp_path, cfg, lateral_selection=True)
