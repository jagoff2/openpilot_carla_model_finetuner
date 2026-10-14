from copy import deepcopy
from functools import partial

import numpy as np
import pytest
import torch

from lane_dataset.adapter import PLAN_MEAN_SLICE
from lane_dataset.observed_corridor import METHOD
from lane_training.centerline import LossConfig, centerline_loss as path_loss
from lane_training.adapters import FlatPlanAdapter, model_centerline_loss
from lane_training.targets import ObservedCorridor, corridor_from_record

# Explicit historical layout fixture, not a selected fine-tuning checkpoint.
LAYOUT = FlatPlanAdapter(2580, 917, 33, 15, yaw_column=11)
centerline_loss = partial(model_centerline_loss, adapter=LAYOUT)


def output(xy, yaw=None, grad=False):
    plan = torch.zeros(1, 33, 15)
    plan[0, :, :2] = torch.tensor(np.array(xy), dtype=torch.float32)
    if yaw is not None:
        plan[0, :, 11] = torch.tensor(np.array(yaw), dtype=torch.float32)
    result = torch.zeros(1, 2580)
    result[:, PLAN_MEAN_SLICE] = plan.reshape(1, -1)
    return result.requires_grad_(grad)


def straight(width=4., start=-5., end=40., valid=True):
    s = np.linspace(start, end, int((end-start)/.1)+1)
    return ObservedCorridor(s, np.c_[s, np.full_like(s, -width/2)],
        np.c_[s, np.full_like(s, width/2)], np.full(len(s)-1, valid, bool), 0.)


def curved(angle=np.pi/2, radius=15.):
    a = np.linspace(-.3, angle+.3, 1001)
    s = a*radius
    center = np.c_[radius*np.sin(a), radius*(1-np.cos(a))]
    normal = np.c_[-np.sin(a), np.cos(a)]
    return ObservedCorridor(s, center-2*normal, center+2*normal,
        np.ones(len(s)-1, bool), 0.)


def test_centered_straight_and_lateral_gradient_only_on_geometry_outputs():
    xy = np.c_[np.linspace(0, 30, 33), np.zeros(33)]
    original = output(xy, grad=True)
    centered, info = centerline_loss(original, original, [straight()])
    assert centered.item() < 1e-9
    shifted = output(xy+[0, .3], grad=True)
    loss, info = centerline_loss(shifted, original, [straight()])
    loss.backward()
    grad = shifted.grad[:, PLAN_MEAN_SLICE].reshape(33, 15)
    assert info['mean_abs_lateral_m'] == pytest.approx(.3, abs=1e-5)
    assert grad[:, 1].sum() > 0
    assert torch.count_nonzero(grad[:, 2:11]) == 0
    assert torch.count_nonzero(grad[:, 12:]) == 0
    assert torch.count_nonzero(shifted.grad[:, :917]) == 0
    assert original.grad is None


def test_progress_anchor_prevents_stationary_or_shortened_path_escape():
    xy = np.c_[np.linspace(0, 30, 33), np.zeros(33)]
    base = output(xy)
    good, _ = centerline_loss(base, base, [straight()])
    short = output(xy*.1, grad=True)
    stationary = output(xy*0, grad=True)
    for pred in (short, stationary):
        loss, info = centerline_loss(pred, base, [straight()])
        loss.backward()
        assert loss > good+5
        assert info['progress_loss'] > 5
        assert torch.isfinite(pred.grad).all()
        assert pred.grad[:, PLAN_MEAN_SLICE].reshape(33, 15)[:, 0].sum() < 0


def test_different_original_progresses_follow_same_geometry_without_recorded_speed():
    corridor = curved()
    for extent in (8., 20.):
        a = np.linspace(0, extent/15, 33)
        xy = np.c_[15*np.sin(a), 15*(1-np.cos(a))]
        pred = output(xy, a)
        _, info = centerline_loss(pred, pred, [corridor])
        assert info['mean_abs_lateral_m'] < .01
        assert info['progress_loss'] < .001


def test_hairpin_uses_ordered_arc_even_when_forward_x_decreases():
    a = np.linspace(0, np.pi+.3, 33)
    xy = np.c_[15*np.sin(a), 15*(1-np.cos(a))]
    assert np.diff(xy[:, 0]).min() < 0
    pred = output(xy, a)
    _, info = centerline_loss(pred, pred, [curved(np.pi+.3)])
    assert info['mean_abs_lateral_m'] < .03
    assert info['progress_loss'] < .01


def test_between_point_corner_cut_is_penalized_when_both_endpoints_are_centered():
    # Only one nonzero segment: its endpoints lie exactly on a quarter-circle.
    # The original model provides the corresponding route distance, not targets.
    length = 15*np.pi/2
    baseline = output(np.r_[[[0., 0.]], np.tile([length, 0.], (32, 1))])
    shortcut = output(np.r_[[[0., 0.]], np.tile([15., 15.], (32, 1))], grad=True)
    loss, info = centerline_loss(shortcut, baseline, [curved()])
    loss.backward()
    assert info['path_samples'] > 50
    assert info['mean_abs_lateral_m'] > 1
    assert info['max_clearance_violation_m'] > 1
    assert torch.isfinite(shortcut.grad).all()


def test_vehicle_body_penalty_detects_line_hugging_with_center_still_inside_lane():
    xy = np.c_[np.linspace(0, 30, 33), np.zeros(33)]
    _, info = centerline_loss(output(xy+[0, 1.]), output(xy), [straight()])
    assert info['max_clearance_violation_m'] == pytest.approx(.25, abs=1e-5)
    assert info['clearance_loss'] > 0


def test_fixed_support_student_cannot_escape_by_leaving_observed_geometry():
    xy = np.c_[np.linspace(0, 30, 33), np.zeros(33)]
    _, good = centerline_loss(output(xy), output(xy), [straight(end=15.)])
    loss, escaped = centerline_loss(output(xy+[100, 100]), output(xy), [straight(end=15.)])
    assert escaped['path_samples'] == good['path_samples']
    assert escaped['footprint_samples'] == good['footprint_samples']
    assert loss > 100


def test_empty_support_masks_nan_and_invalid_supported_values_fail():
    xy = np.c_[np.linspace(0, 30, 33), np.zeros(33)]
    pred = output(xy*np.nan, grad=True)
    loss, info = centerline_loss(pred, output(xy), [straight(valid=False)])
    loss.backward()
    assert loss == 0 and info['valid_labels'] == 0
    assert torch.isfinite(pred.grad).all() and pred.grad.count_nonzero() == 0
    with pytest.raises(ValueError, match='nonfinite supported'):
        centerline_loss(pred, output(xy), [straight()])


def test_optimizing_path_reduces_center_error_without_reducing_progress():
    xy = np.c_[np.linspace(0, 30, 33), np.zeros(33)]
    pred = output(xy+[0, .4], grad=True)
    base = output(xy)
    optimizer = torch.optim.Adam([pred], lr=.03)
    _, first = centerline_loss(pred, base, [straight()])
    for _ in range(35):
        optimizer.zero_grad()
        loss, _ = centerline_loss(pred, base, [straight()])
        loss.backward()
        optimizer.step()
    _, last = centerline_loss(pred, base, [straight()])
    assert last['mean_abs_lateral_m'] < first['mean_abs_lateral_m']*.15
    assert last['progress_loss'] < 1e-4


def record_fixture():
    x = np.arange(12.)
    geometry = dict(left_xyz=np.c_[x, x*0-2, x*0].tolist(), right_xyz=np.c_[x, x*0+2, x*0].tolist())
    indices = np.arange(8)
    source = np.r_[indices, indices+12]
    ref = dict(frame_id=100, timestamp_s=9., camera='wide')
    supported = np.zeros(24, bool); supported[source] = True
    refs = [[ref] if supported[i] else [] for i in range(24)]
    proof = dict(source_indices=indices.tolist(), source_left_indices=indices.tolist(),
        source_right_indices=indices.tolist(), chord_arc_m=indices.tolist(),
        interval_geometry_disagreement_m=[0.]*7, interval_geometry_valid=[True]*7,
        maximum_marking_gap_m=8., spline_evidence_frames=[ref], support_method=METHOD)
    record = dict(accepted=True, qa_passed=True, label_index=120, timestamp_s=10.,
        interpolated_corridor=proof, visibility=dict(supported=supported.tolist(), evidence_frames=refs))
    return geometry, record, [2., 0., 0.], [0., 0., 0.], dict(length=4.8, width=2.2), {100: 9.}


def test_hidden_geometry_and_map_derivatives_do_not_affect_targets():
    args = record_fixture()
    first = corridor_from_record(*args)
    altered = deepcopy(args)
    for key in ('left_xyz', 'right_xyz'):
        altered[0][key][8:] = [[float('nan')]*3]*4
    altered[0]['arc_m'] = [123456]*12
    altered[0]['center_xyz'] = [[999]*3]*12
    altered[0]['curvature_1pm'] = [999]*12
    second = corridor_from_record(*altered)
    for name in ('s', 'left', 'right', 'valid'):
        assert np.array_equal(getattr(first, name), getattr(second, name))


@pytest.mark.parametrize('fault', ['future', 'missing', 'unbound', 'stale', 'unsupported', 'gap'])
def test_target_rejects_bad_observation_evidence(fault):
    args = record_fixture(); record = args[1]
    ref = record['visibility']['evidence_frames'][0][0]
    if fault == 'future': ref['timestamp_s'] = 10.1
    if fault == 'stale': ref['timestamp_s'] = 4.
    if fault == 'unbound': ref['frame_id'] = 200
    if fault == 'missing': record['visibility']['evidence_frames'][0] = []
    if fault == 'unsupported': record['visibility']['supported'][0] = False
    if fault == 'gap': args[0]['left_xyz'][1][0] = 20.
    with pytest.raises(ValueError):
        corridor_from_record(*args)


def test_invalid_intervals_and_outside_extent_are_not_supervised():
    args = record_fixture()
    args[1]['interpolated_corridor']['interval_geometry_valid'][3] = False
    corridor = corridor_from_record(*args)
    assert not corridor.sample(np.array([3.5]))[-1][0]
    assert corridor.sample(np.array([2.5, 4.5]))[-1].all()
    assert not corridor.sample(np.array([-1., 8.]))[-1].any()


def test_cannot_disable_progress_protection():
    with pytest.raises(ValueError): LossConfig(progress_weight=0)


def test_nearly_coincident_dense_grid_and_observed_knots_do_not_create_zero_length_segments():
    args = record_fixture()
    for side in ('left_xyz', 'right_xyz'):
        for point in args[0][side]:
            point[0] *= 1+1e-13
    args[1]['interpolated_corridor']['chord_arc_m'] = (np.arange(8)*(1+1e-13)).tolist()
    corridor = corridor_from_record(*args)
    assert np.diff(corridor.s).min() > .1
    assert corridor.valid.all()
