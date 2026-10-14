import json

import pytest

from lane_training.checkpoint_store import (SELECTION_METRIC, can_promote,
    checkpoint_path, lateral_score, protect_validation, selection_report)
from lane_release.export import choose, select_best
from test_checkpoint_preservation import checkpoint


def geometry(lateral=1., heading=.4, clearance=.2, progress=0.):
    return dict(lateral_loss=lateral,heading_loss=heading,clearance_loss=clearance,
        progress_loss=progress)


def measured(value,progress,*,step=1):
    base=geometry(lateral=3.)
    actual=geometry(lateral=value,progress=progress)
    score=selection_report(lateral_score(actual)+progress,lateral_score(base),actual,base)
    return dict(score,step=step,cohort='same-frames',
        rows=[dict(manifest_sha256='frame-manifest',labels=20,loss=score['loss'])])


def test_forward_progress_does_not_affect_lateral_score():
    assert lateral_score(geometry(progress=0)) == lateral_score(geometry(progress=1000))
    assert lateral_score(geometry()) == pytest.approx(1.+.25*.4+.2)


def test_better_lateral_candidate_wins_despite_worse_longitudinal_diagnostic():
    incumbent=dict(measured(2.,0.,step=1),incumbent_export=True,
        protected_cohorts=[['frame-manifest']])
    candidate=measured(1.,100.,step=2)
    assert candidate['optimization_loss']>incumbent['optimization_loss']
    assert candidate['loss']<incumbent['loss']
    assert can_promote(candidate,incumbent)
    assert select_best([incumbent,candidate])['step']==2


def test_better_longitudinal_diagnostic_cannot_promote_worse_lateral_candidate():
    incumbent=dict(measured(1.,100.,step=1),incumbent_export=True)
    candidate=measured(2.,0.,step=2)
    assert candidate['optimization_loss']<incumbent['optimization_loss']
    assert not can_promote(candidate,incumbent)
    assert select_best([incumbent,candidate])['step']==1


def test_lateral_tie_ignores_longitudinal_tiebreaker():
    incumbent=dict(measured(1.,100.,step=1),incumbent_export=True)
    candidate=measured(1.,0.,step=2)
    assert not can_promote(candidate,incumbent)
    assert select_best([incumbent,candidate])['step']==1


def test_old_combined_scores_require_rescoring_before_comparison():
    candidate=measured(1.,0.)
    legacy=dict(measured(2.,0.),selection_metric='legacy_total_v1')
    with pytest.raises(ValueError,match='same selection metric'):
        can_promote(candidate,legacy)
    with pytest.raises(ValueError,match='same selection metric'):
        choose([candidate,legacy])


def test_rescoring_refreshes_metric_without_replacing_saved_winner(tmp_path):
    first=checkpoint(tmp_path,b'previous winner',1)
    old=measured(1.,100.,step=1)
    old.pop('selection_metric')
    protect_validation(tmp_path,first,old)
    saved=json.loads((tmp_path/'best.json').read_text())
    incumbent=dict(measured(1.,100.,step=1),sha256=first['sha256'],
        protected_cohorts=saved['protected_cohorts'])
    worse=checkpoint(tmp_path,b'worse',2)
    assert not protect_validation(tmp_path,worse,measured(2.,0.,step=2),incumbent)
    updated=json.loads((tmp_path/'best.json').read_text())
    assert updated['sha256']==first['sha256']
    assert updated['selection_metric']==SELECTION_METRIC
    assert checkpoint_path(tmp_path,updated).read_bytes()==b'previous winner'
    assert list((tmp_path/'preserved/rescored').glob('*.json'))
