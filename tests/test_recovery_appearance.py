from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from lane_dataset.review_video import review_annotations
from reports.automatic_corpus import prepare_capture_task, recovery_appearance_task


def test_cached_recovery_appearance_rebinds_command_without_mutating_catalogue():
    cached=dict(family='straight_recovery',recovery=True,appearance=0,
                initial_recovery_offset_m=.25,route={'nested':['unchanged']})
    right=recovery_appearance_task(cached)
    stale_left={**deepcopy(cached),'appearance':1}
    left=recovery_appearance_task(stale_left)
    assert right['initial_recovery_offset_m']==.25
    assert left['initial_recovery_offset_m']==-.25
    assert stale_left['initial_recovery_offset_m']==.25
    assert cached['appearance']==0
    assert recovery_appearance_task(left)==left


def test_worker_rebinds_stale_prepared_job_before_early_budget_return(monkeypatch):
    from reports import automatic_corpus
    monkeypatch.setattr(automatic_corpus,'measured_calibration',
        lambda speed:SimpleNamespace(speed_min_mps=3.4,speed_max_mps=3.6))
    task=dict(family='straight_recovery',recovery=True,appearance=1,
        initial_recovery_offset_m=.25,nominal_speed_mps=3.5,
        maneuver_phase_ranges={'recovery':[30.,40.]},warmup_distance_budget_m=45.,
        minimum_ticks=600,planned_preview_m=25.,route={'length_m':500.})
    prepared=prepare_capture_task(task)
    assert prepared['initial_recovery_offset_m']==-.25
    assert task['initial_recovery_offset_m']==.25


def test_screened_queue_clones_both_recovery_directions(tmp_path,monkeypatch):
    from reports import automatic_corpus, candidate_priority, collection_failures
    output=tmp_path/'reports/automatic';output.mkdir(parents=True)
    (output/'screened-pool.sqlite').touch()
    candidate=dict(group_id='geometry',family='straight_recovery',recovery=True,
        initial_recovery_offset_m=.25,appearance=0,seed=100,
        route={'section':{}},maneuver_phase_ranges={})
    monkeypatch.setattr(automatic_corpus,'read',lambda path:[dict(family='constant_radius')])
    monkeypatch.setattr(collection_failures,'slower_physical_retries',lambda path:[])
    monkeypatch.setattr(collection_failures,'priority_retries',lambda path:[])
    monkeypatch.setattr(candidate_priority,'scheduled_candidates',lambda *args:iter([(candidate,'train')]))
    queue=automatic_corpus.candidate_queue([],tmp_path)
    next(queue);next(queue)  # The two fixed non-recovery pilot jobs.
    right,prepared_right=next(queue);left,prepared_left=next(queue)
    assert prepared_right and prepared_left
    assert right['appearance']==0 and right['initial_recovery_offset_m']==.25
    assert left['appearance']==1 and left['initial_recovery_offset_m']==-.25
    assert candidate['appearance']==0 and candidate['initial_recovery_offset_m']==.25


def test_review_exposes_unlabelled_hold_and_measured_center_recovery():
    x=np.arange(5.)*10.
    center=np.column_stack((x,np.zeros(5),np.zeros(5)))
    geometry=dict(left_xyz=center-[0,1.8,0],right_xyz=center+[0,1.8,0],arc_m=x)
    telemetry=dict(position_world=np.array([[10.,.25,0.],[11.,.25,0.],[12.,-.01,0.]]),
                   route_s=np.array([10.,11.,12.]))
    manifest=dict(recovery_event=dict(onset_index=1,held_offset_m=.25))
    rows=review_annotations(manifest,telemetry,geometry)
    assert rows[0]==('CENTER OFFSET +0.250 m (right)',
                     'RECOVERY SETUP: hold +0.25 m | no labels')
    assert rows[1][1]=='CENTER RECOVERY: target 0.000 m'
    assert rows[2][0]=='CENTER OFFSET -0.010 m (left)'
