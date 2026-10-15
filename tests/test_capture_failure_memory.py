import copy
import numpy as np

from lane_dataset.capture_policy import CAPTURE_POLICY_VERSION
from lane_dataset.config import write_json
from reports.collection_failures import (cached_failure, remember_failure,
    restore_failure_memory, capture_statistics, warmup_signature)


def task(group='a', appearance=0):
    x=np.arange(151.)
    return dict(group_id=group,episode_id=f'{group}-{appearance}',appearance=appearance,
        town='Town03',nominal_speed_mps=3.5,physical_map_initialization=True,
        route=dict(arc_m=x.tolist(),left_xyz=np.c_[x,x*0-1.75,x*0].tolist(),
                   right_xyz=np.c_[x,x*0+1.75,x*0].tolist()))


def failure(message='rolling warmup failed'):
    return dict(status='failed',capture_policy_version=CAPTURE_POLICY_VERSION,errors=[message])


def test_identical_initial_corridor_is_blocked_after_two_appearances_fail():
    state={}
    remember_failure(state,task(),failure())
    assert cached_failure(state,task(appearance=1)) is None
    remember_failure(state,task(appearance=1),failure())
    other=task('other')
    other['route']['right_xyz'][100][1]+=10  # A distant event is not a new warmup.
    assert cached_failure(state,other)['kind']=='repeated_physical_warmup_corridor'
    # A genuinely different initialization path remains eligible.
    changed=copy.deepcopy(other);changed['route']['right_xyz'][20][1]+=.5
    assert cached_failure(state,changed) is None
    changed=copy.deepcopy(other);changed['town']='Town04'
    assert cached_failure(state,changed) is None
    changed=copy.deepcopy(other);changed['nominal_speed_mps']=2.5
    assert cached_failure(state,changed) is None


def test_same_appearance_in_two_splits_does_not_exhaust_weather_alternative():
    state={};first=task()
    remember_failure(state,first,failure())
    remember_failure(state,{**first,'episode_id':'a-validation-0'},failure())
    assert cached_failure(state,task('other')) is None


def test_physical_jerk_rejection_is_remembered_without_weakening_quality_limit():
    state={};result=failure('lateral_jerk_above_limit')
    remember_failure(state,task(),result)
    assert cached_failure(state,task())['kind']=='physical_jerk'
    assert cached_failure(state,task(appearance=1)) is None
    remember_failure(state,task(appearance=1),result)
    assert cached_failure(state,task(appearance=2))['kind']=='physical_jerk'


def test_old_policy_failures_do_not_block_current_collection():
    state={}
    for appearance in (0,1):
        remember_failure(state,task(appearance=appearance),
            {**failure(),'capture_policy_version':CAPTURE_POLICY_VERSION-1})
    assert cached_failure(state,task('other')) is None
    assert warmup_signature({'route':{}}) is None


def test_startup_restores_failure_evidence_and_separates_skips_from_capture_jobs(tmp_path):
    state={'completed':{'a-0':failure(),'a-1':failure(),
        'passed':{'status':'accepted'},'skipped':{'status':'overlap_with_accepted_independent_group'}}}
    for appearance in (0,1):
        write_json(tmp_path/f'a-{appearance}.job.json',dict(task=task(appearance=appearance)))
        write_json(tmp_path/f'a-{appearance}.job.result.json',failure())
    write_json(tmp_path/'passed.job.result.json',dict(status='measured_candidate_passed',
        capture_policy_version=CAPTURE_POLICY_VERSION))
    restore_failure_memory(tmp_path,state)
    assert cached_failure(state,task('other'))['kind']=='repeated_physical_warmup_corridor'
    stats=capture_statistics(state)
    assert stats['completed_jobs']==3 and stats['accepted_jobs']==1 and stats['rejected_jobs']==2
    assert stats['considered_candidates']==4 and stats['skipped_before_capture']==1
    assert stats['current_policy']['acceptance_rate']==1/3
    restore_failure_memory(tmp_path,state)
    assert capture_statistics(state)==stats
