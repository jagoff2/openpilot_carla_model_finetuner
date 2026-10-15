from copy import deepcopy
import numpy as np
import pytest

from lane_dataset.geometry import LaneReference
from lane_behavior.short_native import sequence_budget
from lane_behavior.sequence_guard import rebudget, phase_ranges, validate_teacher_budget
from lane_behavior.planned import PlannedObstacle, planned_scene_geometry
from lane_behavior.target import EgoFootprint, BehaviorConfig


def source_task():
    budget=sequence_budget(200.,obstacle_s=50.)
    return dict(route={'length_m':200.},sequence_budget=budget,minimum_ticks=900,capture_ticks=900,
        nominal_speed_mps=2.5,capture_start_route_m=10.,unmarked_behavior={'obstacles':[dict(
            route_s_m=50.,kind='parked_vehicle')]})


def scene(*,footprint=None,config=None,actor_half=2.):
    source=source_task();task=deepcopy(source)
    q=np.linspace(0.,100.,201)
    left,right=np.c_[q,q*0-4.5,q*0],np.c_[q,q*0+4.5,q*0]
    road=LaneReference(left,right,q)
    task['route']=dict(length_m=100.,arc_m=q.tolist(),unmarked_road=dict(
        road_left_edge_xyz=left.tolist(),road_right_edge_xyz=right.tolist()))
    task['sequence_budget']=rebudget(source,length_m=100.,start_s=10.,obstacle_s=50.)
    polygon=[[50-actor_half,3.7],[50+actor_half,3.7],[50+actor_half,5.2],[50-actor_half,5.2]]
    target=planned_scene_geometry(road,[PlannedObstacle('actor',polygon,'parked_vehicle','static')],
        ego=footprint,config=config)
    assert target.feasible and target.excursions
    return task,target


def test_metric_rebudget_recomputes_envelopes_instead_of_scaling_them():
    source=source_task()
    budget=rebudget(source,length_m=194.,start_s=9.7,obstacle_s=48.5)
    assert budget['capture_ticks']==900 and budget['start_s_m']==9.7
    assert budget['planned_event_envelope_m']==pytest.approx([33.8,63.2])
    assert budget['planned_event_envelope_m'][0]!=pytest.approx(source['sequence_budget']['planned_event_envelope_m'][0]*.97)
    assert budget['required_suffix_end_m']==pytest.approx(93.2)
    assert not budget['actual_timing_qualified']


def test_cropped_start_is_not_silently_relocated_or_budget_relaxed():
    with pytest.raises(ValueError):rebudget(source_task(),length_m=100.,start_s=4.,obstacle_s=50.)
    source=source_task();source['sequence_budget']['maximum_transition_m']=30.
    with pytest.raises(ValueError,match='parent short sequence budget changed'):
        rebudget(source,length_m=100.,start_s=10.,obstacle_s=50.)


def test_actual_teacher_budget_is_planning_only():
    task,target=scene()
    result=validate_teacher_budget(task,target,start_s=10.,ticks=900)
    assert result['passed'] and result['actual_body_dimensions_qualified']
    assert result['actual_teacher_excursions_qualified'] and not result['actual_timing_qualified']
    assert not result['training_ready'] and not result['observed_label_admission']


def test_existing_budget_cannot_silently_expand_and_new_version_requires_placement_contract():
    from lane_behavior.sequence_guard import ON_ROAD_VERSION
    task,target=scene()
    task['sequence_budget']['maximum_transition_m']=14.
    with pytest.raises(ValueError,match='relaxed or changed'):
        validate_teacher_budget(task,target,start_s=10.,ticks=900)
    task['sequence_budget']['version']=ON_ROAD_VERSION
    with pytest.raises(ValueError,match='explicit grounded on-road'):
        validate_teacher_budget(task,target,start_s=10.,ticks=900)


@pytest.mark.parametrize('mutation,reason',[
    ('ego','ego footprint'),('actor','actor footprint'),('transition','teacher transition'),
    ('ticks','runtime start/ticks'),('placement','fixed sequence scene'),('budget','relaxed or changed')])
def test_actual_bound_violations_rejected_before_recording(mutation,reason):
    kwargs={}
    if mutation=='ego':kwargs['footprint']=EgoFootprint(length_m=5.2)
    if mutation=='actor':kwargs['actor_half']=3.2
    if mutation=='transition':kwargs['config']=BehaviorConfig(min_transition_m=9.)
    task,target=scene(**kwargs)
    if mutation=='placement':task['unmarked_behavior']['obstacles'][0]['route_s_m']=51.
    if mutation=='budget':task['sequence_budget']['maximum_actor_half_length_m']=9.
    with pytest.raises(ValueError,match=reason):
        validate_teacher_budget(task,target,start_s=10.,ticks=899 if mutation=='ticks' else 900)
