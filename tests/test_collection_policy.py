from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lane_dataset.bootstrap import RollingWarmup
from lane_dataset.capture import recording_budget
from lane_dataset.controller import SteeringCalibration
from lane_dataset.capture_policy import CAPTURE_POLICY_VERSION, stopping_distance
from lane_dataset.admission import command_domain_errors
from lane_dataset.storage import rename_directory
from reports.collection_failures import cached_failure,remember_failure,retry_after_policy_change


def test_small_cruise_error_does_not_reject_laterally_stable_capture():
    calibration=replace(SteeringCalibration.fixture(),qualified=True,speed_min_mps=3.464,speed_max_mps=3.550)
    gate=RollingWarmup(calibration.speed_min_mps,calibration.speed_max_mps,speed_tolerance_mps=.1)
    for index in range(11):ready=gate.observe(index*.05,3.45,0.,command_accepted=True,visible_reference=True)
    assert ready
    assert not command_domain_errors(calibration,3.45,0.,calibration.delay_s+.05,speed_tolerance_mps=.1)
    assert command_domain_errors(calibration,3.2,0.,calibration.delay_s+.05,speed_tolerance_mps=.1)
    assert gate.report()['measured_speed_domain_mps']==[3.464,3.55]


def test_actual_route_reserve_accepts_warmup_overrun_but_not_missing_event_history():
    task=dict(warmup_distance_budget_m=30.,planned_preview_m=25.,maneuver_phase_ranges={'approach':[50.,60.]})
    args=dict(warmup_progress_m=31.,recording_start_m=31.,route_end_m=200.,ticks=800,maximum_speed_mps=2.55)
    report=recording_budget(task,**args)
    assert report['passed'] and report['warmup_overrun_m']==1.
    with pytest.raises(ValueError,match='pre-maneuver history'):recording_budget(task,**{**args,'recording_start_m':40.})
    with pytest.raises(ValueError,match='remaining route'):recording_budget(task,**{**args,'route_end_m':100.})


def test_stopping_distance_accounts_for_braking_instead_of_constant_speed():
    assert stopping_distance(3.5,3.,.15)==pytest.approx(.525+3.5**2/6)


def test_windows_directory_sharing_error_is_retried_without_clobber(tmp_path,monkeypatch):
    source=tmp_path/'source';target=tmp_path/'target';source.mkdir();(source/'data').write_bytes(b'intact')
    actual=Path.rename;attempts=[]
    def rename(path,destination):
        attempts.append(1)
        if len(attempts)<3:
            error=PermissionError('temporary sharing violation');error.winerror=5;raise error
        return actual(path,destination)
    monkeypatch.setattr(Path,'rename',rename)
    rename_directory(source,target)
    assert len(attempts)==3 and (target/'data').read_bytes()==b'intact'
    source.mkdir()
    with pytest.raises(FileExistsError):rename_directory(source,target)
    assert (target/'data').read_bytes()==b'intact'


def test_failure_memory_is_versioned_and_appearance_aware():
    task=dict(group_id='g',episode_id='g-validation-0',appearance=0)
    state={};failure=dict(status='failed',errors=['known geometry depth fixture failed'],capture_policy_version=CAPTURE_POLICY_VERSION)
    remember_failure(state,task,failure)
    assert cached_failure(state,{**task,'appearance':1})['kind']=='depth_fixture'
    assert not cached_failure(state,{**task,'group_id':'other'})
    assert retry_after_policy_change({**failure,'capture_policy_version':1})
    assert not retry_after_policy_change(failure)
    state={};remember_failure(state,task,{**failure,'errors':['rolling warmup failed']})
    assert cached_failure(state,task)
    assert not cached_failure(state,{**task,'appearance':1})


def test_corrected_admission_retries_completed_events_before_unproven_routes(tmp_path):
    from reports.collection_failures import priority_retries
    from lane_dataset.config import write_json
    result=dict(status='rejected',errors=['insufficient_phase_candidates:approach:3/4',
        'rendered_boundary_disagreement_above_five_cm'],labels={'label_count':19})
    assert retry_after_policy_change(result)
    assert not retry_after_policy_change({**result,'capture_policy_version':3})
    assert not retry_after_policy_change({**result,'status':'accepted'})
    write_json(tmp_path/'state.json',{'completed':{'near':result,'less':{**result,'labels':{'label_count':18}}}})
    for name in ('near','less'):
        write_json(tmp_path/(name+'.job.json'),{'task':{'episode_id':name,'route':{'actual':True},
            'maneuver_phase_ranges':{'approach':[10.,20.]}}})
    assert [task['episode_id'] for task in priority_retries(tmp_path)]==['near','less']


def test_depth_fixture_uses_independent_geometry_at_exact_pixels(monkeypatch):
    import sys
    from lane_dataset.depth_fixture import collision_depth_fixture
    from lane_dataset.capture import OPTICAL_FROM_CARLA
    from lane_dataset.geometry import LaneReference
    class Location:
        def __init__(self,x,y,z):self.x,self.y,self.z=x,y,z
    labels=SimpleNamespace(Roads=1,RoadLines=2)
    monkeypatch.setitem(sys.modules,'carla',SimpleNamespace(Location=Location,CityObjectLabel=labels))
    class World:
        def cast_ray(self,start,end):
            a=np.array([start.x,start.y,start.z]);b=np.array([end.x,end.y,end.z])
            if a[0]<2.5:  # Ego collision box surrounds the mounted camera.
                return [SimpleNamespace(location=Location(*a),label=99)]
            t=-a[2]/(b[2]-a[2])
            return [] if not 0<t<1 else [SimpleNamespace(location=Location(*a),label=99),
                SimpleNamespace(location=Location(*(a+t*(b-a))),label=1)]
    x=np.arange(0.,40.,.5);left=np.c_[x,np.full(len(x),-1.75),np.zeros(len(x))];right=left.copy();right[:,1]=1.75
    ref=LaneReference(left,right)
    k=np.array([[300.,0,320.],[0,300.,240.],[0,0,1.]])
    profile=SimpleNamespace(K=k,width=640,height=480)
    camera=np.eye(4);camera[2,3]=-1.3
    rows=np.arange(480)[:,None]
    depth=np.broadcast_to(np.divide(390.,rows+.5-240,out=np.full((480,1),1000.),where=rows>240),(480,640))
    images={name+suffix:value for name in ('narrow','wide') for suffix,value in (('_rgb_camera_from_world',camera),('_depth',depth))}
    report=collision_depth_fixture(images,SimpleNamespace(profiles=[profile,profile]),ref,World(),OPTICAL_FROM_CARLA)
    assert all(v['qualified'] and v['depth_mode']=='forward' for v in report.values())
    assert all(v['maximum_residual_m']['forward']<1e-6 for v in report.values())
    damaged={**images,'narrow_depth':depth*1.2}
    report=collision_depth_fixture(damaged,SimpleNamespace(profiles=[profile,profile]),ref,World(),OPTICAL_FROM_CARLA)
    assert not report['narrow']['qualified']
    # The route itself can start outside the narrow FOV. Other visible road
    # still supplies independent calibration, without consulting depth errors.
    behind=LaneReference(left-np.array([50.,0,0]),right-np.array([50.,0,0]))
    grid_images={**images,**{cam+'_semantic_segmentation':np.ones((480,640),np.uint8) for cam in ('narrow','wide')}}
    report=collision_depth_fixture(grid_images,SimpleNamespace(profiles=[profile,profile]),behind,World(),OPTICAL_FROM_CARLA)
    assert all(v['qualified'] for v in report.values())
