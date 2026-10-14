from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from lane_dataset.bootstrap import RollingWarmup,check_recording_seam
from lane_dataset.capture import _calibration,longitudinal_setpoint
from lane_dataset.controller import SteeringCalibration
from lane_dataset.simulator import state


def observe(gate,index,speed=5.,lateral=0.,accepted=True,visible=True):
    return gate.observe(index*.05,speed,lateral,command_accepted=accepted,visible_reference=visible)


def test_warmup_requires_half_second_of_measured_steady_intervals():
    gate=RollingWarmup(4.95,5.05)
    for index in range(10):assert not observe(gate,index)
    assert observe(gate,10)
    assert gate.report()["steady_duration_s"]==.5
    with pytest.raises(RuntimeError,match="completed"):observe(gate,11)


def test_raw_launch_jerk_restarts_gate_without_discarding_peak():
    gate=RollingWarmup(4.95,5.05)
    for index in range(9):assert not observe(gate,index)
    assert not observe(gate,9,lateral=.57)  # 11.4 m/s^3 raw launch transient.
    assert gate.consecutive_intervals==0
    assert not observe(gate,10,lateral=0.)
    for index in range(11,20):assert not observe(gate,index)
    assert observe(gate,20)
    assert gate.report()["maximum_observed_lateral_jerk_mps3"]==pytest.approx(11.4)


@pytest.mark.parametrize("invalid",[{"speed":4.949},{"speed":5.051},{"accepted":False},{"visible":False},{"lateral":2.51}])
def test_domain_or_observation_failure_restarts_complete_steady_interval(invalid):
    gate=RollingWarmup(4.95,5.05)
    for index in range(8):observe(gate,index)
    observe(gate,8,**invalid)
    assert gate.consecutive_intervals==0
    assert not gate.ready


@pytest.mark.parametrize("timestamp",[0.,.1])
def test_warmup_rejects_duplicate_or_missing_physical_frames(timestamp):
    gate=RollingWarmup(4.95,5.05)
    observe(gate,0)
    with pytest.raises(ValueError,match="missing/duplicate"):
        gate.observe(timestamp,5.,0.,command_accepted=True,visible_reference=True)


def test_warmup_timeout_preserves_failed_measured_domain():
    gate=RollingWarmup(4.95,5.05,maximum_frames=12)
    for index in range(11):assert not observe(gate,index,speed=4.)
    with pytest.raises(TimeoutError,match="12 frames"):observe(gate,11,speed=4.)
    assert gate.report()["frames"]==12 and not gate.ready


def test_dense_recording_seam_detects_first_frame_jerk_without_smoothing():
    old=dict(frame_id=10,timestamp=.5,rotation_world=[0,0,0],acceleration_world=[0,0,0])
    current=dict(frame_id=11,timestamp=.55,rotation_world=[0,0,np.pi/2],acceleration_world=[-.57,0,0])
    report=check_recording_seam(old,current)
    assert report["lateral_jerk_mps3"]==pytest.approx(11.4)
    assert report["errors"]==["bootstrap_recording_seam_lateral_jerk_above_limit"]
    current["acceleration_world"]=[-.1,0,0]
    assert check_recording_seam(old,current)["passed"]


def test_dense_recording_seam_rejects_missing_frame_even_with_small_derivative():
    old=dict(frame_id=10,timestamp=.5,rotation_world=[0,0,0],acceleration_world=[0,0,0])
    current=dict(frame_id=12,timestamp=.6,rotation_world=[0,0,0],acceleration_world=[0,.1,0])
    assert not check_recording_seam(old,current)["passed"]


def test_state_records_actual_applied_controls_separate_from_planned_command():
    vec=lambda x,y,z:SimpleNamespace(x=x,y=y,z=z)
    transform=SimpleNamespace(location=vec(1,2,3),rotation=SimpleNamespace(roll=0,pitch=0,yaw=90))
    control=SimpleNamespace(steer=.2,throttle=.18,brake=.1,gear=-1,hand_brake=False,reverse=True,manual_gear_shift=True)
    ego=SimpleNamespace(get_transform=lambda:transform,get_velocity=lambda:vec(3,4,0),
        get_acceleration=lambda:vec(0,1,0),get_angular_velocity=lambda:vec(0,0,90),get_control=lambda:control)
    row=state(ego,SimpleNamespace(frame=12,timestamp=SimpleNamespace(elapsed_seconds=.6)))
    assert [row[k] for k in ("steer","throttle","brake","gear","reverse")]==[.2,.18,.1,-1,True]
    assert row["speed"]==5. and row["rotation_world"][2]==pytest.approx(np.pi/2)
    assert row["angular_velocity_world"][2]==pytest.approx(np.pi/2)


def test_calibration_selection_binds_matching_measured_speed_and_preserves_snapshot(tmp_path):
    for speed in (2.5,5.):
        folder=tmp_path/"reports"/f"calibration-{speed}"
        folder.mkdir(parents=True)
        fitted=replace(SteeringCalibration.fixture(),qualified=True,speed_min_mps=speed-.05,speed_max_mps=speed+.05)
        (folder/"fit.json").write_text(json.dumps(fitted.to_dict()))
    selected,binding=_calibration(tmp_path,2.5)
    assert selected.speed_min_mps==2.45
    with pytest.raises(RuntimeError,match="covering 3 m/s"):
        _calibration(tmp_path,3.)
    original=json.loads((tmp_path/"reports/calibration-2.5/fit.json").read_text())
    (tmp_path/"reports/calibration-2.5/fit.json").write_text("{}")
    assert json.loads(__import__("pathlib").Path(binding["path"]).read_text())==original


def test_recorded_longitudinal_integrator_cannot_drift_below_measured_domain():
    cal=replace(SteeringCalibration.fixture(),qualified=True,speed_min_mps=3.46,speed_max_mps=3.55)
    target=3.5
    for _ in range(40):
        target=longitudinal_setpoint(target,-1.5,3.5,cal,recording=True)
        assert 3.46<=target<=3.5
    assert longitudinal_setpoint(.1,1.5,3.5,cal,recording=False)==pytest.approx(.175)
    with pytest.raises(ValueError,match='uncalibrated_speed'):
        longitudinal_setpoint(3.5,-1.5,3.,cal,recording=True)


def test_two_workers_share_only_a_complete_calibration_snapshot(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import time
    import lane_dataset.capture as capture_module
    folder=tmp_path/"reports/calibration-parallel"
    folder.mkdir(parents=True)
    fitted=replace(SteeringCalibration.fixture(),qualified=True,speed_min_mps=2.45,speed_max_mps=2.55)
    (folder/"fit.json").write_text(json.dumps(fitted.to_dict()))
    (folder/"samples.bin").write_bytes(b"physical-samples"*1000)
    copytree=capture_module.shutil.copytree
    def slow_copy(*args,**kwargs):
        time.sleep(.1)
        return copytree(*args,**kwargs)
    monkeypatch.setattr(capture_module.shutil,"copytree",slow_copy)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:capture_module._calibration(tmp_path,2.5),range(2)))
    assert results[0][1]==results[1][1]
    from pathlib import Path
    assert (Path(results[0][1]["path"]).parent/"samples.bin").read_bytes()==b"physical-samples"*1000
    assert not list((tmp_path/"calibrations").glob("*.partial-*"))
