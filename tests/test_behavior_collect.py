from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import errno

import pytest

from lane_dataset.config import sha256_file,write_json
from lane_behavior import collect


@pytest.fixture
def queue(tmp_path):
    source=tmp_path/"source.xodr";source.write_text("<OpenDRIVE/>")
    task=dict(native_unmarked=True,procedural=False,unmarked_road=True,training_ready=False,
        capture_contract="unmarked_rht_obstacle_v1",episode_id="native-parked",
        unmarked_behavior=dict(case="parked_car"),opendrive_path=str(source),
        route=dict(length_m=150.,source_xodr_sha256=sha256_file(source),unmarked_road=dict(source_bindings={})))
    plan=tmp_path/"plan.json"
    write_json(plan,dict(status="offline_candidate_only",split="development",rendered_qualification=False,tasks=[task]))
    job=dict(job_id="parked-train",plan=str(plan),plan_sha256=sha256_file(plan),case="parked_car",split="train",ticks=400,start_s=10.)
    path=tmp_path/"queue.json";write_json(path,dict(schema=collect.SCHEMA,jobs=[job]))
    return path,job,task


def test_dry_validation_reads_pinned_native_plan_without_creating_state(queue,tmp_path):
    path,job,_=queue
    before=set(tmp_path.rglob("*"))
    value=collect.validate_queue(path)
    assert value["jobs"][0]["job"]["split"]=="train"
    assert value["jobs"][0]["required_new_bytes"]==2*value["jobs"][0]["capture_budget_bytes"]
    assert set(tmp_path.rglob("*"))==before
    Path(job["plan"]).write_text("{}")
    with pytest.raises(ValueError,match="SHA256 mismatch"):collect.validate_queue(path)


@pytest.mark.parametrize("field,value",[("split","development"),("split","test"),("ticks",300),("start_s",float("nan")),("typo",1)])
def test_queue_rejects_ambiguous_or_unbounded_jobs(queue,field,value):
    path,job,_=queue;job[field]=value
    # NaN is deliberately malformed input, written through a permissive writer.
    import json
    path.write_text(json.dumps(dict(schema=collect.SCHEMA,jobs=[job])))
    with pytest.raises(ValueError):collect.validate_queue(path)


@pytest.fixture
def runtime(queue,tmp_path,monkeypatch):
    path,job,task=queue;events=[];code=["a"*64];physical=[True];admissions=[];captures=[]
    monkeypatch.setattr(collect,"source_inventory",lambda:{"lane_behavior/collect.py":code[0]})
    monkeypatch.setattr(collect.shutil,"disk_usage",lambda root:SimpleNamespace(free=1000*collect.GIB))
    monkeypatch.setattr(collect,"_assignment",lambda root,task,sha,split:dict(candidate_task_sha256=sha,split=split,frozen_split_sha256="b"*64))
    @contextmanager
    def reserve(*args,**kwargs):
        events.append("reserve")
        try:yield
        finally:events.append("release")
    monkeypatch.setattr(collect,"reserved_port",reserve)
    def capture(**kwargs):
        events.append("capture");captures.append(kwargs)
        directory=tmp_path/("capture-"+str(len(captures)));directory.mkdir()
        report=dict(status="captured_pending_observed_admission",errors=[],cleanup_errors=[],
            physical=dict(passed=physical[0],reasons=[] if physical[0] else ["lateral_jerk_above_limit"]),
            source_hashes=collect.source_inventory(),plan_sha256=job["plan_sha256"],case=job["case"],requested_ticks=400)
        write_json(directory/"capture-report.json",report);write_json(directory/"candidate-task.json",task)
        return report,directory
    def audit(capture,**kwargs):
        assert events[-1]=="release"
        events.append("audit");admissions.append(kwargs)
        return dict(training_ready=True,accepted_labels=7,errors=[],remaining_checks=[],published_manifest="manifest"),capture/"audit"
    monkeypatch.setattr(collect,"run_capture",capture);monkeypatch.setattr(collect,"audit_candidate",audit)
    return SimpleNamespace(path=path,job=job,task=task,events=events,code=code,physical=physical,
        admissions=admissions,captures=captures,make_capture=capture)


def test_capture_releases_reservation_before_exactly_one_admission_and_resume_skips(runtime,tmp_path):
    result=collect.run_queue(runtime.path,root=tmp_path,port=2500)
    assert runtime.events==["reserve","capture","release","audit"]
    assert result["accepted_labels"]==7 and result["executed_jobs"]==1
    assert runtime.admissions[0]["refresh_split_binding"] is True
    assert runtime.admissions[0]["split_assignment"]["split"]=="train"
    assert runtime.admissions[0]["expected_source_hashes"]==collect.source_inventory()
    repeated=collect.run_queue(runtime.path,root=tmp_path,port=2500)
    assert repeated["executed_jobs"]==0 and repeated["results"][0]["skipped"]
    assert len(runtime.captures)==len(runtime.admissions)==1


def test_deterministic_physical_failure_is_not_replayed_or_recaptured_until_code_changes(runtime,tmp_path):
    runtime.physical[0]=False
    first=collect.run_queue(runtime.path,root=tmp_path,port=2500)
    assert first["results"][0]["status"]=="physical_rejected" and not runtime.admissions
    collect.run_queue(runtime.path,root=tmp_path,port=2500)
    assert len(runtime.captures)==1
    runtime.code[0]="c"*64
    collect.run_queue(runtime.path,root=tmp_path,port=2500)
    assert len(runtime.captures)==2 and not runtime.admissions


def test_disk_budget_includes_complete_copy_headroom_and_legacy_reserve(runtime,tmp_path,monkeypatch):
    item=collect.validate_queue(runtime.path)["jobs"][0]
    free=item["required_new_bytes"]+20*collect.GIB-1
    monkeypatch.setattr(collect.shutil,"disk_usage",lambda root:SimpleNamespace(free=free))
    with pytest.raises(RuntimeError,match="publication copy"):
        collect.run_queue(runtime.path,root=tmp_path,port=2500)
    assert not runtime.events


def existing_job(runtime,tmp_path):
    report,capture=runtime.make_capture()
    runtime.events.clear();runtime.captures.clear()
    job={**runtime.job,"existing_capture":str(capture),"capture_report_sha256":sha256_file(capture/"capture-report.json")}
    write_json(runtime.path,dict(schema=collect.SCHEMA,jobs=[job]))
    return capture


def test_existing_capture_is_admitted_without_reservation_or_recapture(runtime,tmp_path,monkeypatch):
    capture=existing_job(runtime,tmp_path)
    def audit(path,**kwargs):
        assert path==capture and not runtime.events
        runtime.admissions.append(kwargs)
        return dict(training_ready=True,accepted_labels=5,errors=[],remaining_checks=[]),capture/"audit"
    monkeypatch.setattr(collect,"audit_candidate",audit)
    result=collect.run_queue(runtime.path,root=tmp_path,port=2500)
    assert result["accepted_labels"]==5 and len(runtime.admissions)==1 and not runtime.captures


@pytest.mark.parametrize("folder",["replays","admission-audits"])
def test_existing_diagnostic_replay_never_runs_again(runtime,tmp_path,folder):
    capture=existing_job(runtime,tmp_path)
    (capture/folder/"previous-diagnostic").mkdir(parents=True)
    result=collect.run_queue(runtime.path,root=tmp_path,port=2500)
    assert result["results"][0]["status"]=="existing_replay_requires_review"
    assert not runtime.admissions and not runtime.captures


def test_admission_exception_is_persisted_and_not_run_twice(runtime,tmp_path,monkeypatch):
    calls=[]
    def fail(*args,**kwargs):calls.append(1);raise RuntimeError("proof failure")
    monkeypatch.setattr(collect,"audit_candidate",fail)
    result=collect.run_queue(runtime.path,root=tmp_path,port=2500)
    assert result["results"][0]["status"]=="admission_failed"
    collect.run_queue(runtime.path,root=tmp_path,port=2500)
    assert calls==[1]


def test_deterministic_split_conflict_is_saved_without_claiming_simulator(runtime,tmp_path,monkeypatch):
    def rejected(*args):raise ValueError("fixed split overlap rejected")
    monkeypatch.setattr(collect,"_assignment",rejected)
    first=collect.run_queue(runtime.path,root=tmp_path,port=2500)
    assert first["results"][0]["status"]=="split_rejected"
    second=collect.run_queue(runtime.path,root=tmp_path,port=2500)
    assert second["executed_jobs"]==0 and not runtime.events


def test_source_change_during_capture_prevents_admission(runtime,tmp_path,monkeypatch):
    original=runtime.make_capture
    def changed(**kwargs):
        value=original(**kwargs);runtime.code[0]="d"*64;return value
    monkeypatch.setattr(collect,"run_capture",changed)
    result=collect.run_queue(runtime.path,root=tmp_path,port=2500)
    assert result["results"][0]["status"]=="capture_failed" and not runtime.admissions
    assert runtime.events[-1]=="release"


def test_reservation_waits_for_running_worker_then_world_lock_without_terminating_it(tmp_path,monkeypatch):
    events=[];checks=[0];clock=[0.]
    monkeypatch.setattr(collect,"acquire_reservation",lambda *a,**k:events.append("acquire") or "lease")
    monkeypatch.setattr(collect,"renew_reservation",lambda *a,**k:events.append("renew") or "renewed")
    monkeypatch.setattr(collect,"release_reservation",lambda *a,**k:events.append("release"))
    monkeypatch.setattr(collect.time,"monotonic",lambda:clock[0])
    monkeypatch.setattr(collect.time,"sleep",lambda amount:clock.__setitem__(0,clock[0]+amount))
    def access(*args):
        checks[0]+=1
        if checks[0]==1:raise RuntimeError("reserved port is still draining collection processes: 123")
    monkeypatch.setattr(collect,"require_reserved_or_idle",access)
    @contextmanager
    def owner(*args):
        if checks[0]==2:raise OSError(errno.EACCES,"busy")
        yield
    monkeypatch.setattr(collect,"world_owner",owner)
    with collect.reserved_port(tmp_path,2500,stop_file=tmp_path/"STOP"):
        events.append("capture_allowed")
    assert checks[0]==3 and events==["acquire","renew","capture_allowed","release"]


def test_reservation_timeout_always_releases_own_lease(tmp_path,monkeypatch):
    released=[];clock=[0.]
    monkeypatch.setattr(collect,"acquire_reservation",lambda *a,**k:"lease")
    monkeypatch.setattr(collect,"release_reservation",lambda *a,**k:released.append(a[1]))
    monkeypatch.setattr(collect.time,"monotonic",lambda:clock[0])
    monkeypatch.setattr(collect.time,"sleep",lambda amount:clock.__setitem__(0,clock[0]+amount))
    def access(*args):raise RuntimeError("reserved port is still draining collection processes: 123")
    monkeypatch.setattr(collect,"require_reserved_or_idle",access)
    with pytest.raises(TimeoutError):
        with collect.reserved_port(tmp_path,2500,wait_s=1.,stop_file=tmp_path/"STOP"):pytest.fail("not drained")
    assert released==["lease"]
@pytest.mark.parametrize('results,expected',[
    ([{'status':'accepted','accepted_labels':40}],0),
    ([{'status':'admission_failed','error':'serialization'}],2),
    ([{'status':'physical_rejected','accepted_labels':0}],2),
    ([{'skipped':True,'prior':{'status':'admission_rejected'}}],2),
    ([{'skipped':True,'prior':{'status':'accepted'}}],0),
])
def test_queue_exit_reports_job_failure(results,expected):
    from lane_behavior.collect import queue_exit_code
    assert queue_exit_code({'results':results})==expected
