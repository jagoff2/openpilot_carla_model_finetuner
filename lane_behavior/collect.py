"""Immutable native capture/admission queue, separate from legacy collection.

Dry validation is the default. --run executes explicit fixed-split jobs once per
job/plan/source identity. A port reservation drains existing collection work,
never interrupts it, and is released before CPU admission. No old data is deleted.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import errno
import json
import math
import os
from pathlib import Path
import re
import shutil
import time

from lane_dataset.config import sha256_file, write_json
from lane_dataset.simulator import world_owner
from .admission import audit_candidate, current_split_assignment, verify_split
from .capture import capture_bounds, run_capture
from .qualification_probe import load_probe_task
from .records import digest
from .split_guard import publication_lock
from .world_access import (acquire_reservation, release_reservation, renew_reservation,
    require_reserved_or_idle)


SCHEMA="native_behavior_queue_v1"
GIB=1024**3
SOURCE_ROOT=Path(__file__).resolve().parents[1]


def source_inventory():
    return {p.relative_to(SOURCE_ROOT).as_posix():sha256_file(p)
        for package in ("lane_behavior","lane_dataset") for p in sorted((SOURCE_ROOT/package).glob("*.py"))}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _once(path, value):
    """An existing attempt/claim is never replaced, including after a crash."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n");stream.flush();os.fsync(stream.fileno())


def _bound_file(path, expected, base):
    path=Path(path)
    if not path.is_absolute():path=base/path
    path=path.resolve()
    if not isinstance(expected,str) or not re.fullmatch("[0-9a-f]{64}",expected) or sha256_file(path)!=expected:
        raise ValueError("immutable input SHA256 mismatch: "+str(path))
    return path


def validate_queue(path):
    """Read/hash/validate only. No reservation, status file, or process is made."""
    path=Path(path).resolve();document=_read(path)
    if set(document)!={"schema","jobs"} or document["schema"]!=SCHEMA or not isinstance(document["jobs"],list) or not document["jobs"]:
        raise ValueError("queue requires schema native_behavior_queue_v1 and nonempty jobs")
    jobs=[];identities=set()
    required={"job_id","plan","plan_sha256","case","split","ticks","start_s"}
    optional={"existing_capture","capture_report_sha256"}
    for original in document["jobs"]:
        if not isinstance(original,dict) or not required<=original.keys() or set(original)-required-optional:
            raise ValueError("job has missing or unknown fields")
        job=dict(original)
        if not isinstance(job["job_id"],str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}",job["job_id"]):
            raise ValueError("job_id must be a bounded printable identifier")
        if job["job_id"] in identities:raise ValueError("duplicate job_id")
        identities.add(job["job_id"])
        if job["split"] not in ("train","validation"):
            raise ValueError("queue job requires explicit fixed train or validation split")
        if job["case"] not in ("clear_road","parked_car","roadside_pedestrian"):
            raise ValueError("unsupported native behavior case")
        plan=_bound_file(job["plan"],job["plan_sha256"],path.parent)
        task=load_probe_task(plan,job["case"])
        budget=math.ceil(capture_bounds(task,job["ticks"],job["start_s"]))
        sources=task["route"]["unmarked_road"].get("source_bindings",{})
        source_bytes=0
        for binding in sources.values():
            source=_bound_file(binding["path"],binding["sha256"],plan.parent)
            source_bytes+=source.stat().st_size
        if ("existing_capture" in job)!=("capture_report_sha256" in job):
            raise ValueError("existing capture requires exact capture_report_sha256")
        if "existing_capture" in job:
            capture=Path(job["existing_capture"])
            if not capture.is_absolute():capture=path.parent/capture
            capture=capture.resolve()
            _bound_file(capture/"capture-report.json",job["capture_report_sha256"],path.parent)
            report=_read(capture/"capture-report.json")
            if (report.get("plan_sha256")!=job["plan_sha256"] or report.get("case")!=job["case"]
                    or report.get("requested_ticks")!=job["ticks"]
                    or digest(_read(capture/"candidate-task.json"))!=digest(task)):
                raise ValueError("existing capture does not match immutable job task/plan/case/budget")
            job["existing_capture"]=str(capture)
            # The publisher copies the complete retained inventory, including raw.
            budget=sum(p.stat().st_size for p in capture.rglob("*") if p.is_file())
        job["plan"]=str(plan)
        jobs.append(dict(job=job,task=task,capture_budget_bytes=budget+source_bytes,
            required_new_bytes=(budget+source_bytes)*(1 if "existing_capture" in job else 2)))
    return dict(schema=SCHEMA,queue=str(path),queue_sha256=sha256_file(path),jobs=jobs)


def _assignment(root, task, task_sha, split):
    with publication_lock(root):
        assignment=current_split_assignment(root,dict(candidate_task_sha256=task_sha,split=split))
        result=verify_split(root,task,task_sha,assignment)
        if not result["passed"]:raise ValueError("fixed split overlap rejected: "+str(result["errors"]))
        return assignment


def _assert_sources(expected):
    if source_inventory()!=expected:
        raise RuntimeError("queue source changed; restart with the new source identity")


def _check_stop(stop_file):
    if Path(stop_file).exists():raise InterruptedError("native behavior queue stop file exists")


@contextmanager
def reserved_port(root, port, *, wait_s=900., stop_file):
    """Keep the existing worker alive through cleanup/publication before reuse."""
    if not math.isfinite(wait_s) or not 0<wait_s<=1800:
        raise ValueError("drain wait must be positive and <=1800 seconds")
    lease=acquire_reservation(root,port,duration_s=3600.,purpose="immutable native behavior queue")
    try:
        deadline=time.monotonic()+wait_s
        while True:
            _check_stop(stop_file)
            try:
                require_reserved_or_idle(root,port)
                with world_owner(port,root):pass
                break
            except RuntimeError as error:
                if not str(error).startswith("reserved port is still draining collection processes:"):
                    raise
            except OSError as error:
                if error.errno not in (errno.EACCES,errno.EAGAIN,errno.EDEADLK) and getattr(error,"winerror",None) not in (32,33):
                    raise
            if time.monotonic()>=deadline:raise TimeoutError("reserved collector port did not drain")
            time.sleep(min(.5,max(0.,deadline-time.monotonic())))
        lease=renew_reservation(root,lease,duration_s=3600.)
        # run_capture acquires world ownership again and checks that no actors
        # remain before changing the map. The reservation prevents new dispatch.
        yield lease
    finally:
        release_reservation(root,lease)


def run_queue(queue, *, root, port, loader_python=None, max_jobs=None, wait_s=900.,
              legacy_reserve_gib=10., stop_file=None):
    root=Path(root).resolve()
    if type(port) is not int or not 1<=port<=65535:raise ValueError("invalid simulator port")
    if not math.isfinite(legacy_reserve_gib) or legacy_reserve_gib<0:raise ValueError("invalid legacy disk reserve")
    validated=validate_queue(queue);sources=source_inventory();code_sha=digest(sources)
    if max_jobs is not None and (type(max_jobs) is not int or max_jobs<=0):raise ValueError("max_jobs must be positive")
    base=root/"reports/unmarked/automatic"
    stop_file=Path(stop_file) if stop_file is not None else base/"STOP"
    results=[];executed=0
    with world_owner("behavior-queue-"+validated["queue_sha256"][:20],root):
        for item in validated["jobs"]:
            _check_stop(stop_file);_assert_sources(sources)
            job=item["job"];key=digest(dict(job=job,source_sha256=code_sha))
            attempt=base/"attempts"/key
            if attempt.exists():
                result=_read(attempt/"result.json") if (attempt/"result.json").exists() else dict(status="incomplete_attempt_requires_review")
                results.append(dict(job_id=job["job_id"],skipped=True,attempt=str(attempt),prior=result))
                continue
            if max_jobs is not None and executed>=max_jobs:break
            # Preflight the fixed split before spending simulator time. Task
            # content digest is sufficient here; admission uses the saved SHA.
            try:
                _assignment(root,item["task"],digest(item["task"]),job["split"])
            except ValueError as error:
                # A geometric split conflict is deterministic for this job; it
                # must not keep rerunning or prevent other independent jobs.
                try:attempt.mkdir(parents=True,exist_ok=False)
                except FileExistsError:
                    results.append(dict(job_id=job["job_id"],skipped=True,status="claimed_by_another_queue"));continue
                executed+=1
                _once(attempt/"attempt.json",dict(schema="native_behavior_attempt_v1",job=job,
                    source_sha256=code_sha,source_hashes=sources,port=port))
                result=dict(status="split_rejected",job_id=job["job_id"],error=str(error),accepted_labels=0)
                _once(attempt/"result.json",result);results.append(result)
                print(json.dumps(dict(event="native_behavior_job_finished",**result)),flush=True)
                continue
            required=item["required_new_bytes"]+int((10.+legacy_reserve_gib)*GIB)
            if shutil.disk_usage(root).free<required:
                raise RuntimeError("insufficient space for raw capture plus complete publication copy, 10 GiB headroom and legacy reserve")
            context=reserved_port(root,port,wait_s=wait_s,stop_file=stop_file) if "existing_capture" not in job else _no_reservation()
            with context:
                _assert_sources(sources)
                # Recheck immutable external inputs after waiting for the worker.
                _bound_file(job["plan"],job["plan_sha256"],Path(queue).resolve().parent)
                try:attempt.mkdir(parents=True,exist_ok=False)
                except FileExistsError:
                    results.append(dict(job_id=job["job_id"],skipped=True,status="claimed_by_another_queue"));continue
                executed+=1
                _once(attempt/"attempt.json",dict(schema="native_behavior_attempt_v1",job=job,
                    source_sha256=code_sha,source_hashes=sources,required_free_bytes=required,port=port))
                try:
                    if "existing_capture" in job:
                        capture=Path(job["existing_capture"])
                        _bound_file(capture/"capture-report.json",job["capture_report_sha256"],capture)
                        report=_read(capture/"capture-report.json")
                    else:
                        shutil.copyfile(job["plan"],attempt/"plan.json")
                        _bound_file(attempt/"plan.json",job["plan_sha256"],attempt)
                        report,capture=run_capture(root=root,port=port,plan=attempt/"plan.json",
                            case=job["case"],ticks=job["ticks"],start_s=job["start_s"])
                        if report.get("source_hashes")!=sources:
                            raise RuntimeError("capture source snapshot differs from queue source identity")
                    _once(attempt/"capture.json",dict(path=str(capture),report_sha256=sha256_file(capture/"capture-report.json")))
                    _assert_sources(sources)
                except Exception as error:
                    result=dict(status="capture_failed",job_id=job["job_id"],error=str(error))
                    _once(attempt/"result.json",result);results.append(result);continue
            # The simulator reservation is now released even when admission is slow.
            result=dict(job_id=job["job_id"],capture=str(capture),status="physical_rejected",accepted_labels=0)
            try:
                if (report.get("status")!="captured_pending_observed_admission" or report.get("errors")
                        or report.get("cleanup_errors") or report.get("physical",{}).get("passed") is not True):
                    result["reasons"]=report.get("physical",{}).get("reasons",report.get("errors",[]))
                elif any((capture/"admission-audits").glob("*")) or any((capture/"replays").glob("*")):
                    result.update(status="existing_replay_requires_review",error="automatic queue never duplicates an existing diagnostic/admission replay")
                else:
                    assignment=_assignment(root,_read(capture/"candidate-task.json"),
                        sha256_file(capture/"candidate-task.json"),job["split"])
                    _once(attempt/"split-assignment.json",assignment)
                    _once(capture/"automatic-admission-claim.json",dict(attempt=str(attempt),source_sha256=code_sha))
                    _once(attempt/"admission-started.json",dict(capture=str(capture)))
                    _assert_sources(sources)
                    audit,output=audit_candidate(capture,corpus_root=root,split_assignment=assignment,publish=True,
                        loader_python=loader_python,refresh_split_binding=True,expected_source_hashes=sources)
                    result.update(status="accepted" if audit.get("training_ready") else "admission_rejected",
                        accepted_labels=audit["accepted_labels"],audit=str(output),errors=audit["errors"],
                        remaining_checks=audit["remaining_checks"],manifest=audit.get("published_manifest"))
            except Exception as error:
                result.update(status="admission_failed",error=str(error))
            _once(attempt/"result.json",result);results.append(result)
            print(json.dumps(dict(event="native_behavior_job_finished",**result)),flush=True)
    return dict(schema=SCHEMA,queue_sha256=validated["queue_sha256"],source_sha256=code_sha,
        executed_jobs=executed,accepted_labels=sum(r.get("accepted_labels",0) for r in results),results=results)


@contextmanager
def _no_reservation():
    yield


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue",type=Path);parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--port",type=int);parser.add_argument("--run",action="store_true")
    parser.add_argument("--loader-python",type=Path);parser.add_argument("--max-jobs",type=int)
    parser.add_argument("--wait-s",type=float,default=900.);parser.add_argument("--legacy-reserve-gib",type=float,default=10.)
    parser.add_argument("--stop-file",type=Path)
    args=parser.parse_args()
    if args.run:
        if args.port is None:parser.error("--run requires --port")
        result=run_queue(args.queue,root=args.root,port=args.port,loader_python=args.loader_python,
            max_jobs=args.max_jobs,wait_s=args.wait_s,legacy_reserve_gib=args.legacy_reserve_gib,stop_file=args.stop_file)
    else:
        if not math.isfinite(args.legacy_reserve_gib) or args.legacy_reserve_gib<0:
            parser.error("--legacy-reserve-gib must be finite and nonnegative")
        validated=validate_queue(args.queue)
        result=dict(valid=True,mode="dry_validation_no_processes_started",schema=SCHEMA,
            queue_sha256=validated["queue_sha256"],jobs=[dict(job_id=x["job"]["job_id"],split=x["job"]["split"],
                case=x["job"]["case"],required_new_bytes=x["required_new_bytes"]) for x in validated["jobs"]],
            fixed_headroom_bytes=int((10.+args.legacy_reserve_gib)*GIB))
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
