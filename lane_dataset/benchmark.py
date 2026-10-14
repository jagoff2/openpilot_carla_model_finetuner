"""One/two-worker measurements include sensor, evidence, compression and failures."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from .config import Config, write_json


def gpu_sample():
    result=subprocess.run(["nvidia-smi","--query-gpu=index,uuid,memory.used,utilization.gpu","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=10)
    if result.returncode: raise RuntimeError(result.stderr)
    return dict(timestamp=time.time(),csv=result.stdout.strip())


def stop_workers(processes):
    """Reap every child before closing its log or returning control."""
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def captures_complete(captures, workers, ticks, mode):
    identities = []
    for item in captures:
        capture = item["capture"]
        manifest = capture.get("manifest", {})
        worker = capture.get("worker", manifest.get("worker"))
        if type(worker) is not int or worker not in range(workers):
            return False
        identities.append(worker)
        if manifest.get("error"):
            return False
        if mode != "generate" and manifest.get("measured_throughput", {}).get("ticks") != ticks:
            return False
        if mode != "generate" and (manifest.get("worker") != worker or manifest.get("port") != 2400 + 100 * worker):
            return False
        if mode == "generate" and any(episode.get("status") != "accepted" for episode in capture.get("episodes", [])):
            return False
    return len(identities) == workers and set(identities) == set(range(workers))


def run(root,workers,ticks,mode,report_directory,speed=5.,traffic=0):
    if workers not in (1,2) or ticks < 4 or mode not in ("benchmark","develop","generate"):
        raise ValueError("invalid worker count, tick count, or benchmark mode")
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be finite and positive")
    out=Path(report_directory);out.mkdir(parents=True,exist_ok=True)
    processes=[];streams=[];samples=[];started=time.perf_counter();started_wall=time.time()
    # Wait for startup through read-only version calls; this code never ticks.
    import carla
    for worker in range(workers):
        deadline=time.monotonic()+120
        while True:
            try:
                client=carla.Client("127.0.0.1",2400+100*worker);client.set_timeout(2)
                if client.get_server_version()!="0.9.16": raise RuntimeError("server version mismatch")
                break
            except RuntimeError:
                if time.monotonic()>deadline: raise
                time.sleep(1)
    try:
        for worker in range(workers):
            stream=(out/f"worker-{worker}.log").open("w",encoding="utf-8");streams.append(stream)
            command=[sys.executable,"-m","lane_dataset","--root",str(root),"benchmark" if mode=="benchmark" else "generate","--port",str(2400+100*worker),"--worker",str(worker)]
            if mode!="benchmark": command += ["--worker-count",str(workers)]
            if mode=="develop": command += ["--development"]
            if mode!="generate": command += ["--ticks",str(ticks),"--speed",str(speed),"--traffic",str(traffic)]
            processes.append(subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0)))
        while True:
            statuses = [p.poll() for p in processes]
            if any(status is not None and status != 0 for status in statuses):
                stop_workers(processes)
                break
            if all(status is not None for status in statuses):
                break
            samples.append(gpu_sample());write_json(out/"gpu-observed.json",samples)
            time.sleep(1)
        elapsed=time.perf_counter()-started
        captures=[]
        pattern={"benchmark":"benchmark-w*.json","develop":"development-w*.json","generate":"generation-worker-*.json"}[mode]
        for path in (Path(root)/"reports").glob(pattern):
            if path.stat().st_mtime>=started_wall:
                doc=json.loads(path.read_text());captures.append(dict(path=str(path),capture=doc))
        processed=sum(item["capture"].get("manifest",{}).get("measured_throughput",{}).get("ticks",0) for item in captures)
        complete=all(p.returncode==0 for p in processes) and captures_complete(captures,workers,ticks,mode)
        report=dict(workers=workers,ticks_per_worker=ticks,requested_speed_mps=speed if mode!="generate" else None,requested_traffic_count=traffic if mode!="generate" else None,wall_seconds=elapsed,processed_ticks=processed,complete_pipeline=complete,
                    aggregate_ticks_per_wall_second=processed/elapsed,captures=captures,
                    exit_codes=[p.returncode for p in processes],gpu_samples=samples,
                    qualified_throughput=False,scope="development pipeline throughput; accepted-label throughput not established")
        write_json(out/"benchmark.json",report)
        print(json.dumps(report,indent=2))
        return report
    finally:
        stop_workers(processes)
        for stream in streams: stream.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);parser.add_argument("--workers",type=int,choices=[1,2],default=2)
    parser.add_argument("--ticks",type=int,default=80);parser.add_argument("--mode",choices=["benchmark","develop","generate"],default="benchmark");parser.add_argument("--report-directory",type=Path,required=True)
    parser.add_argument("--speed",type=float,default=5.)
    parser.add_argument("--traffic",type=int,choices=[0,2,3,4],default=0)
    a=parser.parse_args();report=run(a.root,a.workers,a.ticks,a.mode,a.report_directory,a.speed,a.traffic)
    raise SystemExit(0 if report["complete_pipeline"] else 2)
