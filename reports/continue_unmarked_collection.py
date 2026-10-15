"""Wait for an already-running capture, then admit it and run its fixed queue."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from lane_dataset.config import sha256_file
from lane_behavior.collect import run_queue, source_inventory, validate_queue, queue_exit_code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--queue',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--loader-python',type=Path,required=True)
    p.add_argument('--port',type=int,default=2500)
    p.add_argument('--legacy-reserve-gib',type=float,default=10.)
    args=p.parse_args()
    original=validate_queue(args.queue)
    source=source_inventory()
    report_path=args.capture/'capture-report.json'
    deadline=time.monotonic()+3600
    print(json.dumps(dict(event='waiting_for_existing_physical_capture',capture=str(args.capture))),flush=True)
    while not report_path.exists():
        if source_inventory()!=source:raise RuntimeError('behavior source changed while awaiting capture')
        if time.monotonic()>deadline:raise TimeoutError('physical capture did not finish within one hour')
        time.sleep(10)
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if (report.get('status')!='captured_pending_observed_admission' or report.get('errors')
            or report.get('cleanup_errors') or report.get('physical',{}).get('passed') is not True):
        raise RuntimeError('existing capture failed physical qualification: '+str(report.get('errors') or report.get('physical')))
    if report.get('source_hashes')!=source or source_inventory()!=source:
        raise RuntimeError('completed capture does not match the frozen queue source')
    queue=dict(schema=original['schema'],jobs=[x['job'] for x in original['jobs']])
    queue['jobs'][0].update(existing_capture=str(args.capture.resolve()),capture_report_sha256=sha256_file(report_path))
    path=args.queue.parent/('native-existing-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    with path.open('x',encoding='utf-8') as stream:json.dump(queue,stream,indent=2)
    validate_queue(path)
    print(json.dumps(dict(event='starting_capture_admission_queue',queue=str(path))),flush=True)
    result=run_queue(path,root=args.root,port=args.port,loader_python=args.loader_python,
        legacy_reserve_gib=args.legacy_reserve_gib)
    print(json.dumps(result,indent=2),flush=True)
    return queue_exit_code(result)


if __name__=='__main__':raise SystemExit(main())
