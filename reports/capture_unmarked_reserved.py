"""Capture one bounded physical behavior case while the other GPU collects."""
import argparse
import json
from pathlib import Path
import time

from lane_behavior.world_access import acquire_reservation,release_reservation,require_reserved_or_idle,renew_reservation
from lane_dataset.config import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--port',type=int,default=2500)
    parser.add_argument('--case',choices=('clear_road','parked_car','roadside_pedestrian'),default='parked_car')
    parser.add_argument('--ticks',type=int,default=900)
    parser.add_argument('--start-s',type=float,default=55.)
    parser.add_argument('--ready-file',type=Path,help='optional code-readiness receipt to await before importing the recorder')
    args=parser.parse_args()
    ready_file=args.ready_file
    del args.ready_file
    lease=acquire_reservation(args.root,args.port,duration_s=3600,purpose='unmarked physical avoidance and recovery qualification')
    try:
        deadline=time.monotonic()+1500
        while True:
            try:
                require_reserved_or_idle(args.root,args.port)
                break
            except RuntimeError as error:
                if 'still draining collection processes' not in str(error) or time.monotonic()>deadline:
                    raise
                print(json.dumps(dict(event='reserved_port_draining',port=args.port,detail=str(error))),flush=True)
                write_json(args.root/'reports/unmarked/physical-progress.json',dict(event='reserved_port_draining',port=args.port,detail=str(error),updated=time.time()))
                time.sleep(10)
        lease=renew_reservation(args.root,lease,duration_s=3600)
        if ready_file is not None:
            ready_deadline=time.monotonic()+600
            while not ready_file.exists():
                if time.monotonic()>ready_deadline:raise TimeoutError('bounded code readiness wait expired')
                time.sleep(2)
            from lane_dataset.config import sha256_file
            receipt=json.loads(ready_file.read_text(encoding='utf-8'))
            for path,expected in receipt['source_hashes'].items():
                if sha256_file(path)!=expected:raise RuntimeError('qualification source changed after readiness receipt: '+path)
        from lane_behavior.capture import run_capture
        report,output=run_capture(**vars(args))
        print(json.dumps(dict(event='physical_behavior_result',output=str(output),status=report['status'],
            physical=report.get('physical'),errors=report['errors'],cleanup_errors=report['cleanup_errors'])),flush=True)
        return 0 if report['status']=='captured_pending_observed_admission' and report.get('physical',{}).get('passed') and not report['cleanup_errors'] else 2
    finally:
        release_reservation(args.root,lease)


if __name__=='__main__':raise SystemExit(main())
