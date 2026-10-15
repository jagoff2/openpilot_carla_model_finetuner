"""Keep current training active until admitted unmarked train/validation exist."""
import argparse
import json
from pathlib import Path
import time

import psutil

from lane_behavior.collect import source_inventory
from lane_behavior.mixed import discover, qualification
from lane_dataset.config import sha256_file


def emit(event,**details):
    print(json.dumps(dict(event=event,**details)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--seed-run',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args()
    if args.run_dir.exists():
        if not (args.run_dir/'run.json').is_file():
            raise ValueError('existing mixed directory lacks an immutable run receipt')
        from lane_behavior.mixed_follow import follow
        emit('resuming_existing_mixed_follow',run=str(args.run_dir))
        # follow independently verifies corpus, source, recipe and checkpoints.
        follow(args.corpus,args.run_dir)
        return 0
    seed_run=args.seed_run.resolve();sources=source_inventory();last=None
    while True:
        if source_inventory()!=sources:raise RuntimeError('source changed before mixed training handoff')
        try:
            rows,_=discover(args.corpus,verify_payloads=False)
            ready=qualification(rows)
        except (OSError,json.JSONDecodeError) as error:
            emit('waiting_for_atomic_corpus_update',detail=str(error));time.sleep(10);continue
        counts=ready['cohorts']
        if counts!=last:
            emit('waiting_for_admitted_unmarked_cohorts',cohorts=counts,missing=ready['missing_required_cohorts'])
            last=counts
        if ready['ready']:break
        time.sleep(10)
    # Stop only the exact old trainer, at its normal saved-checkpoint boundary.
    status=json.loads((seed_run/'status.json').read_text(encoding='utf-8'))
    process=None
    try:
        candidate=psutil.Process(status['pid']);command=candidate.cmdline()
        if ('lane_training.production' not in command or '--run-dir' not in command
                or Path(command[command.index('--run-dir')+1]).resolve()!=seed_run):
            raise RuntimeError('prior trainer PID does not match the named seed run')
        process=candidate
    except psutil.NoSuchProcess:pass
    if process is not None:
        (seed_run/'stop.request').touch(exist_ok=True)
        emit('waiting_for_prior_trainer_saved_checkpoint',pid=process.pid,run=str(seed_run))
        deadline=time.monotonic()+1800
        while process.is_running():
            if time.monotonic()>deadline:raise TimeoutError('prior trainer did not stop at checkpoint within30minutes')
            time.sleep(5)
    best=json.loads((seed_run/'best.json').read_text(encoding='utf-8'))
    checkpoint=(seed_run/best['checkpoint']).resolve()
    if not checkpoint.is_relative_to(seed_run) or sha256_file(checkpoint)!=best['sha256']:
        raise RuntimeError('prior immutable best checkpoint integrity failed')
    if source_inventory()!=sources:raise RuntimeError('source changed during mixed handoff')
    emit('starting_mixed_follow',seed_step=best['step'],seed_sha256=best['sha256'],run=str(args.run_dir))
    from lane_behavior.mixed_follow import follow
    follow(args.corpus,args.run_dir,seed_checkpoint=checkpoint,seed_sha256=best['sha256'])
    return 0


if __name__=='__main__':raise SystemExit(main())
