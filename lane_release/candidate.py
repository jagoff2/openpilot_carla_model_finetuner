"""Build a specific preserved checkpoint without changing any best-model pointer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from uuid import uuid4
import zipfile

import torch

from . import export
from lane_training.checkpoint_store import checkpoint_path, immutable_copy, immutable_json


def preserved_record(run, step):
    """Prefer the saved best; otherwise find an unambiguous validated snapshot."""
    best=run/'best.json'
    if best.exists():
        record=json.loads(best.read_text())
        if record['step']==step:return record
    records=[json.loads(path.read_text()) for path in sorted((run/'preserved/records').glob('*.json'))]
    matches=[r for r in records if r.get('step')==step and r.get('rows')]
    if len({r['sha256'] for r in matches})!=1:
        raise ValueError('requested step has no unique preserved validated checkpoint')
    return matches[-1]


def prepare_candidate(run, releases, step):
    run=Path(run).resolve();releases=Path(releases).resolve()
    with export.run_lock(releases):
        record=preserved_record(run,step)
        source=checkpoint_path(run,record)
        if source.parent!=run/'preserved/checkpoints':
            raise ValueError('explicit candidates require an immutable preserved checkpoint')
        torch.set_num_threads(2)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output=releases/f'{stamp}-step{step}-candidate-{uuid4().hex[:8]}'
        output.mkdir(parents=True)
        immutable_json(output/'checkpoint-validation.json',record)
        export.atomic_json(output/'status.json',dict(stage='exporting_pinned_checkpoint',checkpoint_step=step,
            onnx_ready=False,runtime_ready=False,driving_qualified=False))
        try:
            snapshot=output/'source-checkpoint.pt'
            immutable_copy(source,snapshot,record['sha256'])
            model,package=export.load_checkpoint(snapshot)
            if package['step']!=step:raise ValueError('checkpoint step does not match requested step')
            corpus=Path(package['training_state']['config']['corpus']).resolve()
            validation=[]
            for row in record['rows']:
                path=(corpus/row['episode']).resolve()
                if corpus not in path.parents or export.sha(path)!=row['manifest_sha256']:
                    raise ValueError('saved validation manifest no longer matches')
                if json.loads(path.read_text()).get('split')!='validation':
                    raise ValueError('candidate validation must use held-out validation only')
                validation.append(dict(path=str(path),sha256=row['manifest_sha256']))
            if not validation:raise ValueError('saved candidate has no held-out validation')
            package['optimizer']=None
            torch.save(package,output/'checkpoint.pt')
            selection=dict(record,source_checkpoint_sha256=record['sha256'],kind='explicit_preserved_checkpoint',
                requested_step=step,promote_export=False,test_split_used=False,
                selection_basis='saved checkpoint validation; explicit user selection; no reranking',
                export_checkpoint='checkpoint.pt',export_checkpoint_sha256=export.sha(output/'checkpoint.pt'))
            export.atomic_json(output/'selection.json',dict(selected=selection,candidates=[]))
            export.event('exporting_pinned_checkpoint',step=step,sha256=record['sha256'],directory=str(output))
            roles=export.export_models(model,selection,output)
            export.event('verifying_pinned_onnx',step=step,frames=120)
            verification=export.verification_trace(output,roles,validation)
            shutil.copyfile(Path(export.__file__).with_name('device_build.py'),output/'device_build.py')
            names=['device_build.py','verification.npz']+[v[k] for v in roles.values() for k in ('onnx','metadata')]
            payload=dict(schema_version=1,device='c3x',backend='QCOM',roles=roles,
                bundle=export.bundle_template(roles,selection),selected_checkpoint_sha256=record['sha256'],
                step=step,files={name:export.sha(output/name) for name in names},onnx_verification=verification)
            export.atomic_json(output/'payload.json',payload)
            # A source package contains ONNX and metadata, not executable PKLs.
            archive=output/f'cd210-centerline-s{step}-onnx-candidate.zip'
            with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as stream:
                for name in ['payload.json','selection.json','checkpoint-validation.json','onnx-verification.json',*names]:
                    stream.write(output/name,name)
            export.atomic_json(output/'status.json',dict(stage='onnx_ready_device_compile_pending',
                checkpoint_step=step,source_checkpoint_sha256=record['sha256'],onnx_ready=True,
                runtime_ready=False,driving_qualified=False,source_archive=str(archive),
                source_archive_sha256=export.sha(archive)))
            export.event('pinned_candidate_ready',step=step,directory=str(output),source_archive=str(archive))
            return output
        except Exception as error:
            export.atomic_json(output/'status.json',dict(stage='local_export_failed',checkpoint_step=step,
                error=str(error),onnx_ready=False,runtime_ready=False,driving_qualified=False))
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=export.DEFAULT_RUN)
    parser.add_argument('--releases',type=Path,default=export.DEFAULT_RELEASES)
    parser.add_argument('--step',type=int,required=True)
    parser.add_argument('--device',default=export.DEFAULT_DEVICE)
    parser.add_argument('--onnx-only',action='store_true')
    args=parser.parse_args()
    output=prepare_candidate(args.run_dir,args.releases,args.step)
    if not args.onnx_only:
        try:export.compile_remote(output,args.device)
        except export.DeviceUnavailable as error:
            export.event('device_offline_compile_pending',directory=str(output),message=str(error))
            return 3
    return 0


if __name__=='__main__':raise SystemExit(main())
