"""Snapshot, select, export and optionally compile the best current CD210 candidate."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import pickle
import re
import shlex
import shutil
import subprocess
import time
import zipfile
from uuid import uuid4

import numpy as np
import onnx
from onnx import numpy_helper
import torch

from lane_dataset.loader import SequenceEpisode
from lane_training.cd210 import DEFAULT_CHECKPOINT
from lane_training.cd210_model import create_model,load_checkpoint,save_checkpoint,export_onnx
from lane_training.onnx_model import ReleasedModel
from lane_training.production import atomic_json,discover,evaluate_episode,run_lock
from lane_training.checkpoint_store import (checkpoint_path, immutable_copy,
    immutable_json, can_promote, protected_cohorts)
from lane_training.verify_cd210 import float32_graph,runtime

PROJECT=Path(__file__).resolve().parents[1]
DEFAULT_RUN=PROJECT/'runs/cd210-centerline-production'
DEFAULT_RELEASES=PROJECT/'releases/cd210'
DEFAULT_DEVICE='iq@192.168.1.95'


class DeviceUnavailable(RuntimeError):
    """The local export is complete and native compilation can be resumed later."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event(stage,**fields):
    print(json.dumps(dict(stage=stage,**fields)),flush=True)


def snapshot_pointer(run,pointer_path,destination):
    """Retry publication races without locking or interrupting the trainer."""
    run=Path(run).resolve();pointer_path=Path(pointer_path);destination=Path(destination)
    for _ in range(12):
        try:
            before=pointer_path.read_bytes();pointer=json.loads(before)
            filename=pointer.get('file',pointer.get('checkpoint'))
            if not isinstance(filename,str):raise ValueError('checkpoint pointer has no filename')
            source=checkpoint_path(run,pointer)
            temporary=destination.with_suffix('.pt.partial')
            shutil.copyfile(source,temporary)
            actual=sha(temporary)
            if before!=pointer_path.read_bytes() or actual!=pointer['sha256']:
                temporary.unlink(missing_ok=True);time.sleep(.25);continue
            temporary.replace(destination)
            return dict(source_pointer=str(pointer_path),source_checkpoint_sha256=actual,
                snapshot=str(destination),step=pointer['step'])
        except (OSError,json.JSONDecodeError): time.sleep(.25)
    raise RuntimeError('could not obtain a consistent saved-checkpoint snapshot')


def snapshot_previous(run,latest,output):
    """Read the other retained slot, excluding a not-yet-committed newer update."""
    for source in sorted(run.glob('checkpoint-[01].pt')):
        destination=output/f'snapshot-previous-{source.stem}.pt'
        before=sha(source)
        if before==latest['source_checkpoint_sha256']:continue
        shutil.copyfile(source,destination)
        if sha(destination)!=before or sha(source)!=before:
            destination.unlink();continue
        package=torch.load(destination,map_location='cpu',weights_only=True)
        if package['step']>=latest['step']:
            destination.unlink();continue
        yield dict(snapshot=str(destination),source_checkpoint_sha256=before,step=package['step'],kind='fine_tuned')


def choose(scored):
    if not scored or any(not np.isfinite(r['loss']) for r in scored):
        raise ValueError('best selection requires finite held-out validation scores')
    return min(scored,key=lambda r:(r['loss'],not r.get('incumbent_export',False),
        r.get('kind')!='original_cd210',r['step']))


def select_best(scored):
    selected=choose(scored)
    incumbent=next((c for c in scored if c.get('incumbent_export')),None)
    promote_export=incumbent is None
    if incumbent is not None and selected is not incumbent:
        promote_export=can_promote(selected,incumbent)
        if not promote_export:selected=incumbent
    selected=dict(selected)
    selected['promote_export']=promote_export
    selected['protected_cohorts']=protected_cohorts(selected,incumbent)
    return selected


def select_checkpoint(run,output,releases):
    candidates=[]
    for name in ('best.json','resume.json'):
        pointer=run/name
        if pointer.exists():
            snapshot=snapshot_pointer(run,pointer,output/f'snapshot-{name[:-5]}.pt')
            snapshot['kind']='fine_tuned';candidates.append(snapshot)
            if name=='resume.json':candidates.extend(snapshot_previous(run,snapshot,output))
    for path in sorted((run/'preserved/records').glob('*.json')):
        record=json.loads(path.read_text())
        if record.get('kind') not in ('validated_candidate','previous_best'):continue
        if any(c['source_checkpoint_sha256']==record['sha256'] for c in candidates):continue
        dest=output/f'snapshot-preserved-{record["sha256"]}.pt'
        immutable_copy(checkpoint_path(run,record),dest,record['sha256'])
        candidates.append(dict(snapshot=str(dest),source_checkpoint_sha256=record['sha256'],
            step=record['step'],kind='fine_tuned'))
    history=releases/'best-export.json'
    if history.exists():
        record=json.loads(history.read_text());path=(releases/record['checkpoint']).resolve()
        if releases.resolve() not in path.parents or sha(path)!=record['checkpoint_sha256']:
            raise ValueError('previous export checkpoint integrity failed')
        dest=output/'snapshot-export-best.pt';shutil.copyfile(path,dest)
        prior=record
        if not record.get('protected_cohorts'):
            prior=json.loads((path.parent/'selection.json').read_text())['selected']
        candidates.append(dict(snapshot=str(dest),source_checkpoint_sha256=record['source_checkpoint_sha256'],
            step=record['step'],kind=record.get('kind','fine_tuned'),incumbent_export=True,
            protected_cohorts=record.get('protected_cohorts') or protected_cohorts(prior)))
    unique={c['source_checkpoint_sha256']:c for c in candidates};candidates=list(unique.values())
    if not candidates: raise ValueError('no committed training checkpoint is available')
    package=torch.load(candidates[0]['snapshot'],map_location='cpu',weights_only=True)
    corpus=Path(package['training_state']['config']['corpus'])
    training_config=package['training_state']['config']
    rows,_=discover(corpus)
    validation=[r for r in rows if r['split']=='validation']
    if not validation: raise ValueError('held-out accepted validation data is required to select the best checkpoint')
    cohort=hashlib.sha256(''.join(sorted(r['sha256'] for r in validation)).encode()).hexdigest()
    reference=create_model().eval().requires_grad_(False)
    scored=[]
    for candidate in candidates:
        model,package=load_checkpoint(candidate['snapshot'])
        if package['step']!=candidate['step']: raise ValueError('checkpoint step/pointer mismatch')
        if package['training_state']['config']!=training_config:raise ValueError('saved candidates use different training configurations')
        measurements=[]
        event('scoring_checkpoint',step=package['step'],validation_episodes=len(validation))
        for row in validation:
            with SequenceEpisode(row['path']) as episode:
                if episode.manifest['split']!='validation' or sha(episode.path)!=row['sha256']:
                    raise ValueError('held-out validation manifest changed')
                report=evaluate_episode(episode,model,reference,
                    lambda stage,tick,total:event(stage,step=package['step'],tick=tick,total=total))
            measurements.append(dict(manifest=row['id'],manifest_sha256=row['sha256'],**report))
        labels=sum(r['labels'] for r in measurements)
        if not labels:raise ValueError('validation contains no supported labels')
        scored.append(dict(**candidate,cohort=cohort,labels=labels,
            loss=sum(r['loss']*r['labels'] for r in measurements)/labels,
            baseline_loss=sum(r['baseline_loss']*r['labels'] for r in measurements)/labels,validation=measurements))
        del model
    baseline_path=output/'snapshot-original.pt'
    baseline_sha=save_checkpoint(reference,baseline_path,training_state=dict(config=training_config,export_baseline=True))
    baseline=scored[0]['baseline_loss']
    scored.append(dict(snapshot=str(baseline_path),source_checkpoint_sha256=baseline_sha,step=0,kind='original_cd210',
        cohort=cohort,labels=scored[0]['labels'],loss=baseline,baseline_loss=baseline,
        validation=[dict(manifest=r['manifest'],manifest_sha256=r['manifest_sha256'],labels=r['labels'],
            loss=r['baseline_loss']) for r in scored[0]['validation']]))
    selected=select_best(scored)
    selected['improves_original']=selected['loss']<selected['baseline_loss']
    selected['test_split_used']=False
    event('selected',kind=selected['kind'],step=selected['step'],validation_loss=selected['loss'],baseline_loss=selected['baseline_loss'])
    model,package=load_checkpoint(selected['snapshot'])
    package['optimizer']=None
    checkpoint=output/'checkpoint.pt';torch.save(package,checkpoint)
    for candidate in scored:candidate.pop('snapshot',None)
    selected.pop('snapshot',None)
    selected['export_checkpoint']='checkpoint.pt';selected['export_checkpoint_sha256']=sha(checkpoint)
    atomic_json(output/'selection.json',dict(selected=selected,candidates=scored))
    for path in output.glob('snapshot-*.pt'): path.unlink()
    return model,selected,validation


def export_models(model,selection,output):
    bindings=export_onnx(model,output)
    identity=f"local/cd210-centerline/{selection['source_checkpoint_sha256']}"
    roles={}
    for role,entry in bindings.items():
        path=Path(entry['path']);graph=onnx.load(path)
        source=getattr(model,role)
        for initializer in graph.graph.initializer:
            key=source.keys[initializer.name]
            expected=(source.weights[key] if key in source.weights else getattr(source,key)).detach().cpu()
            expected=expected.to(source.original_dtypes[initializer.name]).numpy()
            if not np.array_equal(numpy_helper.to_array(initializer),expected): raise ValueError('export changed checkpoint tensor values')
        props={p.key:p.value for p in graph.metadata_props}
        props.update(base_model_checkpoint=props['model_checkpoint'],model_checkpoint=identity,
            fine_tuning_step=str(selection['step']),fine_tuning_checkpoint_sha256=selection['source_checkpoint_sha256'])
        onnx.helper.set_model_props(graph,props);onnx.checker.check_model(graph);onnx.save(graph,path)
        with (DEFAULT_CHECKPOINT.parent/f'driving_{role}_c210m_metadata.pkl').open('rb') as stream:metadata=pickle.load(stream)
        metadata.update(model_checkpoint=identity,base_model_checkpoint=props['base_model_checkpoint'],fine_tuning_step=selection['step'])
        stem=f"driving_{role}_cd210_centerline_s{selection['step']}_{selection['source_checkpoint_sha256'][:12]}"
        metadata_name=f'{stem}_metadata.pkl'
        with (output/metadata_name).open('wb') as stream:pickle.dump(metadata,stream)
        roles[role]=dict(onnx=path.name,onnx_sha256=sha(path),metadata=metadata_name,
            artifact=f'{stem}_tinygrad.pkl',inputs=list(metadata['input_shapes']))
        bindings[role]['sha256']=roles[role]['onnx_sha256']
    export_report=json.loads((output/'export.json').read_text())
    export_report.update(exports=bindings,checkpoint_sha256=selection['source_checkpoint_sha256'],step=selection['step'])
    atomic_json(output/'export.json',export_report)
    return roles


def verification_trace(output,roles,validation,*,ticks=120):
    if ticks<8: raise ValueError('verification requires at least eight dense ticks')
    sessions={role:runtime(onnx.load(output/entry['onnx'])) for role,entry in roles.items()}
    parity=[];arrays={};history=deque([np.zeros(512,np.float16)]*100,maxlen=100)
    cameras=deque([np.zeros((2,6,128,256),np.uint8)]*5,maxlen=5)
    pulses=np.zeros((100,8),np.float16);previous_desire=np.zeros(8,np.float16)
    with SequenceEpisode(validation[0]['path']) as episode:
        if episode.manifest['split']!='validation':raise ValueError('runtime trace must use held-out validation')
        if sha(episode.path)!=validation[0]['sha256']:raise ValueError('runtime trace manifest changed')
        if len(episode)<ticks:raise ValueError('validation episode is shorter than the required recurrent trace')
        for tick in range(min(ticks,len(episode))):
            cameras.append(episode.frame(tick))
            vi={name:np.concatenate([cameras[0][k],cameras[-1][k]])[None] for k,name in enumerate(('img','big_img'))}
            vision=sessions['vision'].run(None,vi)[0]
            history.append(vision[0,1064:].copy())
            desire=np.array(episode.frames[tick]['desire'],np.float16);desire[0]=0
            pulses[:-1]=pulses[1:];pulses[-1]=np.where(desire-previous_desire>.99,desire,0);previous_desire=desire
            pi=dict(features_buffer=np.stack(list(history)[3::4])[None],desire_pulse=pulses.reshape(25,4,8).max(1)[None],
                traffic_convention=np.array([episode.frames[tick]['traffic_convention']],np.float16))
            policy=sessions['policy'].run(None,pi)[0]
            for role,inputs,values in (('vision',vi,vision),('policy',pi,policy)):
                if not np.isfinite(values).all():raise ValueError('nonfinite exported ONNX output')
                for name,value in dict(**inputs,outputs=values).items():arrays.setdefault(f'{role}.{name}',[]).append(value.copy())
    for role,entry in roles.items():
        path=output/entry['onnx'];graph=onnx.load(path)
        recreated=ReleasedModel(path,precision='float32',expected_sha256=sha(path))
        x={name:torch.from_numpy(arrays[f'{role}.{name}'][-1]).float() if arrays[f'{role}.{name}'][-1].dtype==np.float16
            else torch.from_numpy(arrays[f'{role}.{name}'][-1]) for name in entry['inputs']}
        with torch.no_grad():actual=recreated(x).numpy()
        expected=runtime(float32_graph(graph)).run(None,{k:v.numpy() for k,v in x.items()})[0]
        passed=bool(np.allclose(actual,expected,atol=2e-4,rtol=2e-4))
        parity.append(dict(role=role,passed=passed,max_abs=float(np.max(np.abs(actual-expected)))))
        if not passed:raise ValueError(f'{role} exported ONNX failed independent numerical verification')
    np.savez_compressed(output/'verification.npz',**{k:np.stack(v) for k,v in arrays.items()})
    result=dict(passed=True,frames=len(arrays['policy.outputs']),manifest=validation[0]['path'],
        manifest_sha256=validation[0]['sha256'],precision='FP16 source ONNX trace; FP32 independent export parity',parity=parity)
    atomic_json(output/'onnx-verification.json',result)
    return result


def bundle_template(roles,selection):
    bundle=json.loads(DEFAULT_CHECKPOINT.read_text())
    ref=f"cd210-centerline-s{selection['step']}-{selection['source_checkpoint_sha256'][:12]}"
    bundle.update(displayName=f"CD210 Centerline s{selection['step']}",internalName=ref,ref=ref,status='notDownloading',models=[])
    for role,entry in roles.items():
        spec=dict(type=role)
        for kind,name in (('artifact',entry['artifact']),('metadata',entry['metadata'])):
            spec[kind]=dict(fileName=name,downloadUri=dict(uri=f'file:///data/media/0/models/{name}',sha256=''))
        bundle['models'].append(spec)
    return bundle


def prepare(run,releases):
    releases=Path(releases).resolve()
    with run_lock(releases):
        return _prepare(Path(run).resolve(),releases)


def _prepare(run,releases):
    torch.set_num_threads(2)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+uuid4().hex[:8]
    output=releases/stamp;output.mkdir(parents=True)
    atomic_json(output/'status.json',dict(stage='selecting_checkpoint',runtime_ready=False,driving_qualified=False))
    try:
        model,selection,validation=select_checkpoint(run,output,releases)
        if not selection['promote_export']:
            record=json.loads((releases/'best-export.json').read_text())
            retained=(releases/record['checkpoint']).resolve().parent
            from .device_build import check_files
            check_files(retained,json.loads((retained/'payload.json').read_text()))
            atomic_json(output/'status.json',dict(stage='retained_previous_best',directory=str(retained),
                reason='no comparable strict improvement',runtime_ready=False,driving_qualified=False))
            atomic_json(releases/'last-request.json',dict(directory=str(retained),comparison=str(output)))
            event('retained_previous_best',directory=str(retained),step=selection['step'])
            return retained
        roles=export_models(model,selection,output)
        verification=verification_trace(output,roles,validation)
    except Exception as error:
        atomic_json(output/'status.json',dict(stage='local_export_failed',error=str(error),onnx_ready=False,
            runtime_ready=False,driving_qualified=False))
        raise
    shutil.copyfile(Path(__file__).with_name('device_build.py'),output/'device_build.py')
    names=['device_build.py','verification.npz']+[v[k] for v in roles.values() for k in ('onnx','metadata')]
    manifest=dict(schema_version=1,device='c3x',backend='QCOM',roles=roles,bundle=bundle_template(roles,selection),
        selected_checkpoint_sha256=selection['source_checkpoint_sha256'],step=selection['step'],
        files={name:sha(output/name) for name in names},onnx_verification=verification)
    atomic_json(output/'payload.json',manifest)
    record=dict(checkpoint=str((output/'checkpoint.pt').relative_to(releases)),
        checkpoint_sha256=sha(output/'checkpoint.pt'),source_checkpoint_sha256=selection['source_checkpoint_sha256'],
        step=selection['step'],cohort=selection['cohort'],loss=selection['loss'],kind=selection['kind'],
        protected_cohorts=selection['protected_cohorts'])
    immutable_json(output/'release-record.json',record)
    if selection['promote_export']:
        previous=releases/'best-export.json'
        if previous.exists():
            immutable_json(releases/'history'/(uuid4().hex+'.json'),json.loads(previous.read_text()))
        immutable_json(releases/'history'/(uuid4().hex+'.json'),record)
        atomic_json(previous,record)
    atomic_json(output/'status.json',dict(stage='onnx_ready_device_compile_pending',onnx_ready=True,runtime_ready=False,driving_qualified=False,
        checkpoint_step=selection['step'],checkpoint_kind=selection['kind'],validation_loss=selection['loss'],baseline_loss=selection['baseline_loss']))
    atomic_json(releases/'last-request.json',dict(directory=str(output),step=selection['step'],runtime_ready=False))
    if selection['promote_export']:
        atomic_json(releases/'latest.json',dict(directory=str(output),step=selection['step'],runtime_ready=False))
    event('onnx_ready',directory=str(output),step=selection['step'])
    return output


def ssh_command(host,arguments):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+',host):raise ValueError('use an explicit user@hostname SSH address')
    return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join([str(a) for a in arguments])]


def compile_remote(output,device):
    output=Path(output).resolve()
    with run_lock(output.parent):
        return _compile_remote(output,device)


def _compile_remote(output,device):
    manifest=json.loads((output/'payload.json').read_text())
    from .device_build import check_files
    check_files(output,manifest)
    status_path=output/'status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    if status.get('runtime_ready'):
        archive=Path(status['archive']).resolve()
        if output not in archive.parents or sha(archive)!=status['archive_sha256']:
            raise ValueError('previous successful compiled archive failed integrity verification')
        event('c3x_package_already_ready',archive=str(archive))
        return
    attempt=uuid4().hex
    build=output/'builds'/attempt
    build.mkdir(parents=True)
    remote=f'/data/media/0/cd210-centerline-exports/{output.name}-{attempt}'
    try:
        subprocess.run(ssh_command(device,['mkdir','-p',remote]),check=True,timeout=20,capture_output=True,text=True)
    except (subprocess.SubprocessError,OSError) as error:
        atomic_json(output/'device-pending.json',dict(device=device,error=str(error),runtime_ready=False,
            resume_command=f'python -m lane_release.export --resume "{output}" --device {device}'))
        raise DeviceUnavailable(f'c3x unavailable; verified ONNX retained at {output}. Resume this package when {device} is online.') from error
    for name in ['payload.json',*manifest['files']]:
        subprocess.run(['scp','-q','-o','BatchMode=yes',str(output/name),f'{device}:{remote}/{name}'],check=True,timeout=240)
    command=ssh_command(device,['/usr/local/venv/bin/python',f'{remote}/device_build.py','--package',remote])
    event('compiling_on_c3x',device=device,remote=remote)
    with (build/'device-build.log').open('x',encoding='utf-8') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        for line in process.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
        code=process.wait()
    transfer=subprocess.run(['scp','-q','-r','-o','BatchMode=yes',f'{device}:{remote}/compiled',str(build)],timeout=240)
    if transfer.returncode or not (build/'compiled/device-report.json').exists():
        raise RuntimeError(f'c3x build exited {code} without a complete report; see {build / "device-build.log"}')
    report=json.loads((build/'compiled/device-report.json').read_text())
    if report['source_payload_sha256']!=sha(output/'payload.json'):raise ValueError('device compiled a different source package')
    for name,digest in report['artifacts'].items():
        if sha(build/'compiled'/name)!=digest:raise ValueError('downloaded compiled artifact hash mismatch')
    if code or not report['runtime_ready']:raise RuntimeError('c3x runtime checks failed; see device-report.json')
    archive=build/'cd210-c3x.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as stream:
        for path in sorted((build/'compiled').glob('*')):
            if path.is_file():stream.write(path,path.name)
    atomic_json(output/'status.json',dict(stage='c3x_runtime_verified',onnx_ready=True,runtime_ready=True,driving_qualified=False,
        archive=str(archive),archive_sha256=sha(archive),device=device,checkpoint_step=manifest['step']))
    (output/'device-pending.json').unlink(missing_ok=True)
    latest=output.parent/'latest.json'
    if latest.exists() and json.loads(latest.read_text()).get('directory')==str(output):
        atomic_json(latest,dict(directory=str(output),step=manifest['step'],runtime_ready=True,archive=str(archive)))
        runtime_record=dict(directory=str(output),step=manifest['step'],runtime_ready=True,
            archive=str(archive),archive_sha256=sha(archive))
        best_runtime=output.parent/'best-runtime.json'
        if best_runtime.exists():
            immutable_json(output.parent/'history'/(uuid4().hex+'.json'),json.loads(best_runtime.read_text()))
        immutable_json(output.parent/'history'/(uuid4().hex+'.json'),runtime_record)
        atomic_json(best_runtime,runtime_record)
    event('c3x_package_ready',archive=str(archive),driving_qualified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--releases',type=Path,default=DEFAULT_RELEASES)
    parser.add_argument('--device',default=DEFAULT_DEVICE)
    parser.add_argument('--onnx-only',action='store_true')
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    args.releases=args.releases.resolve()
    output=args.resume.resolve() if args.resume else prepare(args.run_dir.resolve(),args.releases)
    if not args.onnx_only:
        try:compile_remote(output,args.device)
        except DeviceUnavailable as error:
            event('device_offline_compile_pending',directory=str(output),device=args.device,message=str(error))
            return 3
    return 0


if __name__=='__main__':
    raise SystemExit(main())
