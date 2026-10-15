"""Verify a mixed-precision pair from already compiled candidate components."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fp16',type=Path,required=True)
    parser.add_argument('--fp32',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--half-role',choices=['vision','policy'],default='vision')
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False)
    spec=importlib.util.spec_from_file_location('device_build',Path(__file__).with_name('device_build.py'))
    build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)
    os.environ.update(DEV='QCOM',IMAGE='1',FLOAT16='0',NOLOCALS='1',JIT_BATCH_SIZE='0',QCOM_PRIORITY='8',BEAM='0')
    environment=build.preflight(Path('/data/iqpilot'))
    environment.update(build.configure_model_cpu())
    from iqpilot.selfdrive.iqmodeld.tools.firmware import patch_firmware_fetch
    patch_firmware_fetch()
    from tinygrad import Tensor,Device
    assert Device.DEFAULT=='QCOM'
    manifests={k:json.loads((p/'payload.json').read_text()) for k,p in [('fp16',args.fp16),('fp32',args.fp32)]}
    reports={k:json.loads((p/'compiled/device-report.json').read_text()) for k,p in [('fp16',args.fp16),('fp32',args.fp32)]}
    for key,source in [('fp16',args.fp16),('fp32',args.fp32)]:
        build.check_files(source,manifests[key])
        assert reports[key]['source_payload_sha256']==build.sha(source/'payload.json')
        assert manifests[key]['selected_checkpoint_sha256']==manifests['fp32']['selected_checkpoint_sha256']
    manifest=copy.deepcopy(manifests['fp32'])
    manifest.update(precision=f'mixed: {args.half_role} FP16, other role FP32',
        reference_precision='exact FP32 checkpoint tensors and recurrence',
        source_payloads={k:reports[k]['source_payload_sha256'] for k in reports})
    output=args.output/'compiled';output.mkdir()
    runtimes={};files={};contracts={}
    for role in ('vision','policy'):
        key='fp16' if role==args.half_role else 'fp32'
        source=args.fp16 if key=='fp16' else args.fp32
        entry=copy.deepcopy(manifests[key]['roles'][role]);manifest['roles'][role]=entry
        for name in (entry['onnx'],entry['metadata']):
            shutil.copyfile(source/name,args.output/name)
        for kind in ('artifact','metadata'):
            name=entry[kind]
            if build.sha(source/'compiled'/name)!=reports[key]['artifacts'][name]:
                raise RuntimeError('compiled component hash mismatch')
            os.link(source/'compiled'/name,output/name)
            files[name]=build.sha(output/name)
        with (output/entry['artifact']).open('rb') as stream:runtimes[role]=pickle.load(stream)
        contracts[role]=reports[key]['contracts'][role]
        for model in manifest['bundle']['models']:
            if model['type']==role:
                for kind in ('artifact','metadata'):
                    name=entry[kind]
                    model[kind]=dict(fileName=name,downloadUri=dict(uri='file:///data/media/0/models/'+name,sha256=files[name]))
    shutil.copyfile(args.fp32/'verification.npz',args.output/'verification.npz')
    shutil.copyfile(__file__,args.output/'mixed_device.py')
    shutil.copyfile(Path(__file__).with_name('device_build.py'),args.output/'device_build.py')
    manifest['files']={p.name:build.sha(p) for p in args.output.iterdir() if p.is_file()}
    (args.output/'payload.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with np.load(args.fp32/'verification.npz',allow_pickle=False) as archive:
        reference={name:archive[name] for name in archive.files}
    def convert(inputs):
        return {k:Tensor(v.astype(np.uint8 if 'img' in k else np.float32),device='QCOM' if 'img' in k else 'NPY').realize()
                for k,v in sorted(inputs.items())}
    feature_queue=[np.zeros(512,np.float32) for _ in range(100)]
    predictions=[];latencies=[];role_times={'vision':[],'policy':[]}
    for index in range(len(reference['policy.outputs'])):
        build.preflight(Path('/data/iqpilot'))
        started=time.perf_counter()
        inputs={k:reference['vision.'+k][index] for k in manifest['roles']['vision']['inputs']}
        vision=runtimes['vision'](**convert(inputs)).numpy()
        boundary=time.perf_counter();role_times['vision'].append(boundary-started)
        feature_queue.pop(0);feature_queue.append(vision[0,1064:].copy())
        inputs={k:reference['policy.'+k][index] for k in manifest['roles']['policy']['inputs']}
        inputs['features_buffer']=np.stack(feature_queue[3::4])[None]
        policy=runtimes['policy'](**convert(inputs)).numpy()
        ended=time.perf_counter();role_times['policy'].append(ended-boundary)
        predictions.append(policy[0]);latencies.append(ended-started)
        if index%20==0:print(json.dumps(dict(stage='mixed_verify',half_role=args.half_role,tick=index)),flush=True)
    actual=np.stack(predictions)
    result=build.assess(reference['policy.outputs'][:,0,:],actual,latencies,backend='QCOM',lateral_precision=True)
    difference=np.abs(actual[:,:495].reshape(-1,33,15)-reference['policy.outputs'][:,0,:495].reshape(-1,33,15))
    result.update(precision=manifest['precision'],acceptance='lateral absolute tolerances plus whole-policy relative agreement',
        frame_count=len(predictions),environment=environment,
        contracts=contracts,artifacts=files,source_payload_sha256=build.sha(args.output/'payload.json'),
        tinygrad_sources=reports['fp32']['tinygrad_sources'],
        axis_metrics={name:dict(p95=float(np.percentile(difference[:,:,axis],95)),maximum=float(difference[:,:,axis].max()))
                      for axis,name in [(0,'x'),(1,'y')]},
        role_latency_p95_ms={role:float(np.percentile(times[4:],95)*1000) for role,times in role_times.items()})
    (output/'device-report.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(output/'device-trace.npz',policy=actual,latency_s=np.asarray(latencies))
    if result['runtime_ready']:
        (output/'bundle.json').write_text(json.dumps(manifest['bundle'],indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('runtime_ready','metrics','axis_metrics','role_latency_p95_ms')}),flush=True)
    return 0 if result['runtime_ready'] else 2


if __name__=='__main__':raise SystemExit(main())
