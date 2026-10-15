"""Standalone c3x compiler and verifier. Does not activate a model or modify driving software."""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import subprocess
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_files(root,manifest):
    root=Path(root).resolve()
    for name,expected in manifest['files'].items():
        path=(root/name).resolve()
        if root not in path.parents or sha(path)!=expected:
            raise ValueError(f'export payload integrity failed: {name}')


def preflight(iqroot):
    sys.path[:0]=[str(iqroot/'artifacts/package_runtime'),str(iqroot)]
    from iqpilot.common.params import Params
    from iqpilot.system.hardware import HARDWARE
    device=HARDWARE.get_device_type()
    if device!='tizi': raise RuntimeError(f'c3x (tizi) required, found {device}')
    if Params().get_bool('IsOnroad'): raise RuntimeError('device must be offroad for compilation and benchmarking')
    if not Path('/dev/kgsl-3d0').exists(): raise RuntimeError('c3x QCOM device unavailable')
    return dict(device_type=device,is_onroad=False,iqroot=str(iqroot),python=sys.version)


def configure_model_cpu():
    """Use modeld's big core for benchmarking, restoring offroad state at exit."""
    online=Path('/sys/devices/system/cpu/cpu7/online')
    previous=online.read_text().strip()
    affinity=os.sched_getaffinity(0)
    def write(value):
        subprocess.run(['sudo','-n','tee',str(online)],input=value+'\n',text=True,
            stdout=subprocess.DEVNULL,check=True,timeout=10)
    def restore():
        os.sched_setaffinity(0,affinity)
        if previous=='0':
            from iqpilot.common.params import Params
            if not Params().get_bool('IsOnroad'):write(previous)
    try:
        if previous=='0':write('1')
        os.sched_setaffinity(0,{7})
    except Exception:
        restore()
        raise
    atexit.register(restore)
    return dict(cpu_affinity=sorted(os.sched_getaffinity(0)),core7_previous_online=previous,
        restore_offroad_cpu_on_exit=True)


def compile_role(path,first_inputs,output_path):
    """Native split-model ABI: uint8 images on QCOM, float32 policy values on NPY."""
    from tinygrad import Tensor,TinyJit,Device,Context,dtypes
    from tinygrad.nn.onnx import OnnxRunner
    runner=OnnxRunner(path)
    def run(**inputs):
        return next(iter(runner({k:v.to(Device.DEFAULT) for k,v in inputs.items()}).values())).cast('float32')
    jit=TinyJit(run,prune=True)
    def convert(values):
        return {k:Tensor(v.astype(np.uint8 if 'img' in k else np.float32),
            device=Device.DEFAULT if 'img' in k else 'NPY').realize() for k,v in sorted(values.items())}
    values=convert(first_inputs)
    with Context(OPENPILOT_HACKS=1):
        eager=run(**values).numpy()
        for _ in range(3): actual=jit(**values).numpy()
    np.testing.assert_allclose(actual,eager,atol=1e-4,rtol=1e-4)
    with Path(output_path).open('xb') as stream: pickle.dump(jit,stream)
    with Path(output_path).open('rb') as stream: restored=pickle.load(stream)
    np.testing.assert_array_equal(restored(**values).numpy(),actual)
    captured={name:dict(dtype=str(info[2]),device=info[3])
        for name,info in zip(restored.captured.expected_names,restored.captured.expected_input_info,strict=True)}
    for (name,spec),info in zip(captured.items(),restored.captured.expected_input_info,strict=True):
        expected=Device.DEFAULT if 'img' in name else 'NPY'
        if spec['device']!=expected: raise ValueError(f'{name} captured on wrong device: {spec}')
        if 'img' in name and info[2] is not dtypes.uint8: raise ValueError('camera input must remain uint8')
    return restored,convert,captured


def assess(reference,actual,latencies,*,backend,lateral_precision=False):
    if reference.shape!=actual.shape or reference.ndim!=2 or reference.shape[1]!=1000:
        raise ValueError('invalid native policy trace shape')
    if len(latencies)!=len(reference) or len(latencies)<8:raise ValueError('at least eight paired timings are required')
    finite=bool(np.isfinite(actual).all() and np.isfinite(reference).all()
        and np.isfinite(latencies).all() and (np.asarray(latencies)>=0).all())
    if not finite:
        return dict(metrics=dict(finite=False),passed=False,runtime_ready=False,driving_qualified=False,
            scope='nonfinite model output or invalid runtime timing')
    difference=np.abs(actual[:,:495].reshape(-1,33,15)-reference[:,:495].reshape(-1,33,15))
    xy=difference[:,:,:2];yaw=difference[:,:,11]
    warm=np.asarray(latencies[4:],dtype=float)*1000
    metrics=dict(finite=finite,position_xy_p95_m=float(np.percentile(xy,95)),position_xy_max_m=float(xy.max()),
        yaw_p95_rad=float(np.percentile(yaw,95)),yaw_max_rad=float(yaw.max()),
        full_policy_allclose=bool(np.allclose(actual,reference,atol=.05,rtol=.02)),
        full_policy_max_abs=float(np.max(np.abs(actual-reference))),
        pair_latency_p95_ms=float(np.percentile(warm,95)),pair_latency_max_ms=float(warm.max()))
    thresholds=dict(position_xy_p95_m=.05,position_xy_max_m=.2,yaw_p95_rad=.005,yaw_max_rad=.02,pair_latency_p95_ms=50.)
    if lateral_precision:
        # Mixed precision retains the whole-policy relative agreement check.
        # Absolute centimetre limits apply to lateral placement, not forward x.
        metrics.update(position_y_p95_m=float(np.percentile(difference[:,:,1],95)),
            position_y_max_m=float(difference[:,:,1].max()))
        thresholds=dict(position_y_p95_m=.05,position_y_max_m=.2,yaw_p95_rad=.005,yaw_max_rad=.02,pair_latency_p95_ms=50.)
    passed=finite and metrics['full_policy_allclose'] and all(metrics[k]<=v for k,v in thresholds.items())
    return dict(metrics=metrics,thresholds=thresholds,passed=bool(passed),runtime_ready=bool(passed and backend=='QCOM'),
        driving_qualified=False,scope='compiled model pair correctness and latency, excluding camera warp and the rest of modeld; road behavior is not qualified')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--iqroot',type=Path,default=Path('/data/iqpilot'))
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    package=args.package.resolve();manifest=json.loads((package/'payload.json').read_text())
    check_files(package,manifest)
    os.environ.update(DEV='QCOM',IMAGE='1',FLOAT16=str(manifest.get('compiler_float16',1)),NOLOCALS='1',JIT_BATCH_SIZE='0',QCOM_PRIORITY='8',BEAM='0')
    environment=preflight(args.iqroot)
    if args.preflight_only:
        print(json.dumps(environment),flush=True);return 0
    # Match modeld's CPU affinity instead of inheriting sshd's little-core mask.
    environment.update(configure_model_cpu())
    os.environ.pop('IQPILOT_MODEL_STASH',None)
    from iqpilot.selfdrive.iqmodeld.tools.firmware import patch_firmware_fetch
    patch_firmware_fetch()
    import tinygrad
    from tinygrad import Device
    if Device.DEFAULT!='QCOM': raise RuntimeError('refusing a non-QCOM production artifact')
    output=package/'compiled';output.mkdir(exist_ok=False)
    # NPZ is lazy: repeated indexing decompresses the entire sequence. Load once
    # before timing so the runtime measurement includes inference, not archive IO.
    with np.load(package/'verification.npz',allow_pickle=False) as archive:
        reference={name:archive[name] for name in archive.files}
    runtimes={};contracts={};files={}
    for role in ('vision','policy'):
        preflight(args.iqroot)
        entry=manifest['roles'][role]
        first={k:reference[f'{role}.{k}'][0] for k in entry['inputs']}
        path=output/entry['artifact']
        print(json.dumps(dict(stage='compile',role=role)),flush=True)
        jit,convert,contract=compile_role(package/entry['onnx'],first,path)
        runtimes[role]=(jit,convert);contracts[role]=contract
        files[path.name]=sha(path)
        metadata=entry['metadata']
        data=(package/metadata).read_bytes();(output/metadata).write_bytes(data);files[metadata]=sha(output/metadata)
    predictions=[];latencies=[];feature_queue=[np.zeros(512,np.float32) for _ in range(100)]
    component_errors={role:[] for role in runtimes}
    for index in range(len(reference['policy.outputs'])):
        preflight(args.iqroot)
        started=time.perf_counter()
        vision_inputs={k:reference[f'vision.{k}'][index] for k in manifest['roles']['vision']['inputs']}
        run,convert=runtimes['vision'];vision=run(**convert(vision_inputs)).numpy()
        if not np.isfinite(vision).all(): raise ValueError('nonfinite compiled vision output')
        feature_queue.pop(0);feature_queue.append(vision[0,1064:].copy())
        policy_inputs={k:reference[f'policy.{k}'][index] for k in manifest['roles']['policy']['inputs']}
        policy_inputs['features_buffer']=np.stack(feature_queue[3::4])[None]
        run,convert=runtimes['policy'];policy=run(**convert(policy_inputs)).numpy()
        latencies.append(time.perf_counter()-started);predictions.append(policy[0])
        component_errors['vision'].append(float(np.max(np.abs(vision-reference['vision.outputs'][index]))))
        component_errors['policy'].append(float(np.max(np.abs(policy-reference['policy.outputs'][index]))))
        if index%20==0: print(json.dumps(dict(stage='verify',tick=index)),flush=True)
    result=assess(reference['policy.outputs'][:,0,:],np.stack(predictions),latencies,backend=Device.DEFAULT)
    runtime_root=Path(tinygrad.__file__).resolve().parent
    result.update(environment=environment,contracts=contracts,artifacts=files,frame_count=len(predictions),
        component_max_abs_error={k:max(v) for k,v in component_errors.items()},
        tinygrad_sources={str(p.relative_to(runtime_root)):sha(p) for p in sorted(runtime_root.rglob('*.py'))},
        source_payload_sha256=sha(package/'payload.json'))
    (output/'device-report.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(output/'device-trace.npz',policy=np.stack(predictions),latency_s=np.asarray(latencies))
    if not result['runtime_ready']:
        print(json.dumps(result['metrics']),flush=True);return 2
    bundle=manifest['bundle']
    for entry in bundle['models']:
        for kind in ('artifact','metadata'):
            name=entry[kind]['fileName'];entry[kind]['downloadUri']['sha256']=files[name]
    (output/'bundle.json').write_text(json.dumps(bundle,indent=2)+'\n')
    print(json.dumps(dict(runtime_ready=True,driving_qualified=False,metrics=result['metrics'])),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
