"""Create a separate exact-FP32 export from a pinned CD210 candidate."""
import argparse
from collections import deque
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import numpy_helper
import torch

from .export import sha, atomic_json
from lane_training.cd210_model import load_checkpoint
from lane_training.verify_cd210 import float32_graph, runtime


def prepare(candidate, output):
    candidate, output = Path(candidate), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((candidate/'payload.json').read_text())
    selection = json.loads((candidate/'selection.json').read_text())['selected']
    if sha(candidate/'checkpoint.pt') != selection['export_checkpoint_sha256']:
        raise ValueError('pinned checkpoint changed')
    torch.set_num_threads(2)
    model, checkpoint = load_checkpoint(candidate/'checkpoint.pt')
    from lane_dataset.review_hold import assert_checkpoint_not_held
    state=checkpoint['training_state']
    assert_checkpoint_not_held(state['config']['corpus'],state)
    if checkpoint['step'] != manifest['step']:
        raise ValueError('checkpoint step mismatch')
    output.mkdir(parents=True)
    sessions = {}
    for role, entry in manifest['roles'].items():
        source = getattr(model,role)
        graph = float32_graph(onnx.load(candidate/entry['onnx']))
        for init in graph.graph.initializer:
            key=source.keys[init.name]
            tensor=(source.weights[key] if key in source.weights else getattr(source,key)).detach().cpu()
            value=(tensor.float() if tensor.is_floating_point() else tensor).numpy()
            init.CopyFrom(numpy_helper.from_array(value,init.name))
        onnx.checker.check_model(graph)
        onnx.save(graph,output/entry['onnx'])
        entry['onnx_sha256']=sha(output/entry['onnx'])
        old_metadata=entry['metadata']
        entry['artifact']=entry['artifact'].replace('_tinygrad.pkl','_fp32_tinygrad.pkl')
        entry['metadata']=entry['metadata'].replace('_metadata.pkl','_fp32_metadata.pkl')
        shutil.copyfile(candidate/old_metadata,output/entry['metadata'])
        sessions[role]=runtime(graph)
        for item in manifest['bundle']['models']:
            if item['type']==role:
                for kind in ('artifact','metadata'):
                    item[kind]['fileName']=entry[kind]
                    item[kind]['downloadUri']=dict(uri='file:///data/media/0/models/'+entry[kind],sha256='')
    with np.load(candidate/'verification.npz',allow_pickle=False) as archive:
        data={name:archive[name] for name in archive.files}
    history=deque([np.zeros(512,np.float32) for _ in range(100)],maxlen=100)
    arrays={}; parity=[]
    for tick in range(len(data['policy.outputs'])):
        vi={name:data['vision.'+name][tick] for name in manifest['roles']['vision']['inputs']}
        vision=sessions['vision'].run(None,vi)[0]
        history.append(vision[0,1064:].copy())
        pi={name:data['policy.'+name][tick].astype(np.float32) for name in ('desire_pulse','traffic_convention')}
        pi['features_buffer']=np.stack(list(history)[3::4])[None]
        policy=sessions['policy'].run(None,pi)[0]
        for role,inputs,values in (('vision',vi,vision),('policy',pi,policy)):
            for name,value in dict(**inputs,outputs=values).items():
                arrays.setdefault(f'{role}.{name}',[]).append(value.copy())
            if tick in (0,59,119):
                with torch.no_grad():actual=getattr(model,role)({k:torch.from_numpy(v) for k,v in inputs.items()}).numpy()
                passed=bool(np.allclose(actual,values,atol=2e-4,rtol=2e-4))
                parity.append(dict(role=role,tick=tick,passed=passed,max_abs=float(np.max(np.abs(actual-values)))))
                if not passed:raise ValueError('FP32 export does not match pinned checkpoint')
    np.savez_compressed(output/'verification.npz',**{k:np.stack(v) for k,v in arrays.items()})
    shutil.copyfile(Path(__file__).with_name('device_build.py'),output/'device_build.py')
    names=['device_build.py','verification.npz']+[v[k] for v in manifest['roles'].values() for k in ('onnx','metadata')]
    manifest.update(compiler_float16=0,precision='float32',parent_payload_sha256=sha(candidate/'payload.json'),
        files={name:sha(output/name) for name in names},
        onnx_verification=dict(passed=True,frames=len(data['policy.outputs']),precision='exact FP32 checkpoint tensors and recurrence',
            manifest=manifest['onnx_verification']['manifest'],manifest_sha256=manifest['onnx_verification']['manifest_sha256'],parity=parity))
    atomic_json(output/'payload.json',manifest)
    atomic_json(output/'status.json',dict(stage='onnx_ready_device_compile_pending',onnx_ready=True,runtime_ready=False,
        driving_qualified=False,checkpoint_step=manifest['step']))
    print(json.dumps(dict(output=str(output),step=manifest['step'],precision='float32',parity=parity)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.candidate,args.output)
