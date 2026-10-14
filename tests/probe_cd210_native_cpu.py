"""Actual exported CD210 graphs through packaged Tinygrad on CPU, not QCOM."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True,help='Installed matching package_runtime directory')
    parser.add_argument('--clang',default='clang');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(root),str(args.runtime)]
    os.environ.update(DEV='CPU',CC=args.clang,JIT_BATCH_SIZE='0',BEAM='0')
    from lane_release.device_build import compile_role,check_files
    from tinygrad import Device
    assert Device.DEFAULT=='CPU'
    manifest=json.loads((args.package/'payload.json').read_text());check_files(args.package,manifest)
    data=np.load(args.package/'verification.npz',allow_pickle=False)
    args.output.mkdir(parents=True,exist_ok=True);reports=[]
    for role in ('policy','vision'):
        entry=manifest['roles'][role]
        inputs={k:data[f'{role}.{k}'][0] for k in entry['inputs']}
        print(json.dumps(dict(stage='compile_cpu',role=role)),flush=True)
        run,convert,contract=compile_role(args.package/entry['onnx'],inputs,args.output/f'{role}-CPU-only.pkl')
        comparisons=[]
        for tick in (0,40,119):
            values={k:data[f'{role}.{k}'][tick] for k in entry['inputs']}
            actual=run(**convert(values)).numpy();expected=data[f'{role}.outputs'][tick]
            passed=bool(np.allclose(actual,expected,atol=.05,rtol=.02))
            comparisons.append(dict(tick=tick,passed=passed,max_abs=float(np.abs(actual-expected).max())))
        reports.append(dict(role=role,contract=contract,comparisons=comparisons))
        print(json.dumps(reports[-1]),flush=True)
    result=dict(passed=all(c['passed'] for r in reports for c in r['comparisons']),backend='CPU',
        qcom_compiled=False,target_device_executed=False,roles=reports,atol=.05,rtol=.02)
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
