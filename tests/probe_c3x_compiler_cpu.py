"""Run separately with Python 3.12 and the installed packaged Tinygrad runtime.

Exercises the production serialization function on changed inputs using CPU.
This is deliberately not evidence of QCOM compilation or device readiness.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True,help='Installed matching package_runtime directory')
    parser.add_argument('--clang',default='clang')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(root),str(args.runtime)]
    os.environ.update(DEV='CPU',CC=args.clang,JIT_BATCH_SIZE='0',BEAM='0')
    from lane_release.device_build import compile_role
    from tinygrad import Device
    assert Device.DEFAULT=='CPU'
    reports=[]
    for role in ('vision','policy'):
        name='img' if role=='vision' else 'features_buffer'
        dtype=np.uint8 if role=='vision' else np.float16
        values={name:np.arange(4,dtype=dtype).reshape(1,4)}
        run,convert,contract=compile_role(args.fixture/f'{role}.onnx',values,args.fixture/f'{role}.pkl')
        first=run(**convert(values)).numpy()
        changed=run(**convert({name:values[name]+2})).numpy()
        np.testing.assert_array_equal(first,values[name].astype(np.float32)*2+1)
        np.testing.assert_array_equal(changed,(values[name].astype(np.float32)+2)*2+1)
        reports.append(dict(role=role,contract=contract,reloaded_and_changed_input_verified=True))
    result=dict(passed=True,backend='CPU',qcom_compiled=False,target_device_executed=False,roles=reports)
    (args.fixture/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
