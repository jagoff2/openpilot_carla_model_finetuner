"""Read-only replay of retained QA evidence. This never admits or edits a shard."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from lane_dataset.config import write_json
from lane_dataset.geometry import LaneReference
from lane_dataset.rendered_boundaries import (
    check_spatial_resolution, edge_center, stripe_center, summarize_frames,
)


def boundary_replay(shard):
    with gzip.open(shard/'rendered-boundary-samples.json.gz','rt') as stream:
        frames=json.load(stream)
    changed=0
    for frame in frames:
        for camera in frame['cameras'].values():
            offsets=camera['offsets_m']
            for sample in camera['samples']:
                if sample.get('proof_omitted'):continue
                bits=lambda name:np.unpackbits(np.asarray(sample[name],np.uint8))[:len(offsets)].astype(bool)
                residual=np.asarray([np.nan if x is None else x for x in sample['depth_residual_m']])
                result=stripe_center(offsets,bits('marking_bits'),residual,
                    expected_width_m=sample.get('expected_marking_width_m'))
                if not result['measured'] and 'road_bits' in sample:
                    result=edge_center(offsets,bits('road_bits'),bits('exterior_bits'),
                                       sample['luma'],residual,sample['side'])
                if 'cross_section_meters_per_pixel' in sample:
                    result=check_spatial_resolution(result,sample['cross_section_meters_per_pixel'])
                changed+=any(sample.get(k)!=v for k,v in result.items())
                sample.update(result)
    replay=summarize_frames(frames)
    return dict(changed_measurements=changed,**{k:v for k,v in replay.items()
        if not isinstance(v,(dict,list)) or k=='errors'})


def warmup_replay(shard):
    geometry=json.loads((shard/'geometry.json').read_text())
    reference=LaneReference(geometry['left_xyz'],geometry['right_xyz'],geometry['arc_m'])
    with np.load(shard/'bootstrap-telemetry.npz',allow_pickle=False) as raw:
        hint=reference.s[0];progress=[]
        for position in raw['position_world']:
            hint,_=reference.project(position[:2],hint);progress.append(hint)
        return dict(frames=len(progress),
            original_maximum_frame_progress_m=float(np.max(np.abs(np.diff(raw['route_s'])))),
            continuous_maximum_frame_progress_m=float(np.max(np.abs(np.diff(progress)))),
            changed_frames=int(np.count_nonzero(np.abs(raw['route_s']-progress)>1e-6)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--job',action='append',default=[])
    parser.add_argument('--warmup-shard',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=dict(scope='offline diagnostic replay only; no admission or modification of source evidence',jobs={})
    for job in args.job:
        result=json.loads((args.root/'reports/automatic'/(job+'.job.result.json')).read_text())
        shard=Path(result['shard'])
        report['jobs'][job]=dict(shard=str(shard),original_errors=result.get('errors'),
            original_boundary_maximum_m=result.get('boundary',{}).get('maximum_disagreement_m'),
            corrected_boundary_replay=boundary_replay(shard),
            original_lateral_metrics=result.get('physical',{}).get('metrics',{}))
    if args.warmup_shard:
        report['warmup']=dict(shard=str(args.warmup_shard),**warmup_replay(args.warmup_shard))
    write_json(args.output,report)
    print(json.dumps(dict(output=str(args.output),jobs=list(report['jobs']))))


if __name__=='__main__':main()
