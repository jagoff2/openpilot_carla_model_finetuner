"""Read-only audit of physical paths and every independently hashed image frame."""
import argparse
import json
from pathlib import Path

import numpy as np

from lane_dataset.geometry import LaneReference
from lane_dataset.preprocessing import reference_module


def audit(root, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    report=dict(scope='read-only saved physical paths and complete per-member archive integrity; not image-content assessment',recordings=[])
    reader_type=reference_module('frame_archive_v2').FrameArchiveReader
    for shard in sorted((Path(root)/'accepted').iterdir()):
        if not shard.is_dir() or not (shard/'admission-report.json').exists():continue
        errors=[];row=dict(episode=shard.name,errors=errors,archives={})
        try:
            task=json.loads((shard/'requested-task.json').read_text())
            admission=json.loads((shard/'admission-report.json').read_text())
            geometry=json.loads((shard/'geometry.json').read_text())
            with np.load(shard/'telemetry.npz',allow_pickle=False) as values:
                telemetry={key:values[key] for key in values.files}
            reference=LaneReference(geometry['left_xyz'],geometry['right_xyz'],geometry['arc_m'])
            offset=np.array([reference.project(position[:2],hint_s=float(progress))[1]
                for position,progress in zip(telemetry['position_world'],telemetry['route_s'])])
            times=telemetry['timestamp']-telemetry['timestamp'][0]
            recovery=bool(task.get('recovery'));held=task.get('initial_recovery_offset_m')
            onset=5. if held else 0.
            eligible=times>=onset+4. if recovery else np.ones(len(times),bool)
            metric=dict(maximum_absolute_center_offset_m=float(np.max(np.abs(offset))),
                mean_signed_center_offset_m=float(np.mean(offset)),
                p95_absolute_center_offset_m=float(np.percentile(np.abs(offset),95)),
                maximum_absolute_offset_outside_recovery_window_m=float(np.max(np.abs(offset[eligible]))),
                mean_offset_outside_recovery_window_m=float(np.mean(offset[eligible])),
                p95_absolute_offset_outside_recovery_window_m=float(np.percentile(np.abs(offset[eligible]),95)))
            if metric['maximum_absolute_offset_outside_recovery_window_m']>.2:errors.append('center_offset_outside_recovery_window_exceeds_0.2m')
            if abs(metric['mean_offset_outside_recovery_window_m'])>.05:errors.append('mean_center_offset_outside_recovery_window_exceeds_0.05m')
            if metric['p95_absolute_offset_outside_recovery_window_m']>.1:errors.append('p95_center_offset_outside_recovery_window_exceeds_0.1m')
            row.update(family=task.get('family'),appearance=task.get('appearance'),
                declared_held_offset_m=held,recovery_window_end_s=onset+4. if recovery else None,
                offset_first_m=float(offset[0]),offset_at_5s_m=float(np.interp(5.,times,offset)),
                peak_command_steer_index=int(np.argmax(np.abs(telemetry['command_steer']))),
                max_abs_command_steer=float(np.abs(telemetry['command_steer']).max()),metrics=metric,
                accepted_physical_metrics=admission.get('physical',{}).get('metrics',{}))
            with np.load(shard/'labels.npz',allow_pickle=False) as labels:
                row['label_indices']=labels['label_indices'].tolist()
                row['labels_in_held_offset_prefix']=int(np.count_nonzero(times[labels['label_indices']]<onset-1e-6)) if held else 0
            for camera in ('narrow','wide'):
                name=f'packed_{camera}.npz'
                try:
                    with reader_type(shard/name,expected_archive_sha256=admission['artifact_hashes'][name]) as reader:
                        result=reader.validate()
                        if len(reader)!=len(times):raise ValueError('archive/telemetry count mismatch')
                    row['archives'][camera]=result
                except Exception as error:
                    row['archives'][camera]=dict(error=f'{type(error).__name__}: {error}')
                    errors.append(f'{camera}_archive_integrity_failed')
        except Exception as error:
            errors.append(f'{type(error).__name__}: {error}')
        report['recordings'].append(row)
        report['summary']=dict(recordings=len(report['recordings']),
            recordings_with_errors=sum(bool(r['errors']) for r in report['recordings']),
            archives_validated=sum(bool(a.get('all_frames_validated')) for r in report['recordings'] for a in r['archives'].values()),
            frames_validated=sum(a.get('written_count',0) for r in report['recordings'] for a in r['archives'].values() if a.get('all_frames_validated')))
        (output/'capture-archive-audit.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(dict(episode=shard.name,errors=errors,summary=report['summary'])),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();audit(args.root,args.output)
