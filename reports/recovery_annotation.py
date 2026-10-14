"""Annotate the complete recorded straight recovery tail before label selection.

The geometric proposal's ten metres are a minimum, not the end of the
recorded recovery. Preserve the proposal and extend only through a verified
straight, physically driven tail, ending five metres before terminal braking.
No candidate acceptance results are used to choose the annotation.
"""
import copy
import json
from pathlib import Path
import numpy as np
from lane_dataset.config import write_json,sha256_file
from lane_dataset.geometry import LaneReference
from lane_dataset.admission import shard_postprocess_lock


def recovery_annotation(manifest,geometry,telemetry):
    if manifest.get('family')=='straight_recovery':return None
    phases=manifest.get('maneuver_phase_ranges') or {}
    stop=manifest.get('terminal_suffix_policy',{}).get('stop_route_m')
    if stop is None or 'recovery' not in phases:return None
    old=float(phases['recovery'][1]);end=float(stop)-5.
    if end<=old+1e-6:return None
    arc=np.asarray(geometry['arc_m']);curvature=np.asarray(geometry['curvature_1pm'])
    segment=(arc>=old-.5)&(arc<=end+.5)
    if segment.sum()<2 or np.max(np.abs(curvature[segment]))>.004:return None
    if any(max(old,float(a))<min(end,float(b)) for a,b in geometry.get('junction_ranges_m',[])):return None
    progress=np.asarray(telemetry['route_s']);selected=np.flatnonzero((progress>=old)&(progress<=end))
    if not len(selected) or progress[-1]<end or not np.all(telemetry['command_accepted'][selected]):return None
    reference=LaneReference(geometry['left_xyz'],geometry['right_xyz'],arc)
    offsets=[reference.project(telemetry['position_world'][i,:2],hint_s=float(progress[i]))[1] for i in selected]
    if max(abs(float(v)) for v in offsets)>.15:return None
    updated=copy.deepcopy(phases);updated['recovery'][1]=end
    return dict(original_phase_ranges=phases,annotated_phase_ranges=updated,
        method='complete measured straight recovery tail; five metre guard before declared terminal braking',
        evidence_indices=[int(selected[0]),int(selected[-1])],maximum_tail_curvature_1pm=float(np.max(np.abs(curvature[segment]))),
        maximum_tail_center_offset_m=max(abs(float(v)) for v in offsets),uses_label_acceptance_results=False)


def annotate_recorded_recovery(shard):
    shard=Path(shard)
    with shard_postprocess_lock(shard):
        manifest=json.loads((shard/'manifest.json').read_text())
        if manifest.get('status')=='accepted':raise ValueError('accepted shards are immutable')
        geometry=json.loads((shard/'geometry.json').read_text())
        with np.load(shard/'telemetry.npz',allow_pickle=False) as values:telemetry={k:values[k] for k in values.files}
        annotation=recovery_annotation(manifest,geometry,telemetry)
        if annotation is None:return None
        annotation['bindings']={name:sha256_file(shard/name) for name in ('geometry.json','telemetry.npz','requested-task.json')}
        write_json(shard/'recovery-annotation.json',annotation)
        manifest['maneuver_phase_ranges']=annotation['annotated_phase_ranges']
        manifest['intended_progress_m']=annotation['annotated_phase_ranges']['recovery'][1]-float(telemetry['route_s'][0])
        manifest['files']['recovery-annotation.json']=sha256_file(shard/'recovery-annotation.json')
        write_json(shard/'manifest.json',manifest)
        return annotation
