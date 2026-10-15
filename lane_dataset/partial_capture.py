"""Recover complete, independently qualifying events from interrupted captures."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import time

import numpy as np

from .config import sha256_file, write_json
from .preprocessing import reference_module


def recover_capture_prefix(shard):
    """Keep original payloads; rebuild only a complete event and its future suffix.

    This is preparation for ordinary admission, never an admission bypass.
    Incomplete events remain recoverable diagnostics instead of being deleted.
    """
    shard=Path(shard)
    manifest=json.loads((shard/'manifest.json').read_text())
    if not manifest.get('error'):return False
    source=shard/'partial-telemetry.npz'
    if not source.exists():return False
    with np.load(source,allow_pickle=False) as archive:telemetry={k:archive[k] for k in archive.files}
    count=len(telemetry['timestamp'])
    terminal=manifest.get('terminal_suffix_policy',{}).get('stop_route_m',float('inf'))
    invalid=np.flatnonzero(~telemetry['command_accepted'].astype(bool)&(telemetry['route_s']<terminal))
    if len(invalid):count=min(count,int(invalid[0]))
    for event in manifest.get('collision_events',[]):
        count=min(count,int(np.searchsorted(telemetry['frame_id'],event['frame'])))
    if count<301:return False
    phases=manifest.get('maneuver_phase_ranges') or {}
    if not phases or telemetry['route_s'][count-1]<phases['recovery'][1]:return False
    with gzip.open(shard/'candidates.json.gz','rt',encoding='utf-8') as stream:candidates=json.load(stream)
    candidates=[c for c in candidates if c['index']<=count-201]
    from .admission import select_phase_indices
    _,_,errors=select_phase_indices([c['index'] for c in candidates if c.get('command_valid')],telemetry['route_s'],phases)
    if errors:return False
    if manifest.get('traffic_count',0):return False  # Traffic ledger needs independent truncation support.
    module=reference_module('frame_archive_v2')
    frames=[json.loads(line) for line in (shard/'frames.jsonl').read_text().splitlines() if line.strip()]
    if len(frames)!=len(telemetry['timestamp']):raise ValueError('partial capture frame/telemetry mismatch')
    readers={}
    try:
        for cam in ('narrow','wide'):
            readers[cam]=module.FrameArchiveReader(shard/f'packed_{cam}.npz',require_complete=False)
            if len(readers[cam])<count:raise ValueError('partial camera archive missing retained frames')
        # Prepare all outputs before touching the interrupted originals.
        for cam,reader in readers.items():
            with module.FrameArchiveWriter(shard/f'recovered_{cam}.npz',dtype=reader.dtype,
                    frame_shape=reader.frame_shape,expected_count=count) as writer:
                for index in range(count):writer.write(index,reader[index])
                writer.finalize(complete=True)
    finally:
        for reader in readers.values():reader.close()
    originals=shard/('interrupted-payload-'+str(time.time_ns()))
    originals.mkdir()
    original_names=('manifest.json','packed_narrow.npz','packed_wide.npz','frames.jsonl',
        'evidence.jsonl.gz','command-provenance.jsonl.gz','candidates.json.gz',
        'rendered-boundary-samples.json.gz','rendered-boundary-report.json')
    for name in original_names:
        path=shard/name
        if path.exists():path.rename(originals/name)
    for cam in readers:(shard/f'recovered_{cam}.npz').rename(shard/f'packed_{cam}.npz')
    ids=set(telemetry['frame_id'][:count].tolist())
    (shard/'frames.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames[:count]),encoding='utf-8')
    for name in ('evidence.jsonl.gz','command-provenance.jsonl.gz'):
        with gzip.open(originals/name,'rt',encoding='utf-8') as src,gzip.open(shard/name,'wt',encoding='utf-8') as dst:
            for line in src:
                value=json.loads(line);frame=value.get('reference',value)['frame_id']
                if frame in ids:dst.write(line)
    with gzip.open(shard/'candidates.json.gz','wt',encoding='utf-8') as stream:json.dump(candidates,stream)
    np.savez_compressed(shard/'telemetry.npz',**{k:v[:count] for k,v in telemetry.items()})
    from .rendered_boundaries import summarize_frames
    with gzip.open(originals/'rendered-boundary-samples.json.gz','rt',encoding='utf-8') as stream:boundary=json.load(stream)
    boundary=[r for r in boundary if r['frame_id'] in ids]
    with gzip.open(shard/'rendered-boundary-samples.json.gz','wt',encoding='utf-8') as stream:json.dump(boundary,stream)
    report=summarize_frames(boundary)
    report['samples_sha256']=sha256_file(shard/'rendered-boundary-samples.json.gz')
    write_json(shard/'rendered-boundary-report.json',report)
    manifest['capture_interruption']=dict(error=manifest.pop('error'),original_payload_directory=originals.name,
        original_frames=len(frames),retained_frames=count,admission_required=True)
    manifest['collision_events']=[]  # All collision frames and subsequent frames were excluded above.
    manifest['audit_frames']=[r for r in manifest.get('audit_frames',[]) if r['index']<count]
    manifest['archives']={cam:dict(path=f'packed_{cam}.npz',sha256=sha256_file(shard/f'packed_{cam}.npz')) for cam in readers}
    manifest['measured_throughput']=dict(ticks=count,simulated_seconds=count*.05,
        scope='retained uninterrupted complete event; original failure and payloads preserved')
    manifest['files']={p.relative_to(shard).as_posix():sha256_file(p) for p in shard.rglob('*') if p.is_file() and p.name!='manifest.json'}
    write_json(shard/'manifest.json',manifest)
    return True
