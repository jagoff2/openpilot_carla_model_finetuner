import gzip
import json

import numpy as np
import pytest

from lane_dataset.config import write_json
from lane_dataset.partial_capture import recover_capture_prefix
from lane_dataset.preprocessing import reference_module


def gz(path,value):
    with gzip.open(path,'wt',encoding='utf-8') as stream:json.dump(value,stream)


def fixture(shard,count=600):
    shard.mkdir()
    phases={name:[10+i*4,14+i*4] for i,name in enumerate(('approach','entry','apex','exit','recovery'))}
    manifest=dict(error={'message':'late control rejection'},maneuver_phase_ranges=phases,collision_events=[],
        audit_frames=[dict(index=100),dict(index=599)])
    write_json(shard/'manifest.json',manifest)
    accepted=np.ones(count,bool);accepted[-20:]=False
    np.savez_compressed(shard/'partial-telemetry.npz',timestamp=np.arange(count)*.05,
        frame_id=np.arange(count)+1000,route_s=np.arange(count)*.1,command_accepted=accepted)
    gz(shard/'candidates.json.gz',[dict(index=i,command_valid=True) for i in range(100,count-200,10)])
    frames=[dict(index=i,frame_id=i+1000,timestamp=i*.05) for i in range(count)]
    (shard/'frames.jsonl').write_text(''.join(json.dumps(f)+'\n' for f in frames))
    for name in ('evidence.jsonl.gz','command-provenance.jsonl.gz'):
        with gzip.open(shard/name,'wt',encoding='utf-8') as stream:
            for f in frames:stream.write(json.dumps({'reference':f} if name.startswith('evidence') else f)+'\n')
    gz(shard/'rendered-boundary-samples.json.gz',frames)
    write_json(shard/'rendered-boundary-report.json',{})
    module=reference_module('frame_archive_v2')
    for cam in ('narrow','wide'):
        with module.FrameArchiveWriter(shard/f'packed_{cam}.npz',dtype=np.uint8,frame_shape=(2,3),expected_count=count+100) as writer:
            for i in range(count):writer.write(i,np.full((2,3),i%256,np.uint8))
    return manifest


def test_late_failure_recovers_full_event_without_bad_tail_or_original_loss(tmp_path,monkeypatch):
    import lane_dataset.rendered_boundaries as boundaries
    monkeypatch.setattr(boundaries,'summarize_frames',lambda frames:dict(passed=True,frame_count=len(frames)))
    shard=tmp_path/'capture';original=fixture(shard)
    assert recover_capture_prefix(shard)
    m=json.loads((shard/'manifest.json').read_text())
    assert 'error' not in m and m['capture_interruption']['retained_frames']==580
    old=shard/m['capture_interruption']['original_payload_directory']
    assert json.loads((old/'manifest.json').read_text())==original
    assert (old/'packed_narrow.npz').is_file()
    assert len((old/'frames.jsonl').read_text().splitlines())==600
    assert len((shard/'frames.jsonl').read_text().splitlines())==580
    with reference_module('frame_archive_v2').FrameArchiveReader(shard/'packed_narrow.npz') as reader:
        assert len(reader)==580
        np.testing.assert_array_equal(reader[579],np.full((2,3),579%256,np.uint8))
    with np.load(shard/'telemetry.npz') as t:assert t['command_accepted'].all()
    assert m['audit_frames']==[dict(index=100)]
    assert not recover_capture_prefix(shard)


def test_incomplete_event_is_preserved_and_never_relabelled_complete(tmp_path):
    shard=tmp_path/'capture';original=fixture(shard,count=320)
    assert not recover_capture_prefix(shard)
    assert json.loads((shard/'manifest.json').read_text())==original
    assert (shard/'packed_narrow.npz').is_file()
    assert not (shard/'telemetry.npz').exists()
