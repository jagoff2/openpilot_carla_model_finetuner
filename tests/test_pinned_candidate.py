import json
import pytest

from lane_release.candidate import preserved_record, prepare_candidate


def test_requested_step_stays_pinned_after_best_advances(tmp_path):
    records=tmp_path/'preserved/records';records.mkdir(parents=True)
    previous=dict(step=24,sha256='saved24',rows=[{'loss':1.}])
    (records/'previous.json').write_text(json.dumps(previous))
    (tmp_path/'best.json').write_text(json.dumps(dict(step=32,sha256='new32')))
    assert preserved_record(tmp_path,24)==previous
    with pytest.raises(ValueError,match='unique'):preserved_record(tmp_path,999)
    (records/'ambiguous.json').write_text(json.dumps({**previous,'sha256':'other24'}))
    with pytest.raises(ValueError,match='unique'):preserved_record(tmp_path,24)


def test_failed_pinned_export_cannot_replace_previous_best(tmp_path):
    run=tmp_path/'run';run.mkdir();releases=tmp_path/'releases';releases.mkdir()
    pointer=releases/'best-export.json';pointer.write_bytes(b'preserved best pointer')
    artifact=releases/'previous.onnx';artifact.write_bytes(b'preserved model')
    checksum='a'*64
    source=run/'preserved/checkpoints'/(checksum+'.pt');source.parent.mkdir(parents=True)
    source.write_bytes(b'damaged checkpoint')
    (run/'best.json').write_text(json.dumps(dict(step=24,sha256=checksum,
        checkpoint=str(source.relative_to(run)),rows=[])))
    with pytest.raises(ValueError,match='changed while being preserved'):
        prepare_candidate(run,releases,24)
    assert pointer.read_bytes()==b'preserved best pointer'
    assert artifact.read_bytes()==b'preserved model'
