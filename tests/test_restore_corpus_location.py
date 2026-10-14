from lane_dataset.config import Config,write_json,sha256_file
from lane_dataset.storage import Store
from reports import restore_corpus_location as restore


def test_completed_capture_moves_to_original_root_without_changing_camera_payload(tmp_path,monkeypatch):
    source=tmp_path/'temporary';target=tmp_path/'original'
    monkeypatch.setattr(restore,'TEMPORARY',source.resolve());monkeypatch.setattr(restore,'ORIGINAL',target.resolve())
    destination_store=Store(target);origin_store=Store(source)
    for root in (source,target):
        write_json(root/'config.json',dict(sha256=Config().digest(),frozen=False))
        (root/'reports/automatic').mkdir()
    (target/'existing-data.txt').write_text('preserve')
    token,stage=origin_store.reserve('episode',0,100000)
    (stage/'packed_narrow.npz').write_bytes(b'exact camera payload')
    write_json(stage/'manifest.json',dict(status='development',accepted_label_count=0,
        calibration_binding=dict(path=str(source/'calibrations/fit.json')),
        files={'packed_narrow.npz':sha256_file(stage/'packed_narrow.npz'),'reservation.json':sha256_file(stage/'reservation.json')}))
    captured=origin_store.finish(token,False)
    write_json(source/'reports/automatic/state.json',dict(completed={},active=dict(pid=1),status='running'))
    write_json(source/'reports/automatic/episode.job.result.json',dict(shard=str(captured),status='measured_candidate_passed'))
    moved=restore.consolidate(source,target)
    assert not source.exists() and (target/'existing-data.txt').read_text()=='preserve'
    from pathlib import Path
    shard=Path(moved[0]['path']);manifest=restore.read(shard/'manifest.json')
    assert (shard/'packed_narrow.npz').read_bytes()==b'exact camera payload'
    assert manifest['calibration_binding']['path']==str(target/'calibrations/fit.json')
    assert all(sha256_file(shard/name)==digest for name,digest in manifest['files'].items())
    assert destination_store.rows()[0]['path']==str(shard)
    assert restore.read(target/'reports/automatic/episode.job.result.json')['shard']==str(shard)
