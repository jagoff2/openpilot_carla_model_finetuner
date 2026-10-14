import gzip,zipfile
from reports.compact_payloads import recompress


def test_lossless_recompression_preserves_exact_evidence_and_all_frame_members(tmp_path):
    payload=(b'{"observations":[1,2,3,4,5],"valid":true}\n')*10000
    path=tmp_path/'evidence.jsonl.gz'
    with gzip.open(path,'wb',compresslevel=1) as stream:stream.write(payload)
    result=recompress(path)
    assert gzip.decompress(path.read_bytes())==payload
    assert result['after_bytes']<result['before_bytes']
    archive=tmp_path/'frames.npz'
    members={'index.json':b'{"format":"original metadata"}','frames/000000.npy':payload,'frames/000001.npy':payload[::-1]}
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as stream:
        for name,value in members.items():stream.writestr(name,value)
    recompress(archive)
    with zipfile.ZipFile(archive) as stream:
        assert {name:stream.read(name) for name in stream.namelist()}==members
