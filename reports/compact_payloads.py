"""Losslessly recompress completed captures before immutable admission."""
import gzip,hashlib,json,shutil,zipfile
from pathlib import Path
from lane_dataset.config import sha256_file,write_json
from lane_dataset.admission import shard_postprocess_lock


def stream_hash(stream):
    digest=hashlib.sha256()
    for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def recompress(path):
    path=Path(path);temporary=path.with_name(path.name+'.recompressed')
    before=path.stat().st_size;original=sha256_file(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rb') as source,gzip.open(temporary,'wb',compresslevel=9) as dest:shutil.copyfileobj(source,dest,1024*1024)
        with gzip.open(path,'rb') as source,gzip.open(temporary,'rb') as dest:
            if stream_hash(source)!=stream_hash(dest):raise ValueError('recompression changed uncompressed evidence')
    else:
        with zipfile.ZipFile(path) as source,zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as dest:
            for name in source.namelist():dest.writestr(name,source.read(name))
        with zipfile.ZipFile(path) as source,zipfile.ZipFile(temporary) as dest:
            if source.namelist()!=dest.namelist():raise ValueError('recompression changed archive members')
            for name in source.namelist():
                if source.read(name)!=dest.read(name):raise ValueError('recompression changed frame bytes')
    after=temporary.stat().st_size
    if after<before:temporary.replace(path)
    else:temporary.unlink();after=before
    return dict(path=path.name,before_bytes=before,after_bytes=after,original_sha256=original,
        sha256=sha256_file(path),uncompressed_payloads_identical=True)


def compact_capture(shard):
    shard=Path(shard)
    with shard_postprocess_lock(shard):
        m=json.loads((shard/'manifest.json').read_text())
        if m.get('status')=='accepted':raise ValueError('accepted shards are immutable')
        if (shard/'lossless-compaction.json').exists():return json.loads((shard/'lossless-compaction.json').read_text())
        names=('packed_narrow.npz','packed_wide.npz','evidence.jsonl.gz','rendered-boundary-samples.json.gz','candidates.json.gz','command-provenance.jsonl.gz')
        results=[]
        for name in names:
            path=shard/name
            if not path.exists():continue
            if m['files'].get(name)!=sha256_file(path):raise ValueError('capture artifact hash mismatch before compaction:'+name)
            result=recompress(path);results.append(result);m['files'][name]=result['sha256']
            print(json.dumps(dict(event='lossless_compaction',**result)),flush=True)
        for value in m['archives'].values():value['sha256']=sha256_file(shard/value['path'])
        boundary=json.loads((shard/'rendered-boundary-report.json').read_text())
        boundary['samples_sha256']=sha256_file(shard/'rendered-boundary-samples.json.gz')
        write_json(shard/'rendered-boundary-report.json',boundary)
        m['files']['rendered-boundary-report.json']=sha256_file(shard/'rendered-boundary-report.json')
        report=dict(files=results,before_bytes=sum(v['before_bytes'] for v in results),after_bytes=sum(v['after_bytes'] for v in results),
            method='DEFLATE level nine with byte-for-byte decompressed member verification; no frames or evidence removed')
        write_json(shard/'lossless-compaction.json',report)
        m['files']['lossless-compaction.json']=sha256_file(shard/'lossless-compaction.json')
        write_json(shard/'manifest.json',m)
        return report
