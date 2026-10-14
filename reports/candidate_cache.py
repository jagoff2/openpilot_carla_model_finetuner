"""Stream and compress the candidate catalogue without retaining it all in RAM."""
import argparse,json,sqlite3,zlib
from pathlib import Path
from reports.inspect_recorded_turn import array_items

PRIORITY={'severe_hairpin':0,'constant_radius':1,'tightening_s_bend':2,'intersection_turn':3,'straight_recovery':4}


def build_cache(source):
    source=Path(source);target=source.with_suffix('.sqlite');temporary=target.with_suffix('.sqlite.tmp')
    if target.exists():return target
    with sqlite3.connect(temporary) as db:
        db.execute('CREATE TABLE IF NOT EXISTS candidates (id TEXT PRIMARY KEY, priority INTEGER, ticks INTEGER, payload BLOB)')
        count=0
        for task in array_items(source):
            payload=json.dumps(task,separators=(',',':')).encode()
            db.execute('INSERT OR REPLACE INTO candidates VALUES (?,?,?,?)',(task['group_id'],PRIORITY[task['family']],task['minimum_ticks'],zlib.compress(payload,6)))
            count+=1
            if count%1000==0:db.commit();print(json.dumps(dict(cached_candidates=count)),flush=True)
        db.execute('CREATE INDEX IF NOT EXISTS queue_order ON candidates(priority,ticks,id)');db.commit()
    # sqlite context managers commit transactions but do not close connections.
    db.close();temporary.replace(target)
    print(json.dumps(dict(cache=str(target),candidates=count,bytes=target.stat().st_size)),flush=True)
    return target


def read_cache(path):
    with sqlite3.connect(f'file:{Path(path).as_posix()}?mode=ro',uri=True) as db:
        for payload, in db.execute('SELECT payload FROM candidates ORDER BY priority,ticks,id'):
            yield json.loads(zlib.decompress(payload))
    db.close()


def verify_and_remove_source(source):
    from lane_dataset.config import write_json,sha256_file
    source=Path(source).resolve()
    if source.name!='screened-pool.json' or source.parent.name!='automatic' or source.parent.parent.name!='reports':
        raise ValueError('only the generated automatic candidate catalogue can be retired')
    target=build_cache(source);count=0
    with sqlite3.connect(target) as db:
        for task in array_items(source):
            row=db.execute('SELECT payload FROM candidates WHERE id=?',(task['group_id'],)).fetchone()
            if row is None or json.loads(zlib.decompress(row[0]))!=task:raise ValueError('catalogue cache failed exact decoded comparison')
            count+=1
        if db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]!=count:raise ValueError('catalogue candidate count mismatch')
    db.close()
    proof=dict(source=str(source),source_sha256=sha256_file(source),source_bytes=source.stat().st_size,
        cache=str(target),cache_sha256=sha256_file(target),cache_bytes=target.stat().st_size,
        exact_decoded_candidates_verified=count,passed=True)
    write_json(source.with_name('candidate-cache-verification.json'),proof)
    source.unlink()
    print(json.dumps(proof),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('--verify-and-remove-source',action='store_true')
    args=parser.parse_args()
    (verify_and_remove_source if args.verify_and_remove_source else build_cache)(args.source)
