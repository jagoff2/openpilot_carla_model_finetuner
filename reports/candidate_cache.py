"""Stream and compress the candidate catalogue without retaining it all in RAM."""
import argparse,json,sqlite3,zlib,os
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


def ensure_rankings(path):
    """Build a small replaceable index; never rewrite the capture catalogue."""
    from reports.candidate_priority import boundary_risk, RANKING_VERSION
    from collections import Counter
    path=Path(path).resolve();target=path.with_suffix('.queue.sqlite')
    signature=json.dumps([RANKING_VERSION,path.stat().st_size,path.stat().st_mtime_ns])
    if target.exists():
        with sqlite3.connect(f'file:{target.as_posix()}?mode=ro',uri=True) as previous:
            row=previous.execute('SELECT value FROM metadata WHERE key=?',('source',)).fetchone()
        previous.close()
        if row and row[0]==signature:return target
    temporary=target.with_name(target.name+f'.{os.getpid()}.tmp')
    counts=Counter()
    source=sqlite3.connect(f'file:{path.as_posix()}?mode=ro',uri=True)
    output=sqlite3.connect(temporary)
    try:
        output.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT)')
        output.execute('CREATE TABLE ranking (id TEXT PRIMARY KEY,risk INTEGER,gap REAL,missing REAL,family INTEGER,test INTEGER,ticks INTEGER)')
        for i,(identity,priority,ticks,payload) in enumerate(source.execute('SELECT id,priority,ticks,payload FROM candidates'),1):
            task=json.loads(zlib.decompress(payload));risk=boundary_risk(task)
            output.execute('INSERT INTO ranking VALUES (?,?,?,?,?,?,?)',
                (identity,risk['tier'],risk['maximum_unpainted_gap_m'],1.-risk['painted_fraction'],
                 priority,int(task['town']=='Town05'),ticks))
            counts[risk['tier']]+=1
            if i%1000==0:
                output.commit();print(json.dumps(dict(event='candidate_ranking_progress',candidates=i)),flush=True)
        output.execute('INSERT INTO metadata VALUES (?,?)',('source',signature))
        output.execute('INSERT INTO metadata VALUES (?,?)',('counts',json.dumps(dict(counts))))
        output.commit()
    finally:
        output.close();source.close()
    temporary.replace(target)
    print(json.dumps(dict(event='candidate_ranking_ready',path=str(target),counts=dict(counts))),flush=True)
    return target


def read_cache(path, *, test=None):
    ranking=ensure_rankings(path)
    db=sqlite3.connect(f'file:{Path(path).as_posix()}?mode=ro',uri=True)
    try:
        db.execute('ATTACH DATABASE ? AS queue',(f'file:{ranking.as_posix()}?mode=ro',))
        # First give each family and test/train map pool a candidate with good
        # boundary metadata. Short unmarked hairpins cannot monopolize both
        # workers ahead of thousands of observable curves/straights. All quotas
        # and geometry isolation are still checked by the publication owner.
        query='''WITH ordered AS (
            SELECT id,risk,family,test,ROW_NUMBER() OVER (
                PARTITION BY risk,family,test ORDER BY gap,missing,ticks,id) AS ordinal
            FROM queue.ranking)
            SELECT candidates.payload FROM ordered JOIN candidates USING(id)
            WHERE (? IS NULL OR test=?)
            ORDER BY risk,ordinal,family,test,id'''
        for payload, in db.execute(query, (test, test)):
            yield json.loads(zlib.decompress(payload))
    finally:
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
