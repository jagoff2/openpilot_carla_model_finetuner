"""Finish an existing recorder, consolidate its files, and resume the original root."""
from pathlib import Path
import argparse,json,os,shutil,time
import psutil
from lane_dataset.config import Config,PROJECT,DEFAULT_ROOT,sha256_file,write_json
from lane_dataset.storage import Store
from lane_dataset.simulator import world_owner
from reports.automatic_corpus import read,log,run,DEFAULT_PLAN,snapshot

ORIGINAL=DEFAULT_ROOT.resolve()
TEMPORARY=ORIGINAL.with_name(ORIGINAL.name+'-recovery')

def rebase(value,source,target):
    if isinstance(value,dict):return {key:rebase(item,source,target) for key,item in value.items()}
    if isinstance(value,list):return [rebase(item,source,target) for item in value]
    if isinstance(value,str):
        for old,new in ((str(source),str(target)),(source.as_posix(),target.as_posix())):
            if value.lower()==old.lower() or value.lower().startswith((old+'\\').lower()) or value.lower().startswith((old+'/').lower()):
                return new+value[len(old):]
    return value

def merge_files(source,target):
    """Move an owned tree without overwriting any different destination file."""
    source=Path(source);target=Path(target)
    if not source.exists():return
    target.mkdir(parents=True,exist_ok=True)
    for path in sorted(source.rglob('*')):
        if not path.is_file():continue
        destination=target/path.relative_to(source);destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            if sha256_file(path)!=sha256_file(destination):raise ValueError('relocation_destination_conflict:'+str(destination))
            continue
        path.rename(destination)

def consolidate(source,target):
    source=Path(source).resolve();target=Path(target).resolve()
    if source!=TEMPORARY or target!=ORIGINAL:raise ValueError('relocation roots do not match the explicit original and temporary directories')
    if source==target or source in target.parents or target in source.parents:raise ValueError('overlapping relocation roots')
    old_config=read(target/'config.json');new_config=read(source/'config.json')
    if old_config['sha256']!=new_config['sha256'] or old_config['sha256']!=Config().digest():raise ValueError('relocation_configuration_mismatch')
    destination_store=Store(target);origin_store=Store(source)
    with world_owner(2400,source),world_owner(2400,target):
        rows=origin_store.rows()
        if any(row['status']=='active' for row in rows):raise ValueError('cannot relocate a recorder with active reservations')
        if any(row['status']=='accepted' for row in rows):raise ValueError('handoff requires admission after relocation')
        merge_files(source/'calibrations',target/'calibrations')
        merge_files(source/'reports/calibration-measured-3p5',target/'reports/calibration-measured-3p5')
        moved=[]
        for row in rows:
            origin=Path(row['path']).resolve()
            if origin.parent!=source/'quarantine':raise ValueError('unexpected source ledger location')
            destination=target/'quarantine'/origin.name
            if destination.exists():raise FileExistsError(destination)
            manifest=read(origin/'manifest.json')
            for name,digest in manifest.get('files',{}).items():
                path=(origin/name).resolve()
                if origin not in path.parents or sha256_file(path)!=digest:raise ValueError('source capture hash mismatch:'+name)
            origin.rename(destination)
            changed=[]
            for path in destination.rglob('*.json'):
                document=read(path);updated=rebase(document,source,target)
                if document!=updated:write_json(path,updated);changed.append(str(path.relative_to(destination)))
            manifest=read(destination/'manifest.json')
            write_json(destination/'relocation.json',dict(source=str(origin),destination=str(destination),
                camera_and_telemetry_payloads_unchanged=True,rebased_json_files=changed,method='same-volume move after recorder completion; hashes checked before relocation'))
            manifest['files']={name:sha256_file(destination/name) for name in manifest['files']}
            manifest['files']['relocation.json']=sha256_file(destination/'relocation.json')
            write_json(destination/'manifest.json',manifest)
            with destination_store.transaction() as db:
                if db.execute('SELECT id FROM runs WHERE id=?',(row['id'],)).fetchone():raise ValueError('destination ledger identity conflict')
                row['path']=str(destination)
                db.execute('INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,?)',tuple(row[key] for key in ('id','episode','attempt','worker','pid','status','reserved','path','created','reason')))
            moved.append(dict(episode=row['episode'],path=str(destination)))
        origin_output=source/'reports/automatic';destination_output=target/'reports/automatic'
        for path in list(origin_output.iterdir()):
            if not path.is_file() or path.name in ('status.json','state.json') or path.suffix=='.tmp':continue
            destination=destination_output/path.name
            if destination.exists():raise FileExistsError(destination)
            path.rename(destination)
            if destination.suffix=='.json' and not destination.name.startswith('screened-pool'):
                write_json(destination,rebase(read(destination),source,target))
        state=rebase(read(origin_output/'state.json'),source,target);state['active']=None;state['status']='running'
        write_json(destination_output/'state.json',state)
        write_json(destination_output/'relocation-report.json',dict(passed=True,source=str(source),destination=str(target),moved=moved,existing_destination_data_preserved=True))
    # Release source file locks before renaming their containing Windows folder.
    # Retain remaining setup files and the old ledger as a relocation audit.
    # No recursive deletion is used and the two exact roots were checked above.
    audit=destination_output/'relocation-source'
    if audit.exists():raise FileExistsError(audit)
    # SQLite transaction contexts can leave collectable connection objects alive
    # until GC; Windows keeps their database paths open until finalization.
    __import__('gc').collect()
    source.rename(audit)
    return moved

def main():
    global ORIGINAL,TEMPORARY
    parser=argparse.ArgumentParser(description='Explicit recovery utility, never used by normal collection startup.')
    parser.add_argument('--previous-supervisor',type=int,required=True)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--target',type=Path,required=True)
    args=parser.parse_args();ORIGINAL=args.target.resolve();TEMPORARY=args.source.resolve()
    source=TEMPORARY;target=ORIGINAL;output=target/'reports/automatic';output.mkdir(parents=True,exist_ok=True)
    state=read(source/'reports/automatic/state.json');active=state.get('active') or {}
    try:supervisor=psutil.Process(args.previous_supervisor)
    except psutil.NoSuchProcess:supervisor=None
    if supervisor is not None:
        command=supervisor.cmdline()
        if 'reports.automatic_corpus' not in command or '--job' in command or str(source) not in command:raise ValueError('supervisor identity mismatch')
    worker=psutil.Process(active['pid']);worker_command=worker.cmdline()
    if 'reports.automatic_corpus' not in worker_command or '--job' not in worker_command:raise ValueError('worker identity mismatch')
    if supervisor is not None:supervisor.terminate();supervisor.wait(timeout=15)
    log('restore_original_output',root=str(target),preserved_recorder_pid=worker.pid)
    foreign_log=Path(active['log']);state['active']={**active,'stage':'finishing_existing_recording_before_relocation'}
    with foreign_log.open(encoding='utf-8',errors='replace') as reader:
        while True:
            for line in reader:print(line,end='',flush=True)
            snapshot(target,state)
            if not worker.is_running():break
            time.sleep(2)
    # The independent offline scanner writes only the temporary root. Let its
    # current catalog finish before moving that root, with progress visible.
    catalog=source/'reports/automatic/screened-pool-status.json'
    while catalog.exists() and not read(catalog).get('complete'):
        state['active']=dict(stage='finishing_offline_candidate_catalog_before_relocation',catalog=read(catalog))
        snapshot(target,state);log('candidate_catalog',**read(catalog));time.sleep(10)
    log('relocating_completed_capture')
    moved=consolidate(source,target);log('original_output_restored',root=str(target),episodes=moved)
    # Start a fresh interpreter so a long recording does not retain an older
    # supervisor implementation after the handoff.
    import subprocess,sys
    code=subprocess.call([sys.executable,'-u','-m','reports.automatic_corpus','--root',str(target)],cwd=PROJECT)
    raise SystemExit(code)

if __name__=='__main__':main()
