"""Select verified 2 Hz candidates with coverage, without changing any label."""
import gzip,json,shutil
from pathlib import Path
import numpy as np
from lane_dataset.config import write_json,sha256_file
from lane_dataset.admission import shard_postprocess_lock


def choose_indices(candidates,rejections,progress,phases):
    chosen=[]
    for phase,(lo,hi) in phases.items():
        valid=[v for v in candidates if not rejections.get(str(v['index']),['unverified'])
            and (lo<=progress[v['index']]<hi or phase=='recovery' and progress[v['index']]==hi)]
        if len(valid)<4:return None
        crop=[v['index'] for v in valid if (v.get('support') or {}).get('category')=='history_crop_loss']
        other=[v['index'] for v in valid if v['index'] not in crop]
        selected=[]
        for pool in (crop,other):
            if len(selected)==4:break
            count=min(4-len(selected),len(pool))
            selected.extend(pool[i] for i in np.unique(np.rint(np.linspace(0,len(pool)-1,count)).astype(int)))
        chosen.extend(selected)
    return sorted(set(chosen)) if len(set(chosen))==20 else None


def improve_candidate_coverage(shard,report):
    shard=Path(shard)
    with shard_postprocess_lock(shard):
        m=json.loads((shard/'manifest.json').read_text())
        if m.get('status')=='accepted':raise ValueError('accepted shards are immutable')
        if (shard/'candidate-selection.json').exists() or report['build_errors']:return False
        path=shard/m['candidates_file']
        with gzip.open(path,'rt',encoding='utf-8') as stream:candidates=json.load(stream)
        with np.load(shard/'telemetry.npz') as values:progress=values['route_s']
        indices=choose_indices(candidates,report['candidate_rejections'],progress,m['maneuver_phase_ranges'])
        if indices is None:return False
        selected=[v for v in candidates if v['index'] in indices]
        potential=sum(v['support']['category']=='history_crop_loss' for v in selected)
        current=sum(v['visibility_category']=='history_crop_loss' for v in m['label_records'])
        if potential<=current:return False
        original=shard/'candidate-pool.original.json.gz';shutil.copy2(path,original)
        temporary=path.with_name(path.name+'.selection.tmp')
        with gzip.open(temporary,'wt',encoding='utf-8',compresslevel=9) as stream:json.dump(selected,stream,separators=(',',':'),allow_nan=False)
        temporary.replace(path)
        write_json(shard/'candidate-selection.json',dict(method='four verified candidates per phase, preferring measured historical crop-loss support',
            indices=indices,original_sha256=sha256_file(original),candidate_sha256=sha256_file(path),
            original_selected_crop_loss_count=current,proposed_crop_loss_count=potential,
            scope='selection only; original candidate values preserved verbatim; admission replays every selected label'))
        for name in (path.name,original.name,'candidate-selection.json'):m['files'][name]=sha256_file(shard/name)
        write_json(shard/'manifest.json',m)
        return True
