"""Append-only model storage and comparable, non-regressing best promotion."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

from lane_dataset.config import sha256_file, write_json

SELECTION_METRIC = 'lateral_geometry_v1'


def lateral_score(geometry):
    """Lateral quality only; forward progress remains a separate training guard."""
    from .centerline import LossConfig
    cfg = LossConfig()
    return float(cfg.lateral_weight * geometry['lateral_loss']
        + cfg.heading_weight * geometry['heading_loss']
        + cfg.clearance_weight * geometry['clearance_loss'])


def selection_report(optimization_loss, baseline_optimization_loss, geometry, baseline_geometry):
    return dict(selection_metric=SELECTION_METRIC,
        loss=lateral_score(geometry), baseline_loss=lateral_score(baseline_geometry),
        optimization_loss=float(optimization_loss),
        baseline_optimization_loss=float(baseline_optimization_loss),
        progress_guard_loss=float(geometry['progress_loss']),
        baseline_progress_guard_loss=float(baseline_geometry['progress_loss']))


def immutable_copy(source, destination, expected=None):
    """Publish without replacement; even a concurrent writer cannot clobber a file."""
    source, destination = Path(source), Path(destination)
    expected = expected or sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != expected:
            raise ValueError(f'immutable artifact differs: {destination}')
        return expected
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.partial', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with source.open('rb') as reader:
                shutil.copyfileobj(reader, stream)
            stream.flush(); os.fsync(stream.fileno())
        except BaseException:
            stream.close(); temporary.unlink(missing_ok=True)
            raise
    try:
        if sha256_file(temporary) != expected:
            raise ValueError('checkpoint changed while being preserved')
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if sha256_file(destination) != expected:
                raise ValueError(f'immutable artifact differs: {destination}')
    finally:
        temporary.unlink(missing_ok=True)
    return expected


def immutable_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / ('.record-' + uuid4().hex + '.json')
    try:
        write_json(temporary, document)
        immutable_copy(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_path(directory, record):
    directory = Path(directory).resolve()
    filename = record.get('checkpoint', record.get('file'))
    if not isinstance(filename, str):
        raise ValueError('checkpoint pointer has no filename')
    path = (directory / filename).resolve()
    legacy = path.parent == directory and path.name in ('best.pt', 'checkpoint-0.pt', 'checkpoint-1.pt')
    archived = (path.parent == directory / 'preserved' / 'checkpoints'
                and path.name == record['sha256'] + '.pt')
    if not (legacy or archived):
        raise ValueError('unexpected checkpoint filename')
    return path


def preserve(directory, pointer, *, validation=None, kind='validated_candidate'):
    directory = Path(directory).resolve()
    source = checkpoint_path(directory, pointer)
    destination = directory / 'preserved' / 'checkpoints' / (pointer['sha256'] + '.pt')
    immutable_copy(source, destination, pointer['sha256'])
    record = dict(validation or {}, checkpoint=str(destination.relative_to(directory)),
                  sha256=pointer['sha256'], step=pointer['step'], kind=kind)
    immutable_json(directory / 'preserved' / 'records' / (uuid4().hex + '.json'), record)
    return record


def scores(result):
    rows = result.get('rows', result.get('validation', []))
    return {r['manifest_sha256']: dict(loss=r['loss'], labels=r['labels']) for r in rows}


def can_promote(candidate, incumbent=None):
    """Strict current-cohort win, with no regression on earlier promotion cohorts."""
    if not all(math.isfinite(candidate[k]) for k in ('loss', 'baseline_loss')):
        raise ValueError('promotion requires finite validation scores')
    if incumbent is not None and candidate.get('selection_metric') != incumbent.get('selection_metric'):
        raise ValueError('incumbent must be rescored with the same selection metric')
    if not candidate['loss'] < candidate['baseline_loss']:
        return False
    if incumbent is None:
        return True
    if candidate['cohort'] != incumbent['cohort']:
        raise ValueError('incumbent must be evaluated on the same validation cohort')
    if not math.isfinite(incumbent['loss']):
        raise ValueError('incumbent validation score is nonfinite')
    if candidate['loss'] >= incumbent['loss']:
        return False
    new, old = scores(candidate), scores(incumbent)
    for cohort in incumbent.get('protected_cohorts', []):
        if not cohort or not set(cohort) <= new.keys() or not set(cohort) <= old.keys():
            raise ValueError('previous promotion validation data is missing')
        if any(new[k]['labels'] != old[k]['labels'] or new[k]['labels'] <= 0
               or not math.isfinite(new[k]['loss']) or not math.isfinite(old[k]['loss']) for k in cohort):
            raise ValueError('validation support changed for a protected cohort')
        if sum(new[k]['loss'] * new[k]['labels'] for k in cohort) > sum(old[k]['loss'] * old[k]['labels'] for k in cohort):
            return False
    return True


def protected_cohorts(candidate, incumbent=None):
    result = list((incumbent or {}).get('protected_cohorts', []))
    cohort = sorted(scores(candidate))
    if cohort and cohort not in result:
        result.append(cohort)
    return result


def protect_validation(directory, pointer, result, incumbent=None):
    """Save every better-than-stock validation candidate; move only the best pointer."""
    directory = Path(directory)
    best_path = directory / 'best.json'
    old = json.loads(best_path.read_text()) if best_path.exists() else None
    if old is not None:
        # Also imports a legacy best.pt before anything can replace its pointer.
        preserve(directory, old, validation=old, kind='previous_best')
        if incumbent is None or incumbent.get('sha256') != old['sha256']:
            raise ValueError('missing comparison with the current best checkpoint')
    promoted = can_promote(result, incumbent)
    if result['loss'] < result['baseline_loss']:
        record = preserve(directory, pointer, validation=result)
        if promoted:
            record['protected_cohorts'] = protected_cohorts(result, incumbent)
            immutable_json(directory / 'preserved' / 'promotions' / (uuid4().hex + '.json'), record)
            write_json(best_path, record)
    if not promoted and old is not None and (
            old.get('selection_metric') != incumbent.get('selection_metric')
            or old['cohort'] != incumbent['cohort']):
        # Refresh scores for the same immutable winner, never its model bytes.
        refreshed = preserve(directory, old, validation=incumbent, kind='previous_best')
        immutable_json(directory / 'preserved' / 'rescored' / (uuid4().hex + '.json'), refreshed)
        write_json(best_path, refreshed)
    return promoted
