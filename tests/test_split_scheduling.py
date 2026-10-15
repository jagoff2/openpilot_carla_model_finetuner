import json
import sqlite3
import zlib

from lane_dataset.config import write_json
from reports.candidate_cache import read_cache
from reports.candidate_priority import preferred_split, scheduled_candidates


TARGETS = dict(train=4000, validation=500, test=500)


def catalogue(tmp_path, native=60, test=30):
    path = tmp_path / 'pool.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE candidates (id TEXT PRIMARY KEY,priority INTEGER,ticks INTEGER,payload BLOB)')
        for town, count in (('Town06', native), ('Town05', test)):
            for i in range(count):
                task = dict(group_id=f'{town}-{i:03}', family='constant_radius', town=town)
                db.execute('INSERT INTO candidates VALUES (?,?,?,?)',
                    (task['group_id'], 1, 600, zlib.compress(json.dumps(task).encode())))
    return path


def status(root, counts, workers=(), family_counts=None):
    write_json(root / 'reports/automatic/status.json', dict(
        accepted_labels=counts, accepted_by_family=family_counts or {}, active_workers=list(workers)))


def test_live_imbalance_prioritizes_training_and_excludes_full_splits():
    assert preferred_split(dict(train=40, validation=200, test=60), TARGETS, TARGETS) == 'train'
    assert preferred_split(dict(train=1600, validation=200, test=60), TARGETS, TARGETS) == 'test'
    assert preferred_split(TARGETS, TARGETS, TARGETS) is None


def test_filtered_cursors_keep_town05_exclusively_test(tmp_path):
    path = catalogue(tmp_path, native=3, test=2)
    assert {t['town'] for t in read_cache(path, test=False)} == {'Town06'}
    assert {t['town'] for t in read_cache(path, test=True)} == {'Town05'}
    assert len(list(read_cache(path))) == 5


def test_queue_recovers_imbalance_and_preserves_existing_validation(tmp_path):
    path = catalogue(tmp_path, native=3, test=2)
    status(tmp_path, dict(train=40, validation=200, test=60))
    write_json(tmp_path / 'reports/scenario-plan.automatic.json', dict(assignments=[
        dict(group_id='Town06-000', split='validation')]))
    queue = scheduled_candidates(path, tmp_path)
    try:
        task, split = next(queue)
        assert split == 'train' and task['group_id'] == 'Town06-001'
        # A concurrent test capture counts toward the target before publication.
        status(tmp_path, dict(train=160, validation=20, test=0), workers=[
            dict(job='in-flight-test', split='test', family='constant_radius')])
        task, split = next(queue)
        assert split == 'train' and task['town'] != 'Town05'
    finally:
        queue.close()


def test_repeated_acceptance_converges_to_80_10_10_without_geometry_leakage(tmp_path):
    path = catalogue(tmp_path)
    counts = dict(train=0, validation=0, test=0)
    assignments = []
    status(tmp_path, counts)
    queue = scheduled_candidates(path, tmp_path)
    try:
        for _ in range(30):
            task, split = next(queue)
            assert (task['town'] == 'Town05') == (split == 'test')
            assert task['group_id'] not in {r['group_id'] for r in assignments}
            assignments.append(dict(group_id=task['group_id'], split=split))
            counts[split] += 40 if split == 'train' else 20
            write_json(tmp_path / 'reports/scenario-plan.automatic.json', dict(assignments=assignments))
            status(tmp_path, counts)
        assert counts == dict(train=800, validation=100, test=100)
    finally:
        queue.close()


def test_family_quota_does_not_block_other_split_cursor(tmp_path):
    path = catalogue(tmp_path, native=3, test=2)
    status(tmp_path, dict(train=800, validation=200, test=60),
           family_counts={'train/constant_radius': 800})
    queue = scheduled_candidates(path, tmp_path)
    try:
        candidate, split = next(queue)
        assert split == 'test'
        status(tmp_path, dict(train=800, validation=200, test=500),
               family_counts={'train/constant_radius': 800})
        candidate, split = next(queue)
        assert split == 'validation' and candidate['town'] == 'Town06'
    finally:
        queue.close()
