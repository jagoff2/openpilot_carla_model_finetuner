import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor
import pytest
from lane_dataset.config import Config, sha256_file, write_json
from lane_dataset.storage import Store


def test_reservations_atomic_across_workers(tmp_path):
    config = dataclasses.replace(Config(), corpus_cap_bytes=400000, minimum_free_bytes=0)
    store = Store(tmp_path, config)
    def reserve(i):
        try:
            return store.reserve(f"episode{i}", i, 250000)
        except OSError:
            return None
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(reserve, [0, 1]))
    assert sum(r is not None for r in results) == 1


def test_reject_unqualified_no_overwrite_bounded_retry(tmp_path):
    store = Store(tmp_path, dataclasses.replace(Config(), minimum_free_bytes=0))
    token, path = store.reserve("episode", 0, 1000000)
    write_json(path / "manifest.json", {"status": "accepted", "qa": {"passed": False}})
    with pytest.raises(ValueError):
        store.finish(token, True)
    store.finish(token, False, "failed gate")
    for _ in range(2):
        token, _ = store.reserve("episode", 1, 1000000)
        store.finish(token, False, "failed gate")
    with pytest.raises(RuntimeError):
        store.reserve("episode", 0, 1000000)


def test_new_collector_revision_gets_bounded_retries_and_preserves_old_attempts(tmp_path):
    store=Store(tmp_path,dataclasses.replace(Config(),minimum_free_bytes=0))
    paths=[]
    for _ in range(3):
        token,path=store.reserve('episode',0,1000000,retry_policy=1)
        paths.append(store.finish(token,False,'old collector failure'))
    with pytest.raises(RuntimeError,match='bounded'):store.reserve('episode',0,1000000,retry_policy=1)
    token,path=store.reserve('episode',0,1000000,retry_policy=2)
    assert '-a4-' in token and all(p.is_dir() for p in paths)
    store.finish(token,False,'new attempt retained')


def test_publish_hashes_and_crash_recovery(tmp_path):
    store = Store(tmp_path, dataclasses.replace(Config(), minimum_free_bytes=0))
    token, path = store.reserve("episode", 0, 1000000)
    (path / "payload").write_bytes(b"corpus")
    write_json(path / "manifest.json", {"status":"accepted", "qa":{"passed":True}, "qualification_binding":"test-fixture", "files":{"payload":sha256_file(path / "payload")}})
    with pytest.raises(ValueError,match="full admission replay"):
        store.finish(token, True)
    target = store.finish(token, False, "unit fixture cannot qualify corpus")
    assert json.loads((target / "disposition.json").read_text())["status"] == "quarantined"
    token2, path2 = store.reserve("interrupted", 1, 1000000)
    with store.connect() as db:
        db.execute("UPDATE runs SET pid=2147483647 WHERE id=?", (token2,))
    assert store.recover() == [token2]
    assert not path2.exists()


def test_recovery_binds_rename_to_its_reservation(tmp_path):
    store = Store(tmp_path, dataclasses.replace(Config(), minimum_free_bytes=0))
    token, stage = store.reserve("episode", 0, 1000000)
    target = tmp_path / "quarantine" / token
    with pytest.raises(RuntimeError, match="simulated crash"):
        with store.transaction():
            write_json(stage / "disposition.json", dict(status="quarantined"))
            stage.rename(target)
            raise RuntimeError("simulated crash after rename before database commit")
    assert store.recover() == [token]
    assert store.rows()[0]["status"] == "quarantined"
    assert store.rows()[0]["reserved"] == 0
    assert store.recover() == []


def test_recovery_never_promotes_an_unrelated_accepted_attempt(tmp_path):
    store = Store(tmp_path, dataclasses.replace(Config(), minimum_free_bytes=0))
    token, stage = store.reserve("episode", 0, 1000000)
    stage.rename(tmp_path / "preserved-stage")
    accepted = tmp_path / "accepted" / "episode"
    accepted.mkdir()
    write_json(accepted / "reservation.json", dict(id="another-attempt", episode="episode"))
    with pytest.raises(ValueError, match="reservation identity mismatch"):
        store.recover()
    assert store.rows()[0]["id"] == token
    assert store.rows()[0]["status"] == "active"
    assert json.loads((accepted / "reservation.json").read_text())["id"] == "another-attempt"


def test_recovery_verifies_accepted_shard_hashes(tmp_path):
    store = Store(tmp_path, dataclasses.replace(Config(), minimum_free_bytes=0))
    token, stage = store.reserve("episode", 0, 1000000)
    (stage / "payload").write_bytes(b"original")
    write_json(stage / "manifest.json", dict(status="accepted", qa=dict(passed=True),
        qualification_binding={"unit_fixture": True}, files={"payload": sha256_file(stage / "payload")}))
    write_json(stage / "disposition.json", dict(status="accepted"))
    accepted = tmp_path / "accepted" / "episode"
    stage.rename(accepted)
    (accepted / "payload").write_bytes(b"changed")
    with pytest.raises(ValueError, match="shard hash mismatch"):
        store.recover()
    assert store.rows()[0]["status"] == "active"
    (accepted / "payload").write_bytes(b"original")
    assert store.recover() == [token]
    assert store.rows()[0]["status"] == "accepted"
    with pytest.raises(FileExistsError):
        store.reserve("episode", 1, 1000000)


def test_orphaned_reserve_is_quarantined_and_live_writer_preserved(tmp_path, monkeypatch):
    from lane_dataset import storage
    store = Store(tmp_path, dataclasses.replace(Config(), minimum_free_bytes=0))
    original = storage.write_json
    def fail_after_write(path, value):
        original(path, value)
        raise OSError("simulated reservation failure")
    monkeypatch.setattr(storage, "write_json", fail_after_write)
    with pytest.raises(OSError, match="reservation failure"):
        store.reserve("orphan", 0, 1000000)
    monkeypatch.setattr(storage, "write_json", original)
    _, live = store.reserve("live", 1, 1000000)
    recovered = store.recover()
    assert len(recovered) == 1 and recovered[0].startswith("orphan-orphan-a1-")
    assert live.exists()
    assert len(list((tmp_path / "staging").iterdir())) == 1
    assert store.rows()[0]["episode"] == "live"


def test_concurrent_recovery_reclaims_dead_worker_once(tmp_path):
    store = Store(tmp_path, dataclasses.replace(Config(), minimum_free_bytes=0))
    token, _ = store.reserve("interrupted", 0, 1000000)
    with store.connect() as db:
        db.execute("UPDATE runs SET pid=2147483647 WHERE id=?", (token,))
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: store.recover(), range(2)))
    assert sorted(results, key=len) == [[], [token]]


def test_accepted_directory_without_ledger_is_never_reused(tmp_path):
    store = Store(tmp_path, dataclasses.replace(Config(), minimum_free_bytes=0))
    (tmp_path / "accepted" / "episode").mkdir()
    with pytest.raises(FileExistsError):
        store.reserve("episode", 0, 1000000)
