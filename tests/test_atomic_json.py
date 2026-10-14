import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import pytest

from lane_dataset import config


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_retries_transient_windows_replace_failures(tmp_path, monkeypatch, winerror):
    target = tmp_path / "status.json"
    config.write_json(target, {"generation": 1})
    original = Path.replace
    attempts = []

    def locked_replace(source, destination):
        attempts.append(source)
        if len(attempts) <= 2:
            assert json.loads(target.read_text()) == {"generation": 1}
            error = PermissionError("reader holds the destination")
            error.winerror = winerror
            raise error
        return original(source, destination)

    monkeypatch.setattr(Path, "replace", locked_replace)
    monkeypatch.setattr(config.time, "sleep", lambda _: None)
    config.write_json(target, {"generation": 2})
    assert len(attempts) == 3
    assert json.loads(target.read_text()) == {"generation": 2}
    assert not list(tmp_path.glob("*.tmp"))


def test_simultaneous_writers_keep_separate_complete_documents(tmp_path, monkeypatch):
    target = tmp_path / "status.json"
    original = Path.replace
    barrier = threading.Barrier(4)
    staged = []
    seen = set()
    guard = threading.Lock()

    def simultaneous_replace(source, destination):
        with guard:
            first_attempt = source not in seen
            seen.add(source)
            if first_attempt:
                staged.append((source, json.loads(source.read_text())))
        if first_attempt:
            barrier.wait(timeout=10)
        return original(source, destination)

    monkeypatch.setattr(Path, "replace", simultaneous_replace)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: config.write_json(target, {"writer": i}), range(4)))
    assert len({path for path, _ in staged}) == 4
    assert {doc["writer"] for _, doc in staged} == set(range(4))
    assert json.loads(target.read_text())["writer"] in range(4)
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_serialization_preserves_previous_document_and_cleans_temp(tmp_path):
    target = tmp_path / "status.json"
    config.write_json(target, {"valid": True})
    with pytest.raises(ValueError):
        config.write_json(target, {"invalid": float("nan")})
    assert json.loads(target.read_text()) == {"valid": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_persistent_lock_is_bounded_and_preserves_previous_document(tmp_path, monkeypatch):
    target = tmp_path / "status.json"
    config.write_json(target, {"generation": 1})
    replace = config._replace_json_file

    def locked_replace(*_):
        error = PermissionError("persistent Windows lock")
        error.winerror = 5
        raise error

    monkeypatch.setattr(Path, "replace", locked_replace)
    monkeypatch.setattr(config, "_replace_json_file",
                        lambda source, dest: replace(source, dest, timeout=0))
    with pytest.raises(PermissionError):
        config.write_json(target, {"generation": 2})
    assert json.loads(target.read_text()) == {"generation": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_unrelated_io_error_is_not_retried(tmp_path, monkeypatch):
    target = tmp_path / "status.json"

    def disk_failure(*_):
        raise OSError(28, "No space left")

    monkeypatch.setattr(Path, "replace", disk_failure)
    monkeypatch.setattr(config.time, "sleep", lambda _: pytest.fail("must not retry"))
    with pytest.raises(OSError, match="No space left"):
        config.write_json(target, {"generation": 1})
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.skipif(os.name != "nt", reason="Windows destination handle semantics")
def test_real_windows_reader_lock_is_retried(tmp_path, monkeypatch):
    target = tmp_path / "status.json"
    config.write_json(target, {"generation": 1})
    retries = []
    with target.open("r", encoding="utf-8") as reader:
        def release_reader(delay):
            retries.append(delay)
            assert json.load(reader) == {"generation": 1}
            reader.close()

        monkeypatch.setattr(config.time, "sleep", release_reader)
        config.write_json(target, {"generation": 2})
    assert retries
    assert json.loads(target.read_text()) == {"generation": 2}
