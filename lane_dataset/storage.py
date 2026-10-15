"""Transactional cross-process reservations, immutable publication, bounded retries."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
import uuid

from .config import Config, sha256_file, write_json


def rename_directory(source, target, timeout=10.):
    """Retry Windows sharing violations without replacing an existing shard."""
    source,target=Path(source),Path(target)
    deadline=time.monotonic()+timeout
    delay=.02
    while True:
        if target.exists():raise FileExistsError(target)
        try:
            source.rename(target)
            return
        except OSError as error:
            remaining=deadline-time.monotonic()
            if getattr(error,'winerror',None) not in (5,32,33) or remaining<=0:raise
            time.sleep(min(delay,remaining));delay=min(.25,delay*2)


class Store:
    def __init__(self, root, config=None):
        self.root = Path(root).resolve()
        self.config = config or Config()
        self.root.mkdir(parents=True, exist_ok=True)
        for subdir in ("accepted", "staging", "quarantine", "diagnostics", "reports"):
            (self.root / subdir).mkdir(exist_ok=True)
        self.database = self.root / "ledger.sqlite"
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, episode TEXT, attempt INTEGER, worker INTEGER, pid INTEGER, status TEXT, reserved INTEGER, path TEXT, created REAL, reason TEXT)")
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS one_accepted ON runs(episode) WHERE status='accepted'")
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS one_active ON runs(episode) WHERE status='active'")

    def connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.execute("PRAGMA busy_timeout=30000")
        db.row_factory = sqlite3.Row
        return db

    @contextmanager
    def transaction(self):
        db = self.connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def directory_bytes(self):
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    def _snapshot(self, db, additional=0):
        rows = db.execute("SELECT reserved,path FROM runs WHERE status='active'").fetchall()
        # Already written bytes count toward used, not twice toward reservations.
        remaining = sum(max(0, r["reserved"] - sum(p.stat().st_size for p in Path(r["path"]).rglob("*") if p.is_file())) for r in rows)
        used = self.directory_bytes()
        free = shutil.disk_usage(self.root).free
        return dict(used_bytes=used, free_bytes=free, in_flight_remaining_bytes=remaining,
                    projected_bytes=used + remaining + additional,
                    projected_free_bytes=free - remaining - additional,
                    cap_bytes=self.config.corpus_cap_bytes, minimum_free_bytes=self.config.minimum_free_bytes)

    def snapshot(self):
        with self.transaction() as db:
            return self._snapshot(db)

    def reserve(self, episode, worker, worst_case_bytes, *, retry_policy=None):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}", episode):
            raise ValueError("unsafe episode identifier")
        if worst_case_bytes <= 0:
            raise ValueError("reservation must be positive")
        if retry_policy is not None and (type(retry_policy) is not int or retry_policy<1):
            raise ValueError('retry policy must be a positive collector revision')
        with self.transaction() as db:
            existing = db.execute("SELECT * FROM runs WHERE episode=?", (episode,)).fetchall()
            if (self.root / "accepted" / episode).exists() or any(r["status"] in ("active", "accepted") for r in existing):
                raise FileExistsError(f"episode already active or accepted: {episode}")
            attempts=existing
            if retry_policy is not None:
                attempts=[]
                for prior in existing:
                    directory=Path(prior['path'])
                    try:
                        reservation=json.loads((directory/'reservation.json').read_text())
                        policy=reservation.get('retry_policy')
                        if policy is None:
                            manifest=directory/'manifest.json'
                            policy=json.loads(manifest.read_text()).get('capture_policy_version',0) if manifest.exists() else 0
                    except (OSError,ValueError):policy=retry_policy  # Corruption cannot grant a free retry.
                    if policy==retry_policy:attempts.append(prior)
            if len(attempts) >= self.config.max_attempts:
                raise RuntimeError("bounded retry limit reached")
            snap = self._snapshot(db, worst_case_bytes)
            if snap["projected_bytes"] > self.config.corpus_cap_bytes or snap["projected_free_bytes"] < self.config.minimum_free_bytes:
                raise OSError(f"storage reservation rejected: {snap}")
            token = f"{episode}-a{len(existing)+1}-{uuid.uuid4().hex[:10]}"
            stage = self.root / "staging" / token
            stage.mkdir()
            db.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,?)", (token, episode, len(existing)+1, worker, os.getpid(), "active", int(worst_case_bytes), str(stage), time.time(), None))
            write_json(stage / "reservation.json", dict(id=token, episode=episode, worker=worker, pid=os.getpid(), reserved_bytes=worst_case_bytes, storage=snap,retry_policy=retry_policy))
        return token, stage

    def check(self, token):
        with self.transaction() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (token,)).fetchone()
            if row is None or row["status"] != "active":
                raise RuntimeError("reservation not active")
            size = sum(p.stat().st_size for p in Path(row["path"]).rglob("*") if p.is_file())
            snap = self._snapshot(db)
            if size > row["reserved"] or snap["projected_bytes"] > self.config.corpus_cap_bytes or snap["projected_free_bytes"] < self.config.minimum_free_bytes:
                raise OSError("storage guard exceeded")
            return snap

    def finish(self, token, accepted, reason=None):
        if accepted:
            raise ValueError("accepted publication requires publish_qualified and full admission replay")
        with self.transaction() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (token,)).fetchone()
            if row is None or row["status"] != "active":
                raise RuntimeError("only active reservations may be completed")
            return self._quarantine(db, row, reason)

    def _quarantine(self, db, row, reason):
        stage = Path(row["path"]).resolve()
        if stage.parent != self.root / "staging":
            raise ValueError("staging escaped corpus root")
        target = self.root / "quarantine" / row["id"]
        if target.exists():
            raise FileExistsError(target)
        write_json(stage / "disposition.json", dict(status="quarantined", reason=reason))
        rename_directory(stage,target)  # Same-volume atomic visibility to readers.
        db.execute("UPDATE runs SET status='quarantined',reserved=0,path=?,reason=? WHERE id=?", (str(target), reason, row["id"]))
        return target

    def _verify_renamed_destination(self, row, target, status):
        reservation = json.loads((target / "reservation.json").read_text())
        if reservation.get("id") != row["id"] or reservation.get("episode") != row["episode"]:
            raise ValueError("recovery destination reservation identity mismatch")
        disposition = json.loads((target / "disposition.json").read_text())
        if disposition.get("status") != status:
            raise ValueError("recovery destination disposition mismatch")
        if status == "accepted":
            manifest = json.loads((target / "manifest.json").read_text())
            if (manifest.get("status") != "accepted" or not manifest.get("qa", {}).get("passed")
                    or not manifest.get("qualification_binding") or not manifest.get("files")):
                raise ValueError("recovery destination lacks qualified publication metadata")
            for relative, digest in manifest["files"].items():
                file = (target / relative).resolve()
                if target not in file.parents or sha256_file(file) != digest:
                    raise ValueError("recovery destination shard hash mismatch")

    def recover(self):
        recovered = []
        # Holding the writer lock also prevents an uncommitted reserve() directory
        # from being mistaken for an orphan and serializes competing recoveries.
        with self.transaction() as db:
            rows = db.execute("SELECT * FROM runs WHERE status IN ('active','quarantined')").fetchall()
            for row in rows:
                # A crash after rename but before commit is reconciled only to
                # the destination carrying this exact reservation's identity.
                accepted = self.root / "accepted" / row["episode"]
                quarantined = self.root / "quarantine" / row["id"]
                if not Path(row["path"]).exists() and (accepted.exists() or quarantined.exists()):
                    target = quarantined if quarantined.exists() else accepted
                    status = "quarantined" if target == quarantined else "accepted"
                    self._verify_renamed_destination(row, target, status)
                    db.execute("UPDATE runs SET status=?,reserved=0,path=? WHERE id=?", (status, str(target), row["id"]))
                    recovered.append(row["id"])
                    continue
                if row["status"] != "active":
                    continue
                try:
                    import psutil
                    process = psutil.Process(row["pid"])
                    alive = process.is_running() and process.create_time() <= row["created"]
                except (ImportError, OSError):
                    # Without reliable process identity do not reclaim another writer.
                    alive = True
                except Exception as error:
                    if error.__class__.__name__ != "NoSuchProcess":
                        raise
                    alive = False
                if not alive:
                    self._quarantine(db, row, "interrupted_worker")
                    recovered.append(row["id"])
            owned = {Path(row[0]).resolve() for row in db.execute("SELECT path FROM runs")}
            for stage in (self.root / "staging").iterdir():
                if stage.is_dir() and stage.resolve() not in owned:
                    target = self.root / "quarantine" / ("orphan-" + stage.name)
                    if target.exists():
                        target = target.with_name(target.name + "-" + uuid.uuid4().hex[:10])
                    write_json(stage / "disposition.json", dict(status="quarantined", reason="uncommitted_reservation"))
                    rename_directory(stage,target)
                    recovered.append(target.name)
        return recovered

    def rows(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM runs ORDER BY created")]

    def publish_qualified(self, shard):
        """Revalidate a quarantined completed candidate before atomic publication."""
        from .postprocess import shard_postprocess_lock
        with shard_postprocess_lock(shard):
            return self._publish_qualified_locked(shard)

    def _publish_qualified_locked(self, shard):
        from .admission import _qualify_candidate_unlocked
        shard=Path(shard).resolve()
        if shard.parent!=self.root/"quarantine":
            raise ValueError("only owned quarantined candidates may be promoted")
        report=_qualify_candidate_unlocked(shard,dataset_root=self.root)
        if not report["passed"]:
            raise ValueError(f"candidate admission failed: {report['errors']}")
        with self.transaction() as db:
            row=db.execute("SELECT * FROM runs WHERE path=? AND status='quarantined'",(str(shard),)).fetchone()
            if row is None: raise RuntimeError("candidate ledger entry missing")
            target=self.root/"accepted"/row["episode"]
            if target.exists(): raise FileExistsError(target)
            for name,digest in report["artifact_hashes"].items():
                file=(shard/name).resolve()
                if shard not in file.parents or sha256_file(file)!=digest:
                    raise ValueError("candidate changed during publication")
            for binding in report["bindings"].values():
                if sha256_file(binding["path"])!=binding["sha256"]:
                    raise ValueError("qualification evidence changed during publication")
            manifest=json.loads((shard/"manifest.json").read_text())
            manifest.update(status="accepted",accepted_label_count=20,qualification_binding=report["bindings"],qa=dict(passed=True,physical=report["physical"]))
            for record in manifest["label_records"]:
                record.update(accepted=True,qa_passed=True,reviewed=True)
                if manifest.get('automatic_review_file'):
                    record.update(review_mode='automated_measured_v1',human_reviewed=False)
            write_json(shard/"manifest.json",manifest)
            write_json(shard/"disposition.json",dict(status="accepted",reason="all_bound_qualification_gates_passed"))
            rename_directory(shard,target)
            db.execute("UPDATE runs SET status='accepted',reserved=0,path=?,reason=NULL WHERE id=?",(str(target),row["id"]))
        return target
