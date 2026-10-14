"""Crash-released, nonblocking serialization of completed-shard writers."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path


def _assert_mutable(shard):
    manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") == "accepted":
        raise FileExistsError("accepted shards are immutable")


@contextmanager
def shard_postprocess_lock(shard_path):
    """Fail immediately on another writer; OS ownership disappears on a crash.

    The persistent lock file is outside the shard and is never unlinked, so a
    waiter cannot accidentally lock an obsolete file inode. Reentrancy is
    intentionally unsupported; compound writers use explicit unlocked helpers.
    """
    shard = Path(shard_path).resolve()
    _assert_mutable(shard)
    lock_root = shard.parent / ".postprocess-locks"
    lock_root.mkdir(exist_ok=True)
    key = hashlib.sha256(os.path.normcase(str(shard)).encode()).hexdigest()
    with (lock_root / (key + ".lock")).open("a+b") as stream:
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"postprocessing already active for shard: {shard}") from error
        try:
            _assert_mutable(shard)
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
