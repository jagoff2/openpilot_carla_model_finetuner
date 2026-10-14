"""Sequential, bounded-memory archives for independently compressed NumPy frames."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 2
INDEX_MEMBER = "index.json"
MEMBER_TEMPLATE = "frames/{:06d}.npy"
ZIP_COMPRESSION = zipfile.ZIP_DEFLATED
ZIP_COMPRESSLEVEL = 1
ARCHIVE_BOUND_VERSION = 2
NPY_HEADER_BOUND = 4096
ZIP_MEMBER_STRUCTURE_BOUND = 512
INDEX_BASE_BOUND = 16384
INDEX_PER_FRAME_BOUND = 2048


def deflate_compress_bound(source_bytes: int) -> int:
    """Conservative zlib compressBound formula for one DEFLATE payload."""
    size = int(source_bytes)
    if size < 0:
        raise ValueError("source_bytes must be nonnegative")
    return size + (size >> 12) + (size >> 14) + (size >> 25) + 13


def frame_member_worst_case_bytes(frame_nbytes: int) -> int:
    payload_bound = int(frame_nbytes) + NPY_HEADER_BOUND
    return deflate_compress_bound(payload_bound) + ZIP_MEMBER_STRUCTURE_BOUND


def archive_worst_case_bytes(*, dtype: Any, frame_shape: tuple[int, ...],
                             expected_count: int) -> int:
    count = int(expected_count)
    if count < 0:
        raise ValueError("expected_count must be nonnegative")
    frame_nbytes = (int(np.prod(frame_shape, dtype=np.int64))
                    * np.dtype(dtype).itemsize)
    index_source_bound = INDEX_BASE_BOUND + count * INDEX_PER_FRAME_BOUND
    index_member_bound = (deflate_compress_bound(index_source_bound)
                          + ZIP_MEMBER_STRUCTURE_BOUND)
    return count * frame_member_worst_case_bytes(frame_nbytes) + index_member_bound


def _sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_bytes(frame: np.ndarray) -> memoryview:
    return memoryview(frame).cast("B")


class FrameArchiveWriter:
    """Write fixed-shape frames in strictly increasing integer-index order."""

    def __init__(self, path: str | Path, *, dtype: Any, frame_shape: tuple[int, ...],
                 expected_count: int, compresslevel: int = ZIP_COMPRESSLEVEL):
        self.path = Path(path).resolve()
        self.dtype = np.dtype(dtype)
        self.frame_shape = tuple(int(value) for value in frame_shape)
        self.expected_count = int(expected_count)
        if self.expected_count < 0 or not self.frame_shape:
            raise ValueError("expected_count and frame_shape must be valid")
        if self.path.exists():
            raise FileExistsError(f"immutable frame archive exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.compresslevel = int(compresslevel)
        self._entries: list[dict[str, Any]] = []
        self._closed = False
        self._complete: bool | None = None
        self._zip = zipfile.ZipFile(
            self.path, mode="x", compression=ZIP_COMPRESSION,
            compresslevel=self.compresslevel, allowZip64=True)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.expected_count, *self.frame_shape)

    @property
    def frame_nbytes(self) -> int:
        return int(np.prod(self.frame_shape, dtype=np.int64)) * self.dtype.itemsize

    def write(self, index: int, frame: np.ndarray) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("frame archive is closed")
        if int(index) != len(self._entries):
            raise ValueError(f"frame index must be sequential: expected {len(self._entries)}")
        if len(self._entries) >= self.expected_count:
            raise ValueError("frame archive exceeds expected_count")
        array = np.asarray(frame)
        if array.dtype != self.dtype or array.shape != self.frame_shape:
            raise ValueError(
                f"frame dtype/shape mismatch: {array.dtype} {array.shape}, "
                f"expected {self.dtype} {self.frame_shape}")
        contiguous = np.ascontiguousarray(array)
        decoded_sha256 = _sha256_bytes(_frame_bytes(contiguous))
        encoded = io.BytesIO()
        np.save(encoded, contiguous, allow_pickle=False)
        payload = encoded.getvalue()
        member = MEMBER_TEMPLATE.format(index)
        self._zip.writestr(member, payload)
        if self._zip.fp is not None:
            self._zip.fp.flush()
        entry = {
            "index": int(index), "member": member,
            "dtype": self.dtype.str, "shape": list(self.frame_shape),
            "decoded_nbytes": self.frame_nbytes,
            "decoded_sha256": decoded_sha256,
            "npy_member_bytes": len(payload),
            "npy_member_sha256": _sha256_bytes(payload),
        }
        self._entries.append(entry)
        return dict(entry)

    def remaining_worst_case_bytes(self) -> int:
        if self._closed:
            return 0
        remaining = self.expected_count - len(self._entries)
        index_source_bound = (INDEX_BASE_BOUND
                              + self.expected_count * INDEX_PER_FRAME_BOUND)
        index_member_bound = (deflate_compress_bound(index_source_bound)
                              + ZIP_MEMBER_STRUCTURE_BOUND)
        return (remaining * frame_member_worst_case_bytes(self.frame_nbytes)
                + index_member_bound)

    def finalize(self, *, complete: bool) -> dict[str, Any]:
        complete = bool(complete)
        if self._closed:
            if complete != self._complete:
                raise RuntimeError("frame archive was finalized with a different completion state")
            return self.summary()
        if complete and len(self._entries) != self.expected_count:
            raise RuntimeError(
                f"complete archive has {len(self._entries)} of {self.expected_count} frames")
        document = {
            "schema_version": SCHEMA_VERSION,
            "format": "independent-npy-members-in-zip",
            "compression": "ZIP_DEFLATED",
            "compresslevel": self.compresslevel,
            "dtype": self.dtype.str,
            "frame_shape": list(self.frame_shape),
            "expected_count": self.expected_count,
            "written_count": len(self._entries),
            "complete": complete,
            "frames": self._entries,
        }
        self._zip.writestr(
            INDEX_MEMBER,
            (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        self._zip.close()
        self._closed = True
        self._complete = complete
        return self.summary()

    def summary(self) -> dict[str, Any]:
        result = {
            "path": str(self.path), "format": "independent-npy-members-in-zip",
            "compression": "ZIP_DEFLATED", "compresslevel": self.compresslevel,
            "dtype": self.dtype.str, "frame_shape": list(self.frame_shape),
            "shape": list(self.shape), "expected_count": self.expected_count,
            "written_count": len(self._entries), "complete": self._complete,
            "archive_bound_version": ARCHIVE_BOUND_VERSION,
            "frames": [dict(entry) for entry in self._entries],
        }
        if self._closed:
            result.update(archive_bytes=self.path.stat().st_size,
                          archive_sha256=sha256_file(self.path))
        return result

    def close(self) -> None:
        if not self._closed:
            self.finalize(complete=False)

    def __enter__(self) -> "FrameArchiveWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class FrameArchiveReader:
    """Read and validate integer-indexed frames with a NumPy-like first axis."""

    def __init__(self, path: str | Path, *, require_complete: bool = True,
                 expected_archive_sha256: str | None = None):
        self.path = Path(path).resolve()
        if expected_archive_sha256 is not None:
            actual = sha256_file(self.path)
            if actual != expected_archive_sha256:
                raise ValueError("frame archive file hash mismatch")
        self._zip = zipfile.ZipFile(self.path, mode="r", allowZip64=True)
        try:
            names = self._zip.namelist()
            if names.count(INDEX_MEMBER) != 1:
                raise ValueError("frame archive must contain exactly one index.json")
            self.index = json.loads(self._zip.read(INDEX_MEMBER).decode("utf-8"))
            if self.index.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("unsupported frame archive schema")
            if (self.index.get("format") != "independent-npy-members-in-zip"
                    or self.index.get("compression") != "ZIP_DEFLATED"):
                raise ValueError("frame archive format/compression declaration mismatch")
            self.dtype = np.dtype(self.index["dtype"])
            self.frame_shape = tuple(int(value) for value in self.index["frame_shape"])
            self.expected_count = int(self.index["expected_count"])
            self._entries = list(self.index["frames"])
            self.complete = bool(self.index["complete"])
            if int(self.index["written_count"]) != len(self._entries):
                raise ValueError("frame archive written_count mismatch")
            if self.expected_count < 0 or len(self._entries) > self.expected_count:
                raise ValueError("frame archive count exceeds expected_count")
            if self.complete and len(self._entries) != self.expected_count:
                raise ValueError("complete frame archive count mismatch")
            if require_complete and (not self.complete
                                     or len(self._entries) != self.expected_count):
                raise ValueError("frame archive is incomplete")
            expected_names = [entry["member"] for entry in self._entries] + [INDEX_MEMBER]
            if len(names) != len(set(names)) or set(names) != set(expected_names):
                raise ValueError("frame archive member inventory mismatch")
            for index, entry in enumerate(self._entries):
                if int(entry["index"]) != index or entry["member"] != MEMBER_TEMPLATE.format(index):
                    raise ValueError("frame archive indexes are not consecutive")
                if (np.dtype(entry["dtype"]) != self.dtype
                        or tuple(entry["shape"]) != self.frame_shape):
                    raise ValueError("frame archive entry dtype/shape declaration mismatch")
                expected_nbytes = (int(np.prod(self.frame_shape, dtype=np.int64))
                                   * self.dtype.itemsize)
                if int(entry["decoded_nbytes"]) != expected_nbytes:
                    raise ValueError("frame archive entry decoded byte count mismatch")
                info = self._zip.getinfo(entry["member"])
                if info.compress_type != ZIP_COMPRESSION:
                    raise ValueError("frame archive member is not ZIP_DEFLATED")
        except Exception:
            self._zip.close()
            raise

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self), *self.frame_shape)

    def __getitem__(self, index: int) -> np.ndarray:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("FrameArchiveReader supports integer indexing only")
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("frame archive index out of range")
        entry = self._entries[index]
        payload = self._zip.read(entry["member"])
        if len(payload) != int(entry["npy_member_bytes"]):
            raise ValueError(f"frame {index} member length mismatch")
        if _sha256_bytes(payload) != entry["npy_member_sha256"]:
            raise ValueError(f"frame {index} member hash mismatch")
        stream = io.BytesIO(payload)
        array = np.load(stream, allow_pickle=False)
        if stream.tell() != len(payload):
            raise ValueError(f"frame {index} member has trailing bytes")
        if array.dtype != self.dtype or array.shape != self.frame_shape:
            raise ValueError(f"frame {index} decoded dtype/shape mismatch")
        contiguous = np.ascontiguousarray(array)
        if contiguous.nbytes != int(entry["decoded_nbytes"]):
            raise ValueError(f"frame {index} decoded byte length mismatch")
        if _sha256_bytes(_frame_bytes(contiguous)) != entry["decoded_sha256"]:
            raise ValueError(f"frame {index} decoded byte hash mismatch")
        return contiguous

    def validate(self) -> dict[str, Any]:
        for index in range(len(self)):
            self[index]
        return {
            "path": str(self.path), "archive_sha256": sha256_file(self.path),
            "archive_bytes": self.path.stat().st_size, "complete": self.complete,
            "expected_count": self.expected_count, "written_count": len(self),
            "dtype": self.dtype.str, "frame_shape": list(self.frame_shape),
            "all_frames_validated": True,
        }

    def close(self) -> None:
        self._zip.close()

    def __enter__(self) -> "FrameArchiveReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
