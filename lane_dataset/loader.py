"""Bounded-memory episode loader and student-generated recurrent histories."""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path

import numpy as np

from .adapter import lateral_action_target
from .preprocessing import PACKED_FRAME_SHAPE, RebelliousHopeState, reference_module

FrameArchiveReader = reference_module("frame_archive_v2").FrameArchiveReader
FRAME_DT = 0.05


def validate_frame_metadata(frames):
    if not frames:
        raise ValueError("episode must contain frames")
    ids, timestamps = [], []
    for index, frame in enumerate(frames):
        if frame.get("index") != index or not isinstance(frame.get("frame_id"), int):
            raise ValueError("frame indices must be consecutive and frame IDs integer")
        ids.append(frame["frame_id"])
        timestamps.append(frame["timestamp"])
        for key, length in (("desire", 8), ("traffic_convention", 2), ("action_t", 2)):
            value = np.asarray(frame[key], dtype=np.float64)
            if value.shape != (length,) or not np.isfinite(value).all():
                raise ValueError(f"invalid recorded {key}")
            if key == "action_t" and np.any(value < 0):
                raise ValueError("action horizons must be nonnegative")
            if key != "action_t" and (not np.isin(value, [0, 1]).all() or value.sum() != 1):
                raise ValueError(f"{key} must be a one-hot vector")
    if np.any(np.diff(ids) != 1):
        raise ValueError("missing, duplicate, or unordered simulator frame")
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if not np.isfinite(timestamps).all() or not np.allclose(np.diff(timestamps), FRAME_DT, atol=1e-6, rtol=0):
        raise ValueError("timestamps must match uninterrupted synchronous 20 Hz")
    return np.asarray(ids, dtype=np.int64), timestamps


class SequenceEpisode:
    """Read one accepted immutable shard; simulation metadata is never a model input.

    Manifest keys: schema_version=1, status='accepted', frames_file, labels_file,
    archives={narrow:{path,sha256},wide:{path,sha256}}, files={relative_path:sha256}.
    The file inventory must bind metadata, labels, and both archives. Hashes are
    verified before parsing frame metadata or loading target arrays. Archives contain one
    uint8[6,128,256] frame each. Label arrays preserve physical units; no latent
    teacher features are accepted or exposed.
    """
    _required_status = "accepted"
    _allow_unsupported_labels = False

    def __init__(self, manifest_path, *, require_retention=True, verify_hashes=True):
        self.path = Path(manifest_path).resolve()
        if self.path.is_dir():
            self.path = self.path / "manifest.json"
        self.root = self.path.parent
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1 or self.manifest.get("status") != self._required_status:
            raise ValueError(f"this loader requires a {self._required_status} schema-1 shard")
        if self._required_status == "development" and self.manifest.get("accepted_label_count") != 0:
            raise ValueError("diagnostic shards must explicitly declare zero accepted labels")
        self._verify_file_inventory(verify_hashes)
        self.frames = [json.loads(line) for line in self._path(self.manifest["frames_file"]).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.frame_ids, self.timestamps = validate_frame_metadata(self.frames)
        with np.load(self._path(self.manifest["labels_file"]), allow_pickle=False) as archive:
            self.labels = {key: archive[key] for key in archive.files}
        if any("feature" in key or "hidden" in key for key in self.labels):
            raise ValueError("dataset must not contain frozen recurrent teacher features")
        self._validate_labels(require_retention)
        self.archives = []
        try:
            for key in ("narrow", "wide"):
                entry = self.manifest["archives"][key]
                if verify_hashes and not entry.get("sha256"):
                    raise ValueError("archive hashes are required")
                reader = FrameArchiveReader(self._path(entry["path"]),
                    expected_archive_sha256=entry.get("sha256") if verify_hashes else None)
                self.archives.append(reader)
                if len(reader) != len(self.frames) or reader.dtype != np.uint8 or reader.frame_shape != PACKED_FRAME_SHAPE:
                    raise ValueError("packed archive frame count/dtype/shape mismatch")
        except Exception:
            self.close()
            raise

    def _path(self, relative):
        path = (self.root / relative).resolve()
        if self.root != path and self.root not in path.parents:
            raise ValueError("shard member escapes manifest directory")
        return path

    def _verify_file_inventory(self, verify_hashes):
        inventory = self.manifest.get("files")
        if not isinstance(inventory, dict) or not inventory:
            raise ValueError("accepted training shards require manifest.files hash bindings")
        bound = {}
        for relative, digest in inventory.items():
            if (not isinstance(relative, str) or not isinstance(digest, str) or
                len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)):
                raise ValueError("invalid manifest.files SHA-256 binding")
            path = self._path(relative)
            if path in bound:
                raise ValueError("duplicate normalized file path in manifest.files")
            if not path.is_file():
                raise ValueError(f"declared shard file is missing: {relative}")
            bound[path] = digest
        try:
            required = [self.manifest["frames_file"], self.manifest["labels_file"]]
            for camera in ("narrow", "wide"):
                entry = self.manifest["archives"][camera]
                required.append(entry["path"])
                if bound.get(self._path(entry["path"])) != entry.get("sha256"):
                    raise ValueError(f"archive hash disagrees with manifest.files: {camera}")
        except (KeyError, TypeError) as error:
            raise ValueError("incomplete shard file/archive declarations") from error
        if any(self._path(relative) not in bound for relative in required):
            raise ValueError("manifest.files must bind frame metadata, labels and both camera archives")
        if verify_hashes:
            digest_file = reference_module("frame_archive_v2").sha256_file
            for path, expected in bound.items():
                if digest_file(path) != expected:
                    raise ValueError(f"shard file hash mismatch: {path.relative_to(self.root)}")

    def _validate_labels(self, require_retention):
        labels = self.labels
        required = {"label_indices", "plan_targets", "plan_mask", "current_speed", "commanded_curvature", "lateral_mask"}
        if not required.issubset(labels):
            raise ValueError(f"missing canonical labels: {sorted(required - labels.keys())}")
        idx = labels["label_indices"]
        if idx.ndim != 1 or idx.dtype.kind not in "iu" or not len(idx) or np.any(np.diff(idx) <= 0) or idx[0] < 0 or idx[-1] >= len(self.frames):
            raise ValueError("label indices must be unique ordered in-range integers")
        if np.any(idx % 10):
            raise ValueError("accepted labels must lie on the episode 2 Hz candidate grid")
        count = len(idx)
        for key, shape in (("plan_targets", (count, 33, 15)), ("plan_mask", (count, 33, 15)),
                           ("current_speed", (count,)), ("commanded_curvature", (count,)), ("lateral_mask", (count,))):
            if labels[key].shape != shape:
                raise ValueError(f"invalid {key} shape")
        for key in ("plan_mask", "lateral_mask"):
            if labels[key].dtype != np.bool_:
                raise ValueError(f"{key} must be boolean")
        if not np.isfinite(labels["plan_targets"][labels["plan_mask"]]).all():
            raise ValueError("supported plan targets must be finite")
        encoded = lateral_action_target(labels["current_speed"], labels["commanded_curvature"])
        if "lateral_targets" in labels and (labels["lateral_targets"].shape != (count,) or
            not np.allclose(labels["lateral_targets"][labels["lateral_mask"]], encoded[labels["lateral_mask"]], rtol=1e-5, atol=1e-6)):
            raise ValueError("lateral targets disagree with current speed and expert commanded curvature")
        labels["lateral_targets"] = encoded.astype(np.float32)
        if ("longitudinal_targets" in labels) != ("longitudinal_mask" in labels):
            raise ValueError("longitudinal target/mask must both be present or absent")
        if "longitudinal_targets" in labels:
            if labels["longitudinal_targets"].shape != (count,) or labels["longitudinal_mask"].shape != (count,) or labels["longitudinal_mask"].dtype != np.bool_:
                raise ValueError("invalid longitudinal target/mask")
            if not np.isfinite(labels["longitudinal_targets"][labels["longitudinal_mask"]]).all():
                raise ValueError("supported longitudinal targets must be finite")
        supported = labels["plan_mask"].any(axis=(1, 2)) | labels["lateral_mask"]
        if "longitudinal_mask" in labels:
            supported |= labels["longitudinal_mask"]
        if not supported.all() and not self._allow_unsupported_labels:
            raise ValueError("accepted labels must have supported supervision; ambiguous frames belong in diagnostics")
        if require_retention and (self.timestamps[idx[0]] - self.timestamps[0] < 5.0 - 1e-6 or
                                  self.timestamps[-1] - self.timestamps[idx[-1]] < 10.0 - 1e-6):
            raise ValueError("accepted sequences require five seconds of history and ten seconds of future")

    def __len__(self):
        return len(self.frames)

    def frame(self, index):
        return np.stack([archive[index] for archive in self.archives])

    def iter_frames(self):
        for index, metadata in enumerate(self.frames):
            yield self.frame(index), metadata

    def __iter__(self):
        return self.iter_frames()

    def selected_targets(self):
        """Physical targets and recorded action horizons; never privileged metadata."""
        keys = ("label_indices", "plan_targets", "plan_mask", "lateral_targets", "lateral_mask",
                "longitudinal_targets", "longitudinal_mask")
        selected = {key: value for key, value in self.labels.items() if key in keys}
        selected["action_t"] = np.asarray([self.frames[int(i)]["action_t"] for i in self.labels["label_indices"]], np.float32)
        return selected

    def validate_all_frames(self):
        return [archive.validate() for archive in self.archives]

    def close(self):
        for archive in getattr(self, "archives", []):
            archive.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class DiagnosticSequence(SequenceEpisode):
    """Explicit development-only inspection, never a training acceptance path.

    Preserves archive/file hashes, continuity, retention, target encoding, and
    mask checks. Unsupported rows remain masked and may be present for auditing.
    The manifest must remain development with accepted_label_count exactly zero.
    """
    _required_status = "development"
    _allow_unsupported_labels = True


def student_rollout(episode, student, *, device="cpu", auxiliary_dtype=None,
                    detach_history=True, stop_after_last_label=True):
    """Forward dense ticks and return labelled student predictions plus targets.

    `student` accepts a dictionary containing the six native contract keys and
    returns torch[1,2580]. Every recurrent value is freshly generated by this
    student. detach_history=True performs truncated recurrence, without changing
    values or using teacher features. False retains the recurrent autograd graph.
    No optimizer is created and no model parameter is updated.
    """
    import torch
    dtype = auxiliary_dtype or torch.float32
    numpy_state = RebelliousHopeState()
    zero = torch.zeros(512, device=device, dtype=dtype)
    feature_queue = deque([zero] * 96, maxlen=96)
    previous_feature = zero
    labels = set(int(i) for i in episode.labels["label_indices"])
    last = max(labels) if stop_after_last_label else len(episode) - 1
    chosen_outputs = []
    for index in range(last + 1):
        metadata = episode.frames[index]
        packed = episode.frame(index)
        native = numpy_state.push(packed, metadata["desire"], metadata["traffic_convention"], metadata["action_t"], native=False)
        feature_queue.append(previous_feature)
        inputs = {key: torch.as_tensor(value, device=device, dtype=torch.uint8 if key in ("img", "big_img") else dtype)
                  for key, value in native.items() if key != "features_buffer"}
        inputs["features_buffer"] = torch.stack(list(feature_queue)[::4])[None]
        with torch.set_grad_enabled(not detach_history or index in labels):
            output = student(inputs)
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != (1, 2580):
            raise ValueError("student must return torch[1,2580]")
        hidden = output[0, 2066:2578]
        if not bool(torch.isfinite(hidden).all()):
            raise ValueError("student hidden state is nonfinite")
        previous_feature = hidden.detach() if detach_history else hidden
        if index in labels:
            chosen_outputs.append(output[0])
    return torch.stack(chosen_outputs), episode.selected_targets()
