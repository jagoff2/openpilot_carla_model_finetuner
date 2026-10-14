from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import json
from .settings import path_setting

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = path_setting('dataset_root','LANE_DATASET_ROOT',PROJECT/'data/corpus')
FAMILIES = ("straight_recovery", "constant_radius", "tightening_s_bend", "severe_hairpin", "intersection_turn")


@dataclass(frozen=True)
class Config:
    schema_version: int = 1
    dataset_version: str = "lane-centering-dataset-v1"
    seed: int = 20261013
    carla_version: str = "0.9.16"
    carla_root: str = str(path_setting('carla_root','CARLA_ROOT',PROJECT/'external/CARLA_0.9.16'))
    dependency_root: str = str(PROJECT/'lane_reference')
    model_path: str = str(PROJECT/'models/cd210/driving_policy.onnx')
    model_sha256: str = "78477124cbf3ffe30fa951ebada8410b43c4242c6054584d656f1d329b067e15"
    dt: float = 0.05
    max_substep_delta_time: float = 0.003125
    max_substeps: int = 16
    labels_per_episode: int = 20
    candidate_stride: int = 10
    prefix_ticks: int = 100
    suffix_ticks: int = 200
    corpus_cap_bytes: int = 60 * 1024**3
    minimum_free_bytes: int = 30 * 1024**3
    max_attempts: int = 3
    workers: tuple = ((0, 2400, 8400), (1, 2500, 8500))
    targets: dict = field(default_factory=lambda: {"train": 4000, "validation": 500, "test": 500})
    camera_optics_status: str = "assumed_physical_optics_not_device_calibrated"
    crops: str = "stock_MED_and_SBIG_only"
    depth_source: str = "aligned_depth_sensors_pending_known_geometry_validation"
    quality: str = "Epic"

    def document(self):
        return asdict(self)

    def digest(self):
        return hashlib.sha256(json.dumps(self.document(), sort_keys=True).encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        import os
        os.fsync(stream.fileno())
    tmp.replace(path)
